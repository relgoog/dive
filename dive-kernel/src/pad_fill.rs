//! Pad, fill, one-hot and cumulative-sum kernels.

use alloc::{
   vec,
   vec::Vec,
};

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::tensor::{
   ResolvedTensor,
   Shape,
   TensorType,
   extent,
   type_size_bytes,
};

pub struct ContinuousSlice {
   pub inner_bytes: usize,
   pub chunk_bytes: usize,
   pub axis:        usize,
}

impl ContinuousSlice {
   /// Empty slice marker.
   const fn empty() -> Self {
      Self {
         inner_bytes: 0,
         chunk_bytes: 0,
         axis:        0,
      }
   }
}

/// Row-major element strides for dims.
fn row_strides(dims: &[i64]) -> Vec<usize> {
   let mut strides = vec![1_usize; dims.len()];
   let mut step = 1_usize;
   for idx in (0..dims.len()).rev() {
      strides[idx] = step;
      step = step.saturating_mul(extent(dims[idx]));
   }
   strides
}

/// Repeat one element pattern across every output element.
fn splat_bytes(out: &mut [u8], seed: &[u8], elem: usize) {
   if out.is_empty() || elem == 0 {
      return;
   }
   for idx in 0..elem.min(out.len()) {
      out[idx] = if seed.is_empty() {
         0
      } else {
         seed[idx % seed.len()]
      };
   }
   let count = out.len() / elem;
   let mut filled = 1_usize;
   while filled < count {
      let chunk = (count - filled).min(filled);
      let (head, tail) = out.split_at_mut(filled * elem);
      tail[..chunk * elem].copy_from_slice(&head[..chunk * elem]);
      filled += chunk;
   }
}

/// Decode a little-endian signed integer.
///
/// None on an unsupported width or an out-of-range read.
fn read_padding_int(raw: &[u8], at: usize, size: usize) -> Option<i64> {
   if at.saturating_add(size) > raw.len() {
      return None;
   }
   let cell = &raw[at..];
   match size {
      1 => Some(i64::from(cell[0].cast_signed())),
      2 => Some(i64::from(i16::from_le_bytes(dive::lane::<2>(cell)))),
      4 => Some(i64::from(i32::from_le_bytes(dive::lane::<4>(cell)))),
      8 => Some(i64::from_le_bytes(dive::lane::<8>(cell))),
      _ => None,
   }
}

/// Trailing run of equal dims between input and padded shape, in bytes.
#[inline]
#[must_use]
pub fn pad_trailing_run(input: &ResolvedTensor, out_shape: &Shape) -> ContinuousSlice {
   let elem = type_size_bytes(input.tensor_type);
   if elem == 0 {
      return ContinuousSlice::empty();
   }
   let in_dims = &input.shape.dims;
   let out_dims = &out_shape.dims;

   let mut inner = elem;
   let mut trailing = 0_usize;
   while trailing < in_dims.len().min(out_dims.len())
      && in_dims[in_dims.len() - 1 - trailing] == out_dims[out_dims.len() - 1 - trailing]
   {
      inner = inner.saturating_mul(extent(in_dims[in_dims.len() - 1 - trailing]));
      trailing += 1;
   }
   let axis = in_dims.len().saturating_sub(trailing);
   let chunk = if axis == 0 {
      inner
   } else {
      extent(in_dims[axis - 1]).saturating_mul(inner)
   };
   ContinuousSlice {
      inner_bytes: inner,
      chunk_bytes: chunk,
      axis,
   }
}

/// Pad input into output, writing constant bytes outside the copied window.
#[inline]
pub fn pad(
   input: &ResolvedTensor,
   paddings: &ResolvedTensor,
   constant: &ResolvedTensor,
   output: &mut ResolvedTensor,
) -> Status {
   let elem = type_size_bytes(input.tensor_type);
   if elem == 0 || output.element_count == 0 {
      return Status::ok();
   }
   let in_dims = input.shape.dims.clone();
   let rank = in_dims.len();
   let out_len = output.element_count.saturating_mul(elem);

   let pair_size = type_size_bytes(paddings.tensor_type);
   let raw = paddings.bytes();
   let mut before = vec![0_usize; rank];
   let mut after = vec![0_usize; rank];
   for dim_idx in 0..rank {
      before[dim_idx] =
         extent(read_padding_int(raw, dim_idx * 2 * pair_size, pair_size).unwrap_or(0));
      after[dim_idx] =
         extent(read_padding_int(raw, (dim_idx * 2 + 1) * pair_size, pair_size).unwrap_or(0));
   }

   let src = input.bytes().to_vec();
   if before.iter().chain(after.iter()).all(|bound| *bound == 0) && src.len() == out_len {
      output.bytes_mut()[..out_len].copy_from_slice(&src[..out_len]);
      return Status::ok();
   }

   let seed = constant.bytes().to_vec();
   let span = pad_trailing_run(input, &output.shape);
   let out_strides = row_strides(&output.shape.dims);
   let in_strides = row_strides(&in_dims);
   let loop_rank = span.axis.min(rank);
   let bounds = in_dims[..loop_rank]
      .iter()
      .map(|dim| extent(*dim))
      .collect::<Vec<usize>>();
   let cells = bounds
      .iter()
      .fold(1_usize, |acc, bound| acc.saturating_mul(*bound));

   let dst = output.bytes_mut();
   if elem == 1 {
      dst.fill(seed.first().copied().unwrap_or(0));
   } else {
      splat_bytes(dst, &seed, elem);
   }
   if span.inner_bytes == 0 || cells == 0 {
      return Status::ok();
   }

   let mut coord = vec![0_usize; loop_rank];
   for _ in 0..cells {
      let mut in_base = 0_usize;
      let mut out_base = 0_usize;
      for dim_idx in 0..loop_rank {
         in_base = in_base.saturating_add(coord[dim_idx].saturating_mul(in_strides[dim_idx]));
         out_base = out_base.saturating_add(
            coord[dim_idx]
               .saturating_add(before[dim_idx])
               .saturating_mul(out_strides[dim_idx]),
         );
      }
      let from = in_base.saturating_mul(elem);
      let to = out_base.saturating_mul(elem);
      if from.saturating_add(span.inner_bytes) <= src.len()
         && to.saturating_add(span.inner_bytes) <= dst.len()
      {
         dst[to..to + span.inner_bytes].copy_from_slice(&src[from..from + span.inner_bytes]);
      }
      for dim_idx in (0..loop_rank).rev() {
         coord[dim_idx] += 1;
         if coord[dim_idx] < bounds[dim_idx] {
            break;
         }
         coord[dim_idx] = 0;
      }
   }
   Status::ok()
}

/// Broadcast one element value across the whole output.
#[inline]
pub fn fill(value: &ResolvedTensor, output: &mut ResolvedTensor) -> Status {
   let elem = type_size_bytes(output.tensor_type);
   if elem == 0 || output.element_count == 0 {
      return Status::ok();
   }
   let seed = value.bytes().to_vec();
   splat_bytes(output.bytes_mut(), &seed, elem);
   Status::ok()
}

/// Write depth one-hot rows, leaving the off row behind for out-of-range keys.
///
/// `OneHot` takes the axis as an `unsigned int`, so there is no negative axis
/// to normalise, only a clamp against the inserted-dimension rank.
#[inline]
pub fn one_hot(
   indices: &ResolvedTensor,
   on_value: &ResolvedTensor,
   off_value: &ResolvedTensor,
   axis: u32,
   depth: i64,
   output: &mut ResolvedTensor,
) -> Status {
   let elem = type_size_bytes(output.tensor_type);
   let Ok(depth_usize) = usize::try_from(depth) else {
      return Status::ok();
   };
   if elem == 0 || depth_usize == 0 || output.element_count == 0 {
      return Status::ok();
   }

   let pattern = |raw: &[u8]| -> Vec<u8> {
      (0..elem)
         .map(|idx| {
            if raw.is_empty() {
               0
            } else {
               raw[idx % raw.len()]
            }
         })
         .collect()
   };
   let off_pat = pattern(off_value.bytes());
   let on_pat = pattern(on_value.bytes());

   let idx_dims = indices.shape.dims.clone();
   let idx_rank = idx_dims.len();
   let axis_norm = usize::try_from(axis).unwrap_or(idx_rank).min(idx_rank);
   let mut full = idx_dims.clone();
   full.insert(axis_norm, depth);
   let strides = row_strides(&full);

   let idx_size = type_size_bytes(indices.tensor_type);
   let idx_raw = indices.bytes().to_vec();
   let count = indices.element_count;
   let dst = output.bytes_mut();
   splat_bytes(dst, &off_pat, elem);
   if count == 0 {
      return Status::ok();
   }

   for flat in 0..count {
      let Some(raw_key) = read_padding_int(&idx_raw, flat.saturating_mul(idx_size), idx_size)
      else {
         continue;
      };
      let Ok(key) = usize::try_from(raw_key) else {
         continue;
      };
      if key >= depth_usize {
         continue;
      }
      let mut rem = flat;
      let mut base = 0_usize;
      for dim_idx in (0..idx_rank).rev() {
         let dim = extent(idx_dims[dim_idx]).max(1);
         let coord = rem % dim;
         rem /= dim;
         let out_dim = if dim_idx < axis_norm {
            dim_idx
         } else {
            dim_idx + 1
         };
         base = base.saturating_add(coord.saturating_mul(strides[out_dim]));
      }
      let to = base
         .saturating_add(key.saturating_mul(strides[axis_norm]))
         .saturating_mul(elem);
      if to.saturating_add(elem) <= dst.len() {
         dst[to..to + elem].copy_from_slice(&on_pat);
      }
   }
   Status::ok()
}

/// One accumulating lane of a cumulative sum.
#[derive(Clone, Copy)]
enum Lane {
   /// Narrowing on store preserves the modular prefix for every integer width.
   Int(i64),
   /// Float32 accumulates with `fadd.s` after every input.
   Float32(f32),
   /// Float64 accumulates with `fadd.d`.
   Float64(f64),
}

impl Lane {
   /// Additive identity for the dtype.
   const fn zero(ty: TensorType) -> Option<Self> {
      Some(match ty {
         TensorType::Int8
         | TensorType::UInt8
         | TensorType::Int16
         | TensorType::Int32
         | TensorType::UInt16
         | TensorType::UInt32
         | TensorType::Int64
         | TensorType::UInt64 => Self::Int(0_i64),
         TensorType::Float32 => Self::Float32(0.0_f32),
         TensorType::Float64 => Self::Float64(0.0_f64),
         TensorType::Bool | TensorType::Float16 | TensorType::Bfloat => return None,
      })
   }

   /// Adds two lanes of the same kind.
   fn add(self, other: Self) -> Option<Self> {
      Some(match (self, other) {
         (Self::Int(lhs), Self::Int(rhs)) => Self::Int(lhs.wrapping_add(rhs)),
         (Self::Float32(lhs), Self::Float32(rhs)) => Self::Float32(lhs + rhs),
         (Self::Float64(lhs), Self::Float64(rhs)) => Self::Float64(lhs + rhs),
         _ => return None,
      })
   }
}

/// Reads one element into the accumulator lane for its dtype.
fn load_lane(ty: TensorType, cell: &[u8]) -> Option<Lane> {
   Some(match ty {
      TensorType::Int8 => Lane::Int(i64::from(cell[0].cast_signed())),
      TensorType::UInt8 => Lane::Int(i64::from(cell[0])),
      TensorType::Int16 => Lane::Int(i64::from(i16::from_le_bytes(dive::lane::<2>(cell)))),
      TensorType::UInt16 => Lane::Int(i64::from(u16::from_le_bytes(dive::lane::<2>(cell)))),
      TensorType::Int32 => Lane::Int(i64::from(i32::from_le_bytes(dive::lane::<4>(cell)))),
      TensorType::UInt32 => Lane::Int(i64::from(u32::from_le_bytes(dive::lane::<4>(cell)))),
      TensorType::Int64 => Lane::Int(i64::from_le_bytes(dive::lane::<8>(cell))),
      TensorType::UInt64 => Lane::Int(u64::from_le_bytes(dive::lane::<8>(cell)).cast_signed()),
      TensorType::Float32 => Lane::Float32(f32::from_le_bytes(dive::lane::<4>(cell))),
      TensorType::Float64 => Lane::Float64(f64::from_le_bytes(dive::lane::<8>(cell))),
      TensorType::Bool | TensorType::Float16 | TensorType::Bfloat => return None,
   })
}

/// Writes an accumulator lane back at the dtype width.
fn store_lane(ty: TensorType, cell: &mut [u8], lane: Lane) -> Option<()> {
   let whole = match lane {
      Lane::Int(value) => value,
      Lane::Float32(_) | Lane::Float64(_) => 0_i64,
   };
   let mut out = [0_u8; 8];
   let width = match ty {
      TensorType::Int8 | TensorType::UInt8 => {
         out[0] = whole as u8;
         1
      },
      TensorType::Int16 | TensorType::UInt16 => {
         dive::stage(&mut out, &(whole as u16).to_le_bytes())
      },
      TensorType::Int32 | TensorType::UInt32 => {
         dive::stage(&mut out, &(whole as u32).to_le_bytes())
      },
      TensorType::Int64 | TensorType::UInt64 => {
         dive::stage(&mut out, &whole.cast_unsigned().to_le_bytes())
      },
      TensorType::Float32 => {
         let Lane::Float32(value) = lane else {
            return None;
         };
         dive::stage(&mut out, &value.to_le_bytes())
      },
      TensorType::Float64 => {
         let Lane::Float64(value) = lane else {
            return None;
         };
         dive::stage(&mut out, &value.to_le_bytes())
      },
      TensorType::Bool | TensorType::Float16 | TensorType::Bfloat => return None,
   };
   cell[..width].copy_from_slice(&out[..width]);
   Some(())
}

/// Accumulate along axis, honoring exclusive and reverse flags.
#[inline]
pub fn cumsum(
   input: &ResolvedTensor,
   axis: i32,
   exclusive: bool,
   reverse: bool,
   output: &mut ResolvedTensor,
) -> Status {
   const FILE: &str = "cumsum.cc";

   if matches!(
      input.tensor_type,
      TensorType::Bool | TensorType::Float16 | TensorType::Bfloat
   ) {
      dive::log("Unsupported type for Cumsum", FILE);
      return Status::new(Code::Unimplemented);
   }

   let dims = input.shape.dims.clone();
   let rank = dims.len();
   let wide = i32::try_from(rank).unwrap_or(i32::MAX);
   let shifted = if axis < 0_i32 {
      axis.saturating_add(wide)
   } else {
      axis
   };
   let Ok(ax) = usize::try_from(shifted) else {
      return Status::new(Code::InvalidArgument);
   };
   if ax >= rank
      || input.tensor_type != output.tensor_type
      || input.shape != output.shape
      || input.element_count != output.element_count
   {
      return Status::new(Code::InvalidArgument);
   }

   let ty = input.tensor_type;
   let size = type_size_bytes(ty);
   let axis_len = extent(dims[ax]);
   if axis_len == 0 {
      return Status::ok();
   }
   let inner = dims[ax + 1..]
      .iter()
      .fold(1_usize, |acc, dim| acc.saturating_mul(extent(*dim)));
   let outer = dims[..ax]
      .iter()
      .fold(1_usize, |acc, dim| acc.saturating_mul(extent(*dim)));
   let total = outer.saturating_mul(axis_len).saturating_mul(inner);
   let need = total.saturating_mul(size);
   if total == 0 {
      return Status::ok();
   }
   if input.bytes().len() < need || output.bytes().len() < need {
      return Status::new(Code::InvalidArgument);
   }

   let src = input.bytes().to_vec();
   let dst = output.bytes_mut();
   for outer_idx in 0..outer {
      for inner_idx in 0..inner {
         let Some(mut accumulator) = Lane::zero(ty) else {
            return Status::new(Code::Unimplemented);
         };
         for step in 0..axis_len {
            let pos = if reverse { axis_len - 1 - step } else { step };
            let flat = (outer_idx * axis_len + pos) * inner + inner_idx;
            let at = flat * size;
            let Some(value) = load_lane(ty, &src[at..]) else {
               return Status::new(Code::Unimplemented);
            };
            if exclusive {
               if store_lane(ty, &mut dst[at..], accumulator).is_none() {
                  return Status::new(Code::Unimplemented);
               }
               let Some(sum) = accumulator.add(value) else {
                  return Status::new(Code::Internal);
               };
               accumulator = sum;
            } else {
               let Some(sum) = accumulator.add(value) else {
                  return Status::new(Code::Internal);
               };
               accumulator = sum;
               if store_lane(ty, &mut dst[at..], accumulator).is_none() {
                  return Status::new(Code::Unimplemented);
               }
            }
         }
      }
   }
   Status::ok()
}

#[cfg(test)]
mod tests {
   use alloc::vec;

   use crate::{
      pad_fill::{
         cumsum,
         one_hot,
      },
      tensor::TensorType,
      test_support::{
         f32_bytes,
         i32_bytes,
         read_f32,
         tensor,
         zero_tensor,
      },
   };

   #[test]
   fn one_hot_axis_zero_puts_depth_first() {
      let indices = tensor(
         TensorType::Int32,
         vec![3],
         &i32_bytes(&[0_i32, 2_i32, 1_i32]),
      );
      let on = tensor(TensorType::Float32, vec![], &f32_bytes(&[1.0]));
      let off = tensor(TensorType::Float32, vec![], &f32_bytes(&[0.0]));
      let mut output = zero_tensor(TensorType::Float32, vec![4, 3]);
      assert!(
         one_hot(&indices, &on, &off, 0, 4, &mut output).is_ok(),
         "leading axis one hot should succeed"
      );
      assert_eq!(read_f32(&output), vec![
         1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      ]);
   }

   #[test]
   fn one_hot_out_of_range_row_stays_zero() {
      let indices = tensor(TensorType::Int32, vec![2], &i32_bytes(&[1_i32, 9_i32]));
      let on = tensor(TensorType::Float32, vec![], &f32_bytes(&[1.0]));
      let off = tensor(TensorType::Float32, vec![], &f32_bytes(&[0.0]));
      let mut output = zero_tensor(TensorType::Float32, vec![2, 4]);
      assert!(
         one_hot(&indices, &on, &off, 1, 4, &mut output).is_ok(),
         "out of range one hot should succeed with an off row"
      );
      assert_eq!(read_f32(&output), vec![
         0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      ]);
   }

   #[test]
   fn cumsum_float32_uses_float32_accumulator() {
      let input = tensor(
         TensorType::Float32,
         vec![3],
         &f32_bytes(&[16_777_216.0, 1.0, -16_777_216.0]),
      );
      let mut output = zero_tensor(TensorType::Float32, vec![3]);
      assert!(
         cumsum(&input, 0, false, false, &mut output).is_ok(),
         "forward cumulative sum should succeed"
      );
      assert_eq!(read_f32(&output), vec![16_777_216.0, 16_777_216.0, 0.0]);
   }

   #[test]
   fn cumsum_reverse_accumulates_from_end() {
      let input = tensor(
         TensorType::Float32,
         vec![4],
         &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
      );
      let mut output = zero_tensor(TensorType::Float32, vec![4]);
      assert!(
         cumsum(&input, 0, false, true, &mut output).is_ok(),
         "reverse cumulative sum should succeed"
      );
      assert_eq!(read_f32(&output), vec![10.0, 9.0, 7.0, 4.0]);
   }
}
