use alloc::{
   vec,
   vec::Vec,
};

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
      check_op_helper_out_of_line,
   },
};

use crate::tensor::{
   ResolvedTensor,
   Shape,
   TensorType,
   UnresolvedTensor,
   type_size_bytes,
};

/// Saturating product of dims, ceasing on a negative extent.
fn product(dims: &[i64], file: &'static str) -> usize {
   let mut acc = 1_usize;
   for dim in dims {
      let Ok(extent) = usize::try_from(*dim) else {
         dive::log("negative dimension", file);
         dive::cease(19, file);
      };
      acc = acc.saturating_mul(extent);
   }
   acc
}

/// Maps a possibly negative axis onto its range-checked position.
fn normalize_axis(axis: i64, rank: usize, file: &'static str) -> usize {
   let count = i64::try_from(rank).unwrap_or(i64::MAX);
   let fixed = if axis < 0 {
      axis.saturating_add(count)
   } else {
      axis
   };
   if fixed < 0 || fixed >= count {
      dive::log("axis out of range", file);
      dive::cease(23, file);
   }
   usize::try_from(fixed).unwrap_or(0)
}

/// Reads one packed i32 index lane.
fn read_index(picks: &[u8], at: usize) -> i32 {
   let cell = &picks[at * 4..];
   i32::from_le_bytes(dive::lane::<4>(cell))
}

/// Row-major element strides for validated dims.
fn full_strides(dims: &[i64]) -> Vec<usize> {
   let mut strides = vec![1_usize; dims.len()];
   let mut acc = 1_usize;
   for pos in (0..dims.len()).rev() {
      strides[pos] = acc;
      acc = acc.saturating_mul(usize::try_from(dims[pos]).unwrap_or(0));
   }
   strides
}

/// Gather slices along `axis`, zero-filling out-of-range picks.
#[inline]
pub fn gather(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   axis: i64,
   batch_dims: i64,
   output: &mut ResolvedTensor,
) {
   const FILE: &str = "gather.cc";

   let params_rank = params.shape.rank();
   let indices_rank = indices.shape.rank();
   let axis_norm = normalize_axis(axis, params_rank, FILE);
   let Ok(lead) = usize::try_from(batch_dims) else {
      dive::log("batch_dims out of range", FILE);
      dive::cease(23, FILE);
   };
   if lead > axis_norm || lead > indices_rank {
      dive::log("batch_dims out of range", FILE);
      dive::cease(23, FILE);
   }
   for pos in 0..lead {
      if params.shape.dims[pos] != indices.shape.dims[pos] {
         dive::log("batch dimension mismatch", FILE);
         dive::cease(23, FILE);
      }
   }
   let Ok(axis_len) = usize::try_from(params.shape.dims[axis_norm]) else {
      dive::log("negative axis dimension", FILE);
      dive::cease(19, FILE);
   };
   if output.tensor_type != params.tensor_type {
      dive::log("dtype mismatch", FILE);
      dive::cease(23, FILE);
   }
   let elem = type_size_bytes(params.tensor_type);
   let batch = product(&params.shape.dims[..lead], FILE);
   let outer = product(&params.shape.dims[lead..axis_norm], FILE);
   let inner = product(&params.shape.dims[axis_norm + 1..], FILE);
   let per_batch = product(&indices.shape.dims[lead..], FILE);
   if params.element_count != batch * outer * axis_len * inner
      || indices.element_count != batch * per_batch
      || output.element_count != batch * outer * per_batch * inner
   {
      dive::log("shape product mismatch", FILE);
      dive::cease(23, FILE);
   }
   params.require_bytes(params.element_count * elem, FILE);
   indices.require_bytes(indices.element_count * 4, FILE);
   output.require_bytes(output.element_count * elem, FILE);

   let row_bytes = inner * elem;
   let src = params.bytes().to_vec();
   let picks = indices.bytes().to_vec();
   let dst = output.bytes_mut();
   for bb in 0..batch {
      for oo in 0..outer {
         for jj in 0..per_batch {
            let pick = read_index(&picks, bb * per_batch + jj);
            let at = ((bb * outer + oo) * per_batch + jj) * row_bytes;
            let slot = &mut dst[at..at + row_bytes];
            match usize::try_from(pick) {
               Ok(row) if row < axis_len => {
                  let from = ((bb * outer + oo) * axis_len + row) * row_bytes;
                  slot.copy_from_slice(&src[from..from + row_bytes]);
               },
               _ => slot.fill(0),
            }
         }
      }
   }
}

/// Batch geometry shared by the `gather_nd` helpers.
struct NdGeometry {
   /// Leading batch elements.
   batch:       usize,
   /// Index tuples per batch.
   index_count: usize,
   /// Coordinates per tuple.
   keys:        usize,
   /// Trailing elements copied per tuple.
   copy_elems:  usize,
}

/// Validates ranks and splits params and indices into batch geometry.
fn nd_geometry(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   batch_dims: i64,
   file: &'static str,
) -> NdGeometry {
   let params_rank = params.shape.rank();
   let indices_rank = indices.shape.rank();
   let lead = usize::try_from(batch_dims).unwrap_or(usize::MAX);
   if indices_rank < 1 || lead > indices_rank - 1 {
      dive::log("batch_dims out of range", file);
      dive::cease(23, file);
   }
   for pos in 0..lead {
      if params.shape.dims[pos] != indices.shape.dims[pos] {
         dive::log("batch dimension mismatch", file);
         dive::cease(23, file);
      }
   }
   let keys = usize::try_from(indices.shape.dims[indices_rank - 1]).unwrap_or(usize::MAX);
   if lead.saturating_add(keys) > params_rank {
      dive::log("index depth out of range", file);
      dive::cease(23, file);
   }
   let batch = product(&params.shape.dims[..lead], file);
   let index_count = product(&indices.shape.dims[lead..indices_rank - 1], file);
   let copy_elems = product(&params.shape.dims[lead + keys..], file);
   if indices.element_count != batch * index_count * keys {
      dive::log("shape product mismatch", file);
      dive::cease(23, file);
   }
   NdGeometry {
      batch,
      index_count,
      keys,
      copy_elems,
   }
}

#[inline]
pub fn resolve_gather_nd_output(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   batch_dims: i64,
   output: &mut UnresolvedTensor,
) -> ResolvedTensor {
   const FILE: &str = "gather_nd_utils.cc";

   let geo = nd_geometry(params, indices, batch_dims, FILE);
   let indices_rank = indices.shape.rank();
   let lead = usize::try_from(batch_dims).unwrap_or(0);
   let mut dims = indices.shape.dims[..indices_rank - 1].to_vec();
   dims.extend_from_slice(&params.shape.dims[lead + geo.keys..]);
   if dims.len() > 16 {
      dive::log("output rank exceeding max rank", FILE);
      dive::cease(23, FILE);
   }
   output.resolve(Shape::new(dims))
}

#[must_use]
#[inline]
pub fn gather_nd_copy_size_bytes(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   batch_dims: i64,
) -> usize {
   let geo = nd_geometry(params, indices, batch_dims, "gather_nd_utils.cc");
   type_size_bytes(params.tensor_type) * geo.copy_elems
}

/// Copy gathered slices, zero-filling out-of-range tuples.
#[inline]
pub fn gather_nd_fallback(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   batch_dims: i64,
   output: &mut ResolvedTensor,
   copy_size_bytes: usize,
) {
   const FILE: &str = "gather_nd_z280.cc";

   let geo = nd_geometry(params, indices, batch_dims, FILE);
   if output.tensor_type != params.tensor_type {
      dive::log("dtype mismatch", FILE);
      dive::cease(23, FILE);
   }
   let elem = type_size_bytes(params.tensor_type);
   if copy_size_bytes != elem * geo.copy_elems {
      dive::log("copy size mismatch", FILE);
      dive::cease(23, FILE);
   }
   if output.element_count != geo.batch * geo.index_count * geo.copy_elems {
      dive::log("shape product mismatch", FILE);
      dive::cease(23, FILE);
   }
   params.require_bytes(params.element_count * elem, FILE);
   indices.require_bytes(indices.element_count * 4, FILE);
   output.require_bytes(output.element_count * elem, FILE);

   let params_strides = full_strides(&params.shape.dims);
   let params_dims = params.shape.dims.clone();
   let outer_shape = indices.shape.dims[..indices.shape.rank() - 1].to_vec();
   let outer_strides = full_strides(&outer_shape);
   let lead = usize::try_from(batch_dims).unwrap_or(0);
   let src = params.bytes().to_vec();
   let picks = indices.bytes().to_vec();
   let dst = output.bytes_mut();
   for flat in 0..geo.batch * geo.index_count {
      let mut base = 0_usize;
      for dd in 0..lead {
         let extent = usize::try_from(outer_shape[dd]).unwrap_or(1).max(1);
         base += (flat / outer_strides[dd]) % extent * params_strides[dd];
      }
      let mut at = Some(base);
      for kk in 0..geo.keys {
         let pick = read_index(&picks, flat * geo.keys + kk);
         at = match usize::try_from(pick) {
            Ok(coord) if i64::from(pick) < params_dims[lead + kk] => {
               at.map(|sum| sum + coord * params_strides[lead + kk])
            },
            _ => None,
         };
         if at.is_none() {
            break;
         }
      }
      let to = flat * copy_size_bytes;
      let slot = &mut dst[to..to + copy_size_bytes];
      match at {
         Some(from) => slot.copy_from_slice(&src[from * elem..from * elem + copy_size_bytes]),
         None => slot.fill(0),
      }
   }
}

#[inline]
pub fn gather_nd(
   params: &ResolvedTensor,
   indices: &ResolvedTensor,
   batch_dims: i64,
   output: &mut UnresolvedTensor,
) -> ResolvedTensor {
   let mut done = resolve_gather_nd_output(params, indices, batch_dims, output);
   let status = done.allocate_backing_buffer();
   check_op_helper_out_of_line(&status, "gather_nd");
   let width = gather_nd_copy_size_bytes(params, indices, batch_dims);
   gather_nd_fallback(params, indices, batch_dims, &mut done, width);
   done
}

/// Adds one update cell using the destination storage width.
fn add_scatter_cell(ty: TensorType, destination: &mut [u8], update: &[u8]) -> Option<()> {
   match ty {
      TensorType::Bool => destination[0] = u8::from(destination[0] != 0 || update[0] != 0),
      TensorType::Int8 | TensorType::UInt8 => {
         destination[0] = destination[0].wrapping_add(update[0]);
      },
      TensorType::Int16 | TensorType::UInt16 => {
         let value = u16::from_le_bytes(dive::lane::<2>(destination))
            .wrapping_add(u16::from_le_bytes(dive::lane::<2>(update)));
         destination.copy_from_slice(&value.to_le_bytes());
      },
      TensorType::Float16 | TensorType::Bfloat => return None,
      TensorType::Int32 | TensorType::UInt32 => {
         let value = u32::from_le_bytes(dive::lane::<4>(destination))
            .wrapping_add(u32::from_le_bytes(dive::lane::<4>(update)));
         destination.copy_from_slice(&value.to_le_bytes());
      },
      TensorType::Float32 => {
         let value = f32::from_le_bytes(dive::lane::<4>(destination))
            + f32::from_le_bytes(dive::lane::<4>(update));
         destination.copy_from_slice(&value.to_le_bytes());
      },
      TensorType::Int64 | TensorType::UInt64 => {
         let value = u64::from_le_bytes(dive::lane::<8>(destination))
            .wrapping_add(u64::from_le_bytes(dive::lane::<8>(update)));
         destination.copy_from_slice(&value.to_le_bytes());
      },
      TensorType::Float64 => {
         let value = f64::from_le_bytes(dive::lane::<8>(destination))
            + f64::from_le_bytes(dive::lane::<8>(update));
         destination.copy_from_slice(&value.to_le_bytes());
      },
   }
   Some(())
}

/// Accumulate scatter updates into a zeroed output, skipping out-of-range
/// tuples.
#[inline]
#[must_use]
pub fn scatter_nd(
   updates: &ResolvedTensor,
   indices: &ResolvedTensor,
   output: &mut ResolvedTensor,
) -> Status {
   const FILE: &str = "scatter_nd.cc";

   if matches!(
      updates.tensor_type,
      TensorType::Float16 | TensorType::Bfloat
   ) {
      return Status::new(Code::Unimplemented);
   }

   let indices_rank = indices.shape.rank();
   if indices_rank < 1 {
      dive::log("index rank out of range", FILE);
      dive::cease(23, FILE);
   }
   let keys = usize::try_from(indices.shape.dims[indices_rank - 1]).unwrap_or(usize::MAX);
   if keys > output.shape.rank() {
      dive::log("index depth out of range", FILE);
      dive::cease(23, FILE);
   }
   if updates.tensor_type != output.tensor_type {
      dive::log("dtype mismatch", FILE);
      dive::cease(23, FILE);
   }
   let ty = output.tensor_type;
   let elem = type_size_bytes(ty);
   let outer_count = product(&indices.shape.dims[..indices_rank - 1], FILE);
   let copy_elems = product(&output.shape.dims[keys..], FILE);
   if output.element_count != product(&output.shape.dims, FILE)
      || indices.element_count != outer_count * keys
      || updates.element_count != outer_count * copy_elems
   {
      dive::log("shape product mismatch", FILE);
      dive::cease(23, FILE);
   }
   indices.require_bytes(indices.element_count * 4, FILE);
   updates.require_bytes(updates.element_count * elem, FILE);
   output.require_bytes(output.element_count * elem, FILE);

   let copy_bytes = copy_elems * elem;
   let strides = full_strides(&output.shape.dims);
   let out_dims = output.shape.dims.clone();
   let picks = indices.bytes().to_vec();
   let src = updates.bytes().to_vec();
   let dst = output.bytes_mut();
   dst.fill(0);
   for flat in 0..outer_count {
      let mut at = Some(0_usize);
      for kk in 0..keys {
         let pick = read_index(&picks, flat * keys + kk);
         at = match usize::try_from(pick) {
            Ok(coord) if i64::from(pick) < out_dims[kk] => at.map(|sum| sum + coord * strides[kk]),
            _ => None,
         };
         if at.is_none() {
            break;
         }
      }
      if let Some(to) = at {
         for offset in 0..copy_elems {
            let from = flat * copy_bytes + offset * elem;
            let destination = (to + offset) * elem;

            if add_scatter_cell(
               ty,
               &mut dst[destination..destination + elem],
               &src[from..from + elem],
            )
            .is_none()
            {
               return Status::new(Code::Unimplemented);
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
      gather::{
         gather,
         scatter_nd,
      },
      tensor::TensorType,
      test_support::{
         i32_bytes,
         read_i32,
         tensor,
         zero_tensor,
      },
   };

   #[test]
   fn negative_axis_normalizes() {
      let params = tensor(
         TensorType::Int32,
         vec![2, 3],
         &i32_bytes(&[1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32]),
      );
      let indices = tensor(TensorType::Int32, vec![2], &i32_bytes(&[2_i32, 0_i32]));
      let mut output = zero_tensor(TensorType::Int32, vec![2, 2]);
      gather(&params, &indices, -1, 0, &mut output);
      assert_eq!(read_i32(&output), vec![3_i32, 1_i32, 6_i32, 4_i32]);
   }

   #[test]
   fn out_of_range_zero_fills_row() {
      let params = tensor(
         TensorType::Int32,
         vec![3, 2],
         &i32_bytes(&[1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32]),
      );
      let indices = tensor(TensorType::Int32, vec![2], &i32_bytes(&[5_i32, -1_i32]));
      let mut output = zero_tensor(TensorType::Int32, vec![2, 2]);
      gather(&params, &indices, 0, 0, &mut output);
      assert_eq!(read_i32(&output), vec![0_i32; 4]);
   }

   #[test]
   fn scatter_nd_accumulates_duplicate_indices() {
      let indices = tensor(
         TensorType::Int32,
         vec![3, 1],
         &i32_bytes(&[1_i32, 1_i32, 3_i32]),
      );
      let updates = tensor(
         TensorType::Int32,
         vec![3],
         &i32_bytes(&[10_i32, 5_i32, 30_i32]),
      );
      let mut output = zero_tensor(TensorType::Int32, vec![4]);
      assert!(
         scatter_nd(&updates, &indices, &mut output).is_ok(),
         "scatter should accept duplicate indices"
      );
      assert_eq!(read_i32(&output), vec![0_i32, 15_i32, 0_i32, 30_i32]);
   }
}
