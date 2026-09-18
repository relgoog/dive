use alloc::vec::Vec;

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

/// Sort key for one element.
///
/// The `kernels::TopK` family contains no float compare instruction at all.
/// Float32's sort loop at 0x85DA8 orders with `bgeu` and `bltu`, so it compares
/// the raw bit pattern as an unsigned integer. That agrees with numeric order
/// only while the values are non-negative, which is the same assumption the
/// vectorised `TopKVector` makes.
#[inline]
#[must_use]
pub const fn sort_key(value: f32) -> u32 {
   value.to_bits()
}

/// `TopK` indexes a twelve entry jump table by `dtype - 1`, so Bool falls off
/// the bottom. Float16 and Bfloat share the out-of-range bail target, which
/// logs at topk.cc:151 and yields code 12.
#[inline]
#[must_use]
pub const fn top_k_supports(ty: TensorType) -> bool {
   match ty {
      TensorType::Bool | TensorType::Float16 | TensorType::Bfloat => false,
      TensorType::Int8
      | TensorType::UInt8
      | TensorType::Int16
      | TensorType::Int32
      | TensorType::Float32
      | TensorType::UInt16
      | TensorType::UInt32
      | TensorType::Int64
      | TensorType::UInt64
      | TensorType::Float64 => true,
   }
}

/// Rejects the dtypes whose jump table slot is the bail target.
#[inline]
#[must_use]
pub const fn check_top_k_dtype(ty: TensorType) -> Status {
   if top_k_supports(ty) {
      Status::ok()
   } else {
      dive::log("unsupported TopK dtype", "topk.cc");
      Status::new(Code::Unimplemented)
   }
}

/// Inserts into a descending top-k window, returning the slot taken or the
/// window depth when the value does not place.
#[inline]
pub fn insert_value(value: i32, index: i32, values: &mut [i32], indices: &mut [i32]) -> usize {
   let depth = values.len().min(indices.len());
   let Some(pos) = values
      .iter()
      .take(depth)
      .position(|current| value >= *current)
   else {
      return depth;
   };
   for target in (pos + 1..depth).rev() {
      values[target] = values[target - 1];
      indices[target] = indices[target - 1];
   }
   values[pos] = value;
   indices[pos] = index;
   pos
}

/// Compares two source positions with the kernel's dtype-specific ordering.
fn selected_before(
   ty: TensorType,
   bytes: &[u8],
   width: usize,
   left_index: usize,
   right_index: usize,
) -> bool {
   let left = &bytes[left_index * width..][..width];
   let right = &bytes[right_index * width..][..width];

   match ty {
      TensorType::Int8 => left[0].cast_signed() > right[0].cast_signed(),
      TensorType::UInt8 => left[0] > right[0],
      TensorType::Int16 => {
         i16::from_le_bytes(dive::lane::<2>(left)) > i16::from_le_bytes(dive::lane::<2>(right))
      },
      TensorType::Int32 => {
         i32::from_le_bytes(dive::lane::<4>(left)) > i32::from_le_bytes(dive::lane::<4>(right))
      },
      TensorType::Float32 | TensorType::UInt32 => {
         u32::from_le_bytes(dive::lane::<4>(left)) > u32::from_le_bytes(dive::lane::<4>(right))
      },
      TensorType::UInt16 => {
         u16::from_le_bytes(dive::lane::<2>(left)) > u16::from_le_bytes(dive::lane::<2>(right))
      },
      TensorType::Int64 => {
         i64::from_le_bytes(dive::lane::<8>(left)) > i64::from_le_bytes(dive::lane::<8>(right))
      },
      TensorType::UInt64 | TensorType::Float64 => {
         u64::from_le_bytes(dive::lane::<8>(left)) > u64::from_le_bytes(dive::lane::<8>(right))
      },
      TensorType::Bool | TensorType::Float16 | TensorType::Bfloat => false,
   }
}

/// Selects the highest ordered positions while retaining source order for ties.
fn select_indices(
   ty: TensorType,
   input: &[u8],
   width: usize,
   depth: usize,
   keep: usize,
) -> Vec<usize> {
   let mut selected = Vec::with_capacity(keep);

   for candidate in 0..depth {
      let slot = selected
         .iter()
         .position(|current| selected_before(ty, input, width, candidate, *current))
         .unwrap_or(selected.len());

      if slot < keep {
         selected.insert(slot, candidate);
         selected.truncate(keep);
      }
   }

   selected
}

#[inline]
pub fn top_k_vector(
   input: &[f32],
   values: &mut [f32],
   indices: &mut [i32],
   batch: i32,
   depth: i32,
   keep: i32,
) -> Status {
   let (Ok(batch_count), Ok(depth_count), Ok(keep_count)) = (
      usize::try_from(batch),
      usize::try_from(depth),
      usize::try_from(keep),
   ) else {
      return Status::new(Code::InvalidArgument);
   };
   let Some(input_len) = batch_count.checked_mul(depth_count) else {
      return Status::new(Code::InvalidArgument);
   };
   let Some(output_len) = batch_count.checked_mul(keep_count) else {
      return Status::new(Code::InvalidArgument);
   };

   if depth_count == 0
      || keep_count > depth_count
      || input.len() < input_len
      || values.len() < output_len
      || indices.len() < output_len
   {
      dive::log("invalid TopKVector dimensions", "top_k_vector.cc");
      return Status::new(Code::InvalidArgument);
   }

   for row in 0..batch_count {
      let input_row = &input[row * depth_count..][..depth_count];
      let mut selected: Vec<usize> = Vec::with_capacity(keep_count);

      for candidate in 0..depth_count {
         let key = input_row[candidate].to_bits().cast_signed();
         let slot = selected
            .iter()
            .position(|current| key > input_row[*current].to_bits().cast_signed())
            .unwrap_or(selected.len());

         if slot < keep_count {
            selected.insert(slot, candidate);
            selected.truncate(keep_count);
         }
      }

      for (slot, source) in selected.into_iter().enumerate() {
         values[row * keep_count + slot] = input_row[source];
         indices[row * keep_count + slot] = source as i32;
      }
   }

   Status::ok()
}

#[inline]
pub fn top_k(
   input: &ResolvedTensor,
   values: &mut ResolvedTensor,
   indices: &mut ResolvedTensor,
) -> Status {
   const FILE: &str = "topk.cc";

   let dtype_status = check_top_k_dtype(input.tensor_type);
   if !dtype_status.is_ok() {
      return dtype_status;
   }
   if values.tensor_type != input.tensor_type || indices.tensor_type != TensorType::Int32 {
      dive::log("TopK output dtype mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   if values.shape != indices.shape || values.shape.rank() != input.shape.rank() {
      dive::log("TopK output shape mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   if input.shape.rank() > 0
      && input.shape.dims[..input.shape.rank() - 1] != values.shape.dims[..values.shape.rank() - 1]
   {
      dive::log("TopK outer shape mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }

   let depth = input.shape.dims.last().copied().map_or(1, extent);
   let keep = values.shape.dims.last().copied().map_or(1, extent);
   if depth == 0
      || keep > depth
      || depth > i32::MAX as usize
      || input.element_count != input.shape.num_elements()
      || !input.element_count.is_multiple_of(depth)
   {
      dive::log("TopK invalid depth or k", FILE);
      return Status::new(Code::InvalidArgument);
   }

   let rows = input.element_count / depth;
   let Some(output_count) = rows.checked_mul(keep) else {
      return Status::new(Code::InvalidArgument);
   };
   if values.element_count != output_count
      || indices.element_count != output_count
      || values.element_count != values.shape.num_elements()
      || indices.element_count != indices.shape.num_elements()
   {
      dive::log("TopK output size mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }

   let width = type_size_bytes(input.tensor_type);
   let Some(input_size) = input.element_count.checked_mul(width) else {
      return Status::new(Code::InvalidArgument);
   };
   let Some(values_size) = values.element_count.checked_mul(width) else {
      return Status::new(Code::InvalidArgument);
   };
   let Some(indices_size) = indices.element_count.checked_mul(4) else {
      return Status::new(Code::InvalidArgument);
   };
   if input.bytes().len() < input_size
      || values.bytes().len() < values_size
      || indices.bytes().len() < indices_size
   {
      dive::log("TopK backing buffer too small", FILE);
      return Status::new(Code::InvalidArgument);
   }

   let input_bytes = input.bytes();
   let value_bytes = values.bytes_mut();
   let index_bytes = indices.bytes_mut();

   for row in 0..rows {
      let source = &input_bytes[row * depth * width..][..depth * width];
      let selected = select_indices(input.tensor_type, source, width, depth, keep);

      for (slot, index) in selected.into_iter().enumerate() {
         let value_at = (row * keep + slot) * width;
         let source_at = index * width;
         value_bytes[value_at..value_at + width]
            .copy_from_slice(&source[source_at..source_at + width]);

         let index_at = (row * keep + slot) * 4;
         index_bytes[index_at..index_at + 4].copy_from_slice(&(index as i32).to_le_bytes());
      }
   }

   Status::ok()
}

/// Canonicalises an axis against `rank`, dropping values that stay out of
/// range.
fn canonical_axis(axis: i32, rank: usize) -> Option<usize> {
   let shifted = if axis < 0_i32 {
      axis + rank as i32
   } else {
      axis
   };
   usize::try_from(shifted).ok().filter(|dim| *dim < rank)
}

#[inline]
#[must_use]
pub fn populate_non_axis(rank: usize, axes: &[i32]) -> Vec<i32> {
   let reduced = axes
      .iter()
      .filter_map(|ax| canonical_axis(*ax, rank))
      .collect::<Vec<usize>>();
   (0..rank)
      .filter(|dim| !reduced.contains(dim))
      .map(|dim| dim as i32)
      .collect()
}

#[inline]
#[must_use]
pub fn get_reduced_offset(shape: &Shape, full_index: &[i32], axes: &[i32]) -> usize {
   let rank = shape.rank();
   if full_index.len() < rank {
      dive::cease(3, "topk_reduce.cc");
   }
   let mut offset = 0_usize;
   for dim in 0..rank {
      if axes.iter().any(|ax| canonical_axis(*ax, rank) == Some(dim)) {
         continue;
      }
      offset = offset
         .saturating_mul(usize::try_from(shape.dims[dim]).unwrap_or(0))
         .saturating_add(usize::try_from(full_index[dim]).unwrap_or(0));
   }
   offset
}

/// `reduce_internal` carries the loop index in a `std::array<int, 10>`, so ten
/// is the rank ceiling for every odometer here.
pub const MAX_RANK: usize = 10;

/// Which axis group advanced on one `get_next_index` step.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextIndex {
   /// Both groups wrapped back to zero.
   Exhausted = 0,
   /// A dim in the second group advanced.
   Outer     = 256,
   /// A dim in the first group advanced.
   Inner     = 257,
}

/// Advances the last axis in `axes` that has room, resetting the ones past it.
fn step_axes(shape: &Shape, axes: &[i32], index: &mut [i32; MAX_RANK]) -> bool {
   let rank = shape.rank();
   for axis in axes.iter().rev() {
      let Some(dim) = canonical_axis(*axis, rank) else {
         dive::cease(3, "topk_reduce.cc");
      };
      let next = index[dim] + 1_i32;
      if i64::from(next) != shape.dims[dim] {
         index[dim] = next;
         return true;
      }
      index[dim] = 0_i32;
   }
   false
}

/// Steps the reduction odometer over two axis groups, inner group first.
///
/// The binary returns 257 when a dim in the first span advanced, 256 when one
/// in the second did, and 0 once both wrapped, so callers can tell which level
/// ticked. It carries only on `dims[axis] == index[axis] + 1`.
#[inline]
pub fn get_next_index(
   shape: &Shape,
   inner_axes: &[i32],
   outer_axes: &[i32],
   index: &mut [i32; MAX_RANK],
) -> NextIndex {
   if step_axes(shape, inner_axes, index) {
      return NextIndex::Inner;
   }
   if step_axes(shape, outer_axes, index) {
      return NextIndex::Outer;
   }
   NextIndex::Exhausted
}

/// Steps every dim from the innermost outward, returning false once it wraps.
#[inline]
pub fn argmax_next_index(shape: &Shape, index: &mut [i32; MAX_RANK]) -> bool {
   let rank = shape.rank();
   if rank > MAX_RANK {
      dive::cease(3, "topk_reduce.cc");
   }
   for dim in (0..rank).rev() {
      let next = index[dim] + 1_i32;
      if i64::from(next) != shape.dims[dim] {
         index[dim] = next;
         return true;
      }
      index[dim] = 0_i32;
   }
   false
}

#[cfg(test)]
mod tests {
   use alloc::vec;

   use crate::{
      tensor::{
         Shape,
         TensorType,
      },
      test_support::{
         f32_bytes,
         i32_bytes,
         read_f32,
         tensor,
         zero_tensor,
      },
      topk_reduce::{
         MAX_RANK,
         NextIndex,
         get_next_index,
         populate_non_axis,
         top_k,
      },
   };

   #[test]
   fn populate_non_axis_and_next_index_cover_once() {
      let shape = Shape::new(vec![2, 3, 4]);
      let non_axis = populate_non_axis(3, &[1_i32]);
      assert_eq!(non_axis, vec![0_i32, 2_i32]);
      let mut index = [0_i32; MAX_RANK];
      let mut seen = vec![[0_i32, 0_i32, 0_i32]];
      while get_next_index(&shape, &non_axis, &[], &mut index) != NextIndex::Exhausted {
         seen.push([index[0], index[1], index[2]]);
      }
      assert_eq!(seen, vec![
         [0_i32, 0_i32, 0_i32],
         [0_i32, 0_i32, 1_i32],
         [0_i32, 0_i32, 2_i32],
         [0_i32, 0_i32, 3_i32],
         [1_i32, 0_i32, 0_i32],
         [1_i32, 0_i32, 1_i32],
         [1_i32, 0_i32, 2_i32],
         [1_i32, 0_i32, 3_i32],
      ]);
   }

   /// The binary returns 257 when a dim in the first span moved and 256 when
   /// one in the second did, so a caller can tell which loop level ticked.
   #[test]
   fn next_index_reports_which_group_advanced() {
      let shape = Shape::new(vec![2, 2]);
      let mut index = [0_i32; MAX_RANK];
      assert_eq!(
         get_next_index(&shape, &[1_i32], &[0_i32], &mut index),
         NextIndex::Inner
      );
      assert_eq!(index[..2], [0_i32, 1_i32]);
      assert_eq!(
         get_next_index(&shape, &[1_i32], &[0_i32], &mut index),
         NextIndex::Outer,
         "the inner span wraps, so the outer one takes the carry"
      );
      assert_eq!(index[..2], [1_i32, 0_i32]);
      get_next_index(&shape, &[1_i32], &[0_i32], &mut index);
      assert_eq!(
         get_next_index(&shape, &[1_i32], &[0_i32], &mut index),
         NextIndex::Exhausted
      );
      assert_eq!(index[..2], [0_i32, 0_i32]);
   }

   /// Float32 is ordered by `bgeu` on the raw bit pattern, so a negative value
   /// sorts above every positive one.
   #[test]
   fn orders_floats_by_unsigned_bit_pattern() {
      let input = tensor(
         TensorType::Float32,
         vec![3],
         &f32_bytes(&[1.0_f32, -2.0_f32, 3.0_f32]),
      );
      let mut values = zero_tensor(TensorType::Float32, vec![3]);
      let mut indices = zero_tensor(TensorType::Int32, vec![3]);
      assert!(
         top_k(&input, &mut values, &mut indices).is_ok(),
         "raw bit ordered TopK should succeed"
      );
      assert_eq!(read_f32(&values), [-2.0_f32, 3.0_f32, 1.0_f32]);
      assert_eq!(indices.bytes(), i32_bytes(&[1_i32, 2_i32, 0_i32]));
   }
}
