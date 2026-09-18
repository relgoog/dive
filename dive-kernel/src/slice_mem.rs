//! Strided slice and roll kernels.

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
   extent,
   type_size_bytes,
};

/// `SliceOffsetGenerator` holds six 48-byte arrays, so rank tops out at six.
/// The constructor ceases 16 below one and 17 at seven or more.
pub const MAX_SLICE_RANK: usize = 6;

/// A size of -1 means "run to the end of the dim from `starts`".
pub const SIZE_TO_END: i64 = -1;

/// Source tag the constructor passes to `DiveRuntime_Cease`.
const FILE: &str = "slice_offset_generator.cc";

/// Row-major element strides for a shape.
fn row_major_strides(shape: &[i64]) -> Vec<i64> {
   let mut strides = vec![1_i64; shape.len()];
   let mut acc = 1_i64;
   for i in (0..shape.len()).rev() {
      strides[i] = acc;
      acc = acc.saturating_mul(shape[i]);
   }
   strides
}

#[inline]
#[must_use]
pub const fn elem_size_of(tensor: &ResolvedTensor) -> usize {
   type_size_bytes(tensor.tensor_type)
}

#[inline]
#[must_use]
pub const fn check_slice_args(rank: usize, starts_len: usize, sizes_len: usize) -> Status {
   if starts_len == rank && sizes_len == rank {
      Status::ok()
   } else {
      Status::new(Code::InvalidArgument)
   }
}

/// Resolves one slice extent, expanding the `-1` sentinel against the shape.
#[inline]
#[must_use]
pub const fn resolve_extent(dim: i64, start: i64, size: i64) -> i64 {
   if size == SIZE_TO_END {
      dim - start
   } else {
      size
   }
}

pub struct ContinuousSlice {
   /// Bytes in the contiguous inner run.
   pub inner_bytes: usize,
   /// First dim where the slice narrows.
   pub split_dim:   usize,
   /// Outer vectors covered by the view.
   pub vectors:     usize,
}

impl ContinuousSlice {
   #[inline]
   #[must_use]
   pub const fn total_bytes(&self) -> usize {
      self.inner_bytes * self.vectors
   }

   /// Bytes copied per generator step.
   ///
   /// `InsertSlice` folds the first partial dim into the run as
   /// `v18 = size * inner`, so the run is wider than the fully matching
   /// trailing dims alone.
   #[inline]
   #[must_use]
   pub fn run_bytes(&self, shape: &[i64], starts: &[i64], sizes: &[i64]) -> usize {
      if self.split_dim == 0 {
         return self.inner_bytes;
      }
      let dim = self.split_dim - 1;
      extent(resolve_extent(shape[dim], starts[dim], sizes[dim])).saturating_mul(self.inner_bytes)
   }
}

/// Splits a slice shape into an inner contiguous run and outer vector count.
///
/// # Panics
///
/// Panics when the shape and slice shape ranks differ.
#[inline]
#[must_use]
pub fn get_continuous_slice(
   elem_size: usize,
   shape: &[i64],
   starts: &[i64],
   sizes: &[i64],
) -> ContinuousSlice {
   assert_eq!(shape.len(), sizes.len(), "shape and sizes ranks must match");
   assert_eq!(
      shape.len(),
      starts.len(),
      "shape and starts ranks must match"
   );
   let rank = shape.len();
   let mut inner_elems = 1_usize;
   let mut split_dim = 0_usize;
   for i in (0..rank).rev() {
      let size = resolve_extent(shape[i], starts[i], sizes[i]);
      if shape[i] != size {
         split_dim = i + 1;
         break;
      }
      inner_elems = inner_elems.saturating_mul(extent(size));
   }
   let vectors = (0..split_dim).fold(1_usize, |acc, i| {
      acc.saturating_mul(extent(resolve_extent(shape[i], starts[i], sizes[i])))
   });
   ContinuousSlice {
      inner_bytes: inner_elems * elem_size,
      split_dim,
      vectors,
   }
}

/// Walks the outer positions of a slice view, one element offset per step.
///
/// The object keeps a counter, an accumulated element offset, a stride, and a
/// resolved extent per dim. The innermost dim is never stepped, since it is the
/// contiguous run the caller copies in one go.
pub struct SliceOffsetGenerator {
   /// Per-dim counter, `this + 0` in the C++ object.
   index:    Vec<i64>,
   /// Per-dim element contribution, `this + 48`.
   accum:    Vec<i64>,
   /// Row-major stride in elements, `this + 96`.
   step:     Vec<i64>,
   /// Resolved slice extent, `this + 168`.
   extents:  Vec<i64>,
   /// Slice origin, reloaded on carry.
   starts:   Vec<i64>,
   /// Offsets this generator will yield, `this + 216`.
   total:    u64,
   /// Offsets yielded so far, `this + 224`.
   produced: u64,
}

impl SliceOffsetGenerator {
   /// Builds a generator over `rank` dims of `shape`, honouring `-1` sizes.
   #[inline]
   #[must_use]
   pub fn new(rank: usize, shape: &[i64], starts: &[i64], sizes: &[i64]) -> Self {
      if rank == 0 {
         dive::cease(16, FILE);
      }
      if rank > MAX_SLICE_RANK {
         dive::cease(17, FILE);
      }
      let extents = (0..rank)
         .map(|i| resolve_extent(shape[i], starts[i], sizes[i]))
         .collect::<Vec<i64>>();

      let mut step = vec![1_i64; rank];
      for i in (0..rank - 1).rev() {
         step[i] = shape[i + 1].saturating_mul(step[i + 1]);
      }
      let accum = (0..rank)
         .map(|i| starts[i].saturating_mul(step[i]))
         .collect::<Vec<i64>>();

      // The innermost extent is the contiguous run, so it is not a loop bound.
      let total = extents[..rank - 1].iter().fold(1_u64, |acc, size| {
         acc.saturating_mul(u64::try_from(*size).unwrap_or(0))
      });

      Self {
         index: vec![0_i64; rank],
         accum,
         step,
         extents,
         starts: starts[..rank].to_vec(),
         total,
         produced: 0,
      }
   }

   #[inline]
   #[must_use]
   pub const fn total(&self) -> u64 {
      self.total
   }

   #[inline]
   #[must_use]
   pub const fn produced(&self) -> u64 {
      self.produced
   }

   #[inline]
   #[must_use]
   pub const fn is_exhausted(&self) -> bool {
      self.produced >= self.total
   }

   /// Yields the current element offset, then carries the odometer.
   ///
   /// The C++ always returns and always bumps the produced count, so callers
   /// gate on `produced < total` rather than on a sentinel.
   #[inline]
   pub fn get_next_offset(&mut self) -> i64 {
      let offset = self.accum.iter().sum::<i64>();
      self.produced = self.produced.saturating_add(1);

      let rank = self.index.len();
      for dim in (0..rank.saturating_sub(1)).rev() {
         let next = self.index[dim] + 1;
         self.index[dim] = next;
         self.accum[dim] = self.accum[dim].saturating_add(self.step[dim]);
         if next != self.extents[dim] {
            break;
         }
         self.index[dim] = 0;
         self.accum[dim] = self.starts[dim].saturating_mul(self.step[dim]);
      }
      offset
   }

   /// Consumes the second innermost dim as one vectorised run.
   ///
   /// Unlike `get_next_offset` this advances `produced` by the run length
   /// rather than by one, and the carry starts two dims in. Ceases 59 when
   /// there is no second innermost dim to vectorise.
   #[inline]
   pub fn get_next_vector_base_offset(&mut self) -> VectorRun {
      let rank = self.index.len();
      if rank <= 1 {
         dive::cease(59, FILE);
      }
      let run = VectorRun {
         base:  self.accum.iter().sum::<i64>(),
         step:  self.step[rank - 2],
         count: self.extents[rank - 2],
      };
      self.produced = self
         .produced
         .saturating_add(u64::try_from(run.count).unwrap_or(0));

      for dim in (0..rank.saturating_sub(2)).rev() {
         let next = self.index[dim] + 1;
         self.index[dim] = next;
         self.accum[dim] = self.accum[dim].saturating_add(self.step[dim]);
         if next != self.extents[dim] {
            break;
         }
         self.index[dim] = 0;
         self.accum[dim] = self.starts[dim].saturating_mul(self.step[dim]);
      }
      run
   }
}

/// One vectorised run: a base element offset, the stride between consecutive
/// rows, and how many rows the run covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorRun {
   pub base:  i64,
   pub step:  i64,
   pub count: i64,
}

/// Copies bytes from a compact source into a strided destination region.
///
/// # Panics
///
/// Panics when the destination, start, and size ranks differ.
#[inline]
pub fn insert_slice(
   dst: &mut [u8],
   dst_shape: &[i64],
   src: &[u8],
   starts: &[i64],
   sizes: &[i64],
   elem_size: usize,
) -> usize {
   let rank = dst_shape.len();
   assert_eq!(
      rank,
      starts.len(),
      "destination shape and starts ranks must match"
   );
   assert_eq!(
      rank,
      sizes.len(),
      "destination shape and sizes ranks must match"
   );
   if rank == 0 {
      dst[..elem_size].copy_from_slice(&src[..elem_size]);
      return elem_size;
   }
   let view = get_continuous_slice(elem_size, dst_shape, starts, sizes);
   let total = view.total_bytes();
   if view.split_dim == 0 {
      let strides = row_major_strides(dst_shape);
      let at = base_offset(starts, &strides, elem_size);
      dst[at..at + total].copy_from_slice(&src[..total]);
      return total;
   }
   let run = view.run_bytes(dst_shape, starts, sizes);
   let mut offsets = SliceOffsetGenerator::new(view.split_dim, dst_shape, starts, sizes);
   let mut cursor = 0_usize;
   while !offsets.is_exhausted() {
      let at = extent(offsets.get_next_offset()) * elem_size;
      dst[at..at + run].copy_from_slice(&src[cursor..cursor + run]);
      cursor += run;
   }
   total
}

/// Copies bytes from a strided source region into a compact destination.
///
/// # Panics
///
/// Panics when the source, start, and size ranks differ.
#[inline]
pub fn extract_slice(
   src: &[u8],
   src_shape: &[i64],
   dst: &mut [u8],
   starts: &[i64],
   sizes: &[i64],
   elem_size: usize,
) -> usize {
   let rank = src_shape.len();
   assert_eq!(
      rank,
      starts.len(),
      "source shape and starts ranks must match"
   );
   assert_eq!(rank, sizes.len(), "source shape and sizes ranks must match");
   if rank == 0 {
      dst[..elem_size].copy_from_slice(&src[..elem_size]);
      return elem_size;
   }
   let view = get_continuous_slice(elem_size, src_shape, starts, sizes);
   let total = view.total_bytes();
   if view.split_dim == 0 {
      let strides = row_major_strides(src_shape);
      let at = base_offset(starts, &strides, elem_size);
      dst[..total].copy_from_slice(&src[at..at + total]);
      return total;
   }
   let run = view.run_bytes(src_shape, starts, sizes);
   let mut offsets = SliceOffsetGenerator::new(view.split_dim, src_shape, starts, sizes);
   let mut cursor = 0_usize;
   while !offsets.is_exhausted() {
      let at = extent(offsets.get_next_offset()) * elem_size;
      dst[cursor..cursor + run].copy_from_slice(&src[at..at + run]);
      cursor += run;
   }
   total
}

/// Byte offset of the slice origin.
fn base_offset(starts: &[i64], strides: &[i64], elem_size: usize) -> usize {
   let elems = starts
      .iter()
      .zip(strides.iter())
      .fold(0_i64, |acc, (start, stride)| {
         acc.saturating_add(start.saturating_mul(*stride))
      });
   extent(elems).saturating_mul(elem_size)
}

/// Rolls tensor bytes along one axis by a signed shift.
#[inline]
pub fn roll(src: &[u8], axis: usize, shift: i64, shape: &[i64], elem_size: usize, dst: &mut [u8]) {
   let axis_len = extent(shape[axis]);
   if axis_len == 0 {
      return;
   }
   let total = shape
      .iter()
      .fold(1_usize, |acc, dim| acc.saturating_mul(extent(*dim)))
      * elem_size;
   let wide = i64::try_from(axis_len).unwrap_or(1);
   let shift_elems = extent((shift % wide + wide) % wide);
   if shift_elems == 0 {
      dst[..total].copy_from_slice(&src[..total]);
      return;
   }
   let inner = shape[axis + 1..]
      .iter()
      .fold(1_usize, |acc, dim| acc.saturating_mul(extent(*dim)));
   let outer = shape[..axis]
      .iter()
      .fold(1_usize, |acc, dim| acc.saturating_mul(extent(*dim)));
   let run = inner * elem_size;
   for block in 0..outer {
      for pos in 0..axis_len {
         let from = (pos + axis_len - shift_elems) % axis_len * run + block * axis_len * run;
         let to = (block * axis_len + pos) * run;
         dst[to..to + run].copy_from_slice(&src[from..from + run]);
      }
   }
}

#[cfg(test)]
mod tests {
   use alloc::{
      vec,
      vec::Vec,
   };

   use crate::slice_mem::{
      SIZE_TO_END,
      SliceOffsetGenerator,
   };

   /// The constructor takes `(rank, shape, starts, sizes)`, resolves a `-1`
   /// size against the shape, and iterates the outer dims only. The
   /// innermost dim is the contiguous run the caller copies in one go.
   #[test]
   fn offset_generator_walks_the_outer_dims() {
      let shape = [2_i64, 3_i64, 4_i64];
      let starts = [0_i64, 0_i64, 0_i64];
      let sizes = [2_i64, 3_i64, 4_i64];
      let mut offsets = SliceOffsetGenerator::new(3, &shape, &starts, &sizes);
      assert_eq!(offsets.total(), 6, "2 * 3, the innermost dim is the run");
      let mut seen = Vec::new();
      while !offsets.is_exhausted() {
         seen.push(offsets.get_next_offset());
      }
      assert_eq!(seen, vec![0_i64, 4, 8, 12, 16, 20]);
      assert_eq!(offsets.produced(), 6);
   }

   /// A size of -1 resolves to `dims[i] - starts[i]`.
   #[test]
   fn offset_generator_expands_the_size_sentinel() {
      let shape = [4_i64, 5_i64];
      let starts = [1_i64, 2_i64];
      let open = SliceOffsetGenerator::new(2, &shape, &starts, &[SIZE_TO_END, SIZE_TO_END]);
      let spelled = SliceOffsetGenerator::new(2, &shape, &starts, &[3_i64, 3_i64]);
      assert_eq!(open.total(), spelled.total());
      assert_eq!(open.total(), 3, "4 - 1 outer positions");
   }

   /// Rank is validated against the six 48-byte arrays in the C++ object.
   #[test]
   #[should_panic(expected = "DiveRuntime_Cease 17")]
   fn offset_generator_rejects_rank_seven() {
      let shape = [1_i64; 7];
      let built = SliceOffsetGenerator::new(7, &shape, &[0_i64; 7], &[1_i64; 7]);
      drop(built);
   }

   /// `GetNextVectorBaseOffset` hands back the second innermost dim as a run
   /// and advances `produced` by its length, not by one.
   #[test]
   fn offset_generator_yields_a_vector_run() {
      let shape = [2_i64, 3_i64, 4_i64];
      let mut offsets = SliceOffsetGenerator::new(3, &shape, &[0_i64; 3], &shape);
      let run = offsets.get_next_vector_base_offset();
      assert_eq!(run.base, 0_i64);
      assert_eq!(run.step, 4_i64, "stride of the second innermost dim");
      assert_eq!(run.count, 3_i64, "its extent");
      assert_eq!(offsets.produced(), 3);
      assert_eq!(offsets.get_next_vector_base_offset().base, 12_i64);
   }

   #[test]
   #[should_panic(expected = "DiveRuntime_Cease 59")]
   fn vector_base_offset_needs_two_dims() {
      let mut offsets = SliceOffsetGenerator::new(1, &[4_i64], &[0_i64], &[4_i64]);
      let run = offsets.get_next_vector_base_offset();
      assert_eq!(run.count, 0_i64, "unreachable, the call ceases first");
   }
}
