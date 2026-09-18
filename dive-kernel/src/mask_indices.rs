//! Dense index extraction from boolean masks, and the scalar select fallback.

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::tensor::{
   ResolvedTensor,
   type_size_bytes,
};

#[inline]
pub fn mask_indices(mask: &[u8], indices: &mut [i64]) -> usize {
   let mut written = 0_usize;
   for (pos, bit) in mask.iter().enumerate() {
      if *bit != 0 {
         if written < indices.len() {
            indices[written] = pos as i64;
         }
         written += 1;
      }
   }
   written.min(indices.len())
}

#[inline]
pub fn unioned_mask_indices(masks: &[&[u8]], indices: &mut [i64]) -> usize {
   let Some(len) = masks.iter().map(|mask| mask.len()).min() else {
      return 0;
   };
   let mut written = 0_usize;
   for pos in 0..len {
      if masks.iter().any(|mask| mask[pos] != 0) {
         if written < indices.len() {
            indices[written] = pos as i64;
         }
         written += 1;
      }
   }
   written.min(indices.len())
}

#[inline]
pub fn select_fallback(
   cond: &ResolvedTensor,
   lhs: &ResolvedTensor,
   rhs: &ResolvedTensor,
   out: &mut ResolvedTensor,
) -> Status {
   const FILE: &str = "select.cc";

   if cond.element_count != lhs.element_count
      || lhs.element_count != rhs.element_count
      || rhs.element_count != out.element_count
   {
      dive::log("select shape mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   if lhs.tensor_type != rhs.tensor_type || lhs.tensor_type != out.tensor_type {
      dive::log("select dtype mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   let elem = type_size_bytes(lhs.tensor_type);
   let cond_elem = type_size_bytes(cond.tensor_type);
   cond.require_bytes(cond.element_count * cond_elem, FILE);
   lhs.require_bytes(lhs.element_count * elem, FILE);
   rhs.require_bytes(rhs.element_count * elem, FILE);
   out.require_bytes(out.element_count * elem, FILE);

   let count = lhs.element_count;
   let pick = cond.bytes();
   let first = lhs.bytes();
   let second = rhs.bytes();
   let dest = out.bytes_mut();
   for idx in 0..count {
      let src = if pick[idx * cond_elem] != 0 {
         first
      } else {
         second
      };
      dest[idx * elem..idx * elem + elem].copy_from_slice(&src[idx * elem..idx * elem + elem]);
   }
   Status::ok()
}
