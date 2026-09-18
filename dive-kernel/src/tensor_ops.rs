use alloc::{
   format,
   string::{
      String,
      ToString as _,
   },
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
   type_size_bytes,
};

#[inline]
#[must_use]
pub fn validate_shape(shape: &Shape) -> Status {
   if shape.dims.iter().any(|dim| *dim <= 0) {
      return Status::new(Code::InvalidArgument);
   }
   Status::ok()
}

#[inline]
#[must_use]
pub fn shapes_equal(left: &Shape, right: &Shape) -> bool {
   left.dims == right.dims
}

#[inline]
#[must_use]
pub fn shape_to_str(shape: &Shape) -> String {
   let dims = shape
      .dims
      .iter()
      .map(i64::to_string)
      .collect::<Vec<String>>()
      .join(", ");
   format!("[{dims}]")
}

#[inline]
pub fn shape_to_str_into(shape: &Shape, buf: &mut [u8]) -> usize {
   let text = shape_to_str(shape);
   let bytes = text.as_bytes();
   let count = bytes.len().min(buf.len());
   buf[..count].copy_from_slice(&bytes[..count]);
   count
}

pub struct PadSlice {
   pub inner_bytes: usize,
   pub chunk_bytes: usize,
   pub split_dim:   usize,
}

/// Finds the innermost continuous slice of the tensor inside the target shape.
///
/// # Panics
///
/// Panics when the tensor and target ranks differ.
#[inline]
#[must_use]
pub fn pad_continuous_slice(tensor: &ResolvedTensor, target: &Shape) -> PadSlice {
   let elem = type_size_bytes(tensor.tensor_type);
   let rank = tensor.shape.rank();
   assert_eq!(rank, target.rank(), "pad slice needs matching ranks");
   let mut inner_elems = 1_usize;
   for pos in (0..rank).rev() {
      if tensor.shape.dims[pos] != target.dims[pos] {
         let inner_bytes = elem.saturating_mul(inner_elems);
         let wide = usize::try_from(target.dims[pos]).unwrap_or(0);
         return PadSlice {
            inner_bytes,
            chunk_bytes: wide.saturating_mul(inner_bytes),
            split_dim: pos,
         };
      }
      inner_elems =
         inner_elems.saturating_mul(usize::try_from(tensor.shape.dims[pos]).unwrap_or(0));
   }
   let inner_bytes = elem.saturating_mul(inner_elems);
   PadSlice {
      inner_bytes,
      chunk_bytes: inner_bytes,
      split_dim: 0,
   }
}

/// Increments a mixed-radix index vector, returning false on wraparound.
///
/// # Panics
///
/// Panics when the limits and indices slices have different lengths.
#[inline]
pub fn loop_index_increment(limits: &[i64], indices: &mut [i32]) -> bool {
   if limits.is_empty() {
      return false;
   }
   assert_eq!(
      limits.len(),
      indices.len(),
      "gather index needs matching limits and indices"
   );
   for pos in (0..limits.len()).rev() {
      indices[pos] += 1_i32;
      if i64::from(indices[pos]) < limits[pos] {
         return true;
      }
      indices[pos] = 0_i32;
   }
   false
}

#[inline]
#[must_use]
pub const fn status_to_string(status: &Status) -> &'static str {
   match status.code() {
      Code::Ok => "OK",
      Code::Cancelled => "CANCELLED",
      Code::Unknown => "UNKNOWN",
      Code::InvalidArgument => "INVALID_ARGUMENT",
      Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
      Code::NotFound => "NOT_FOUND",
      Code::AlreadyExists => "ALREADY_EXISTS",
      Code::PermissionDenied => "PERMISSION_DENIED",
      Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
      Code::FailedPrecondition => "FAILED_PRECONDITION",
      Code::Aborted => "ABORTED",
      Code::OutOfRange => "OUT_OF_RANGE",
      Code::Unimplemented => "UNIMPLEMENTED",
      Code::Internal => "INTERNAL",
      Code::Unavailable => "UNAVAILABLE",
      Code::DataLoss => "DATA_LOSS",
      Code::Unauthenticated => "UNAUTHENTICATED",
   }
}

#[inline]
pub const fn status_update(dst: &mut Status, src: &Status) {
   if dst.is_ok() && !src.is_ok() {
      *dst = *src;
   }
}

#[inline]
pub fn fw_check_ok_failed(code: Code, file: &str, line: i32) -> ! {
   let _: i32 = line;
   dive::log(file, "status_macros.cc");
   dive::cease(code as i32, "status_macros.cc");
}
