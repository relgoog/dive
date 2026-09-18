//! Inverse affine warping with nearest-neighbour resampling.

use alloc::{
   vec,
   vec::Vec,
};
use core::f32::math::round;

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::tensor::{
   ResolvedTensor,
   TensorType,
   type_size_bytes,
};

/// Samples one pixel with zero fill outside the image bounds.
#[inline]
pub fn sample<T>(src: &[T], src_w: usize, src_h: usize, x: i64, y: i64) -> T
where
   T: Copy + Default,
{
   let (Ok(col), Ok(row)) = (usize::try_from(x), usize::try_from(y)) else {
      return T::default();
   };
   if col >= src_w || row >= src_h {
      return T::default();
   }
   src[row * src_w + col]
}

/// Inverse-maps one destination raster position through the affine matrix.
fn inv_affine_source(matrix: &[f32; 6], ox: usize, oy: usize) -> (i64, i64) {
   let col = ox as f32;
   let row = oy as f32;
   let sx = round(matrix[0] * col + matrix[1] * row + matrix[2]) as i64;
   let sy = round(matrix[3] * col + matrix[4] * row + matrix[5]) as i64;
   (sx, sy)
}

#[inline]
pub fn warp_inv_affine_u8(
   src: &[u8],
   src_w: usize,
   src_h: usize,
   matrix: &[f32; 6],
   dst: &mut [u8],
   dst_w: usize,
   dst_h: usize,
) -> Status {
   if src.len() < src_w * src_h || dst.len() < dst_w * dst_h {
      dive::log("warp u8 shape mismatch", "custom_warp_affine_kernel.cc");
      return Status::new(Code::InvalidArgument);
   }
   for oy in 0..dst_h {
      for ox in 0..dst_w {
         let (sx, sy) = inv_affine_source(matrix, ox, oy);
         dst[oy * dst_w + ox] = sample(src, src_w, src_h, sx, sy);
      }
   }
   Status::ok()
}

#[inline]
pub fn warp_inv_affine_f32(
   src: &[f32],
   src_w: usize,
   src_h: usize,
   matrix: &[f32; 6],
   dst: &mut [f32],
   dst_w: usize,
   dst_h: usize,
) -> Status {
   if src.len() < src_w * src_h || dst.len() < dst_w * dst_h {
      dive::log("warp f32 shape mismatch", "custom_warp_affine_kernel.cc");
      return Status::new(Code::InvalidArgument);
   }
   for oy in 0..dst_h {
      for ox in 0..dst_w {
         let (sx, sy) = inv_affine_source(matrix, ox, oy);
         dst[oy * dst_w + ox] = sample(src, src_w, src_h, sx, sy);
      }
   }
   Status::ok()
}

#[inline]
pub fn custom_warp_affine_kernel(
   src: &ResolvedTensor,
   matrix: &[f32; 6],
   dst: &mut ResolvedTensor,
) -> Status {
   const FILE: &str = "custom_warp_affine_kernel.cc";

   if src.shape.rank() != 2 || dst.shape.rank() != 2 {
      dive::log("warp rank mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   if src.tensor_type != dst.tensor_type {
      dive::log("warp dtype mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }
   let src_h = usize::try_from(src.shape.dims[0]).unwrap_or(0);
   let src_w = usize::try_from(src.shape.dims[1]).unwrap_or(0);
   let dst_h = usize::try_from(dst.shape.dims[0]).unwrap_or(0);
   let dst_w = usize::try_from(dst.shape.dims[1]).unwrap_or(0);
   let elem = type_size_bytes(src.tensor_type);
   src.require_bytes(src.element_count * elem, FILE);
   dst.require_bytes(dst.element_count * elem, FILE);

   match src.tensor_type {
      TensorType::UInt8 => {
         let pixels = src.bytes().to_vec();
         warp_inv_affine_u8(&pixels, src_w, src_h, matrix, dst.bytes_mut(), dst_w, dst_h)
      },
      TensorType::Float32 => {
         let pixels = src
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|cell| f32::from_le_bytes(*cell))
            .collect::<Vec<f32>>();
         let mut out = vec![0.0_f32; dst_w * dst_h];
         let status = warp_inv_affine_f32(&pixels, src_w, src_h, matrix, &mut out, dst_w, dst_h);
         for (cell, value) in dst
            .bytes_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(out.iter())
         {
            cell.copy_from_slice(&value.to_le_bytes());
         }
         status
      },
      TensorType::Bool
      | TensorType::Int8
      | TensorType::Int16
      | TensorType::Int32
      | TensorType::Float16
      | TensorType::Bfloat
      | TensorType::UInt16
      | TensorType::UInt32
      | TensorType::Int64
      | TensorType::UInt64
      | TensorType::Float64 => {
         dive::log("warp unsupported type", FILE);
         Status::new(Code::Unimplemented)
      },
   }
}
