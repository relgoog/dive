//! Rotate, crop and resize with bilinear resampling.

use core::f32::math::{
   floor,
   mul_add,
};

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::warp_affine::sample;

#[must_use]
#[inline]
pub fn produce_discrete_sample_point(
   src_center: [f32; 2],
   dst_center: [f32; 2],
   dst_point: [f32; 2],
   cos_theta: f32,
   sin_theta: f32,
   scale: f32,
) -> [f32; 2] {
   let dx = (dst_point[0] - dst_center[0]) * scale;
   let dy = (dst_point[1] - dst_center[1]) * scale;
   let rotated_x = mul_add(-sin_theta, dy, cos_theta * dx);
   let rotated_y = mul_add(cos_theta, dy, sin_theta * dx);
   [
      rotated_x + src_center[0] - 0.5_f32,
      rotated_y + src_center[1] - 0.5_f32,
   ]
}

#[inline]
pub fn produce_discrete_sample_points(
   src_center: [f32; 2],
   dst_center: [f32; 2],
   dst_points: &[[f32; 2]],
   cos_theta: f32,
   sin_theta: f32,
   scale: f32,
   out: &mut [[f32; 2]],
) -> Status {
   if dst_points.len() != out.len() {
      dive::log("sample points shape mismatch", "rotate_crop_resize.cc");
      return Status::new(Code::InvalidArgument);
   }
   for (slot, point) in out.iter_mut().zip(dst_points.iter()) {
      *slot =
         produce_discrete_sample_point(src_center, dst_center, *point, cos_theta, sin_theta, scale);
   }
   Status::ok()
}

#[inline]
pub fn produce_discrete_sample_points_hoisted(
   src_center: [f32; 2],
   dst_center: [f32; 2],
   dst_points: &[[f32; 2]],
   mat: &[f32; 4],
   out: &mut [[f32; 2]],
) -> Status {
   if dst_points.len() != out.len() {
      dive::log("hoisted points shape mismatch", "rotate_crop_resize.cc");
      return Status::new(Code::InvalidArgument);
   }
   for (slot, point) in out.iter_mut().zip(dst_points.iter()) {
      let dx = point[0] - dst_center[0];
      let dy = point[1] - dst_center[1];
      *slot = [
         mul_add(mat[1], dy, mat[0] * dx) + src_center[0] - 0.5_f32,
         mul_add(mat[3], dy, mat[2] * dx) + src_center[1] - 0.5_f32,
      ];
   }
   Status::ok()
}

#[inline]
pub fn clamp_sample_points(points: &mut [[f32; 2]], size: [i32; 2]) {
   let max_x = (size[0] - 1_i32).max(0_i32) as f32;
   let max_y = (size[1] - 1_i32).max(0_i32) as f32;
   for point in points.iter_mut() {
      point[0] = point[0].clamp(0.0, max_x);
      point[1] = point[1].clamp(0.0, max_y);
   }
}

/// Bilinearly interpolates the source at fractional coordinates.
fn bilinear_u8(src: &[u8], src_w: usize, src_h: usize, pos_x: f32, pos_y: f32) -> f32 {
   let x0 = floor(pos_x) as i64;
   let y0 = floor(pos_y) as i64;
   let frac_x = pos_x - x0 as f32;
   let frac_y = pos_y - y0 as f32;
   let top_left = f32::from(sample(src, src_w, src_h, x0, y0));
   let top_right = f32::from(sample(src, src_w, src_h, x0 + 1, y0));
   let bottom_left = f32::from(sample(src, src_w, src_h, x0, y0 + 1));
   let bottom_right = f32::from(sample(src, src_w, src_h, x0 + 1, y0 + 1));
   top_left * (1.0 - frac_x) * (1.0 - frac_y)
      + top_right * frac_x * (1.0 - frac_y)
      + bottom_left * (1.0 - frac_x) * frac_y
      + bottom_right * frac_x * frac_y
}

#[expect(
   clippy::too_many_arguments,
   reason = "kernel signature mirrors C++ warp entry point"
)]
#[inline]
pub fn rotate_crop_resize(
   src: &[u8],
   src_w: usize,
   src_h: usize,
   dst: &mut [f32],
   dst_w: usize,
   dst_h: usize,
   src_center: [f32; 2],
   dst_center: [f32; 2],
   cos_theta: f32,
   sin_theta: f32,
   scale: f32,
) -> Status {
   if src.len() < src_w * src_h || dst.len() < dst_w * dst_h {
      dive::log("rotate crop shape mismatch", "rotate_crop_resize.cc");
      return Status::new(Code::InvalidArgument);
   }
   for oy in 0..dst_h {
      for ox in 0..dst_w {
         let point = produce_discrete_sample_point(
            src_center,
            dst_center,
            [ox as f32, oy as f32],
            cos_theta,
            sin_theta,
            scale,
         );
         dst[oy * dst_w + ox] = bilinear_u8(src, src_w, src_h, point[0], point[1]);
      }
   }
   Status::ok()
}
