//! The four-way integral pooling kernel.

use core::f32::math::mul_add;

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::{
   accessor::{
      ElementType,
      TensorAccessor,
   },
   tensor::{
      ResolvedTensor,
      TensorType,
   },
};

/// Cease codes the pooling kernel raises before it touches any data.
///
/// 104 and 105 fire when the three element counts disagree, 107 when the
/// output tensor type is not `TensorType::Float16`.
pub const POOLING_ELEMENT_COUNT_MISMATCH: i32 = 104;
pub const POOLING_ELEMENT_COUNT_MISMATCH_THIRD: i32 = 105;
pub const POOLING_NOT_FLOAT16: i32 = 107;

pub struct IntegralPoolingContext<'tensor> {
   pub coefficients:             &'tensor ResolvedTensor,
   pub coefficient_element_type: ElementType,
   pub source:                   &'tensor ResolvedTensor,
   pub source_element_type:      ElementType,
   pub output:                   &'tensor mut ResolvedTensor,
   pub output_element_type:      ElementType,
   pub outer_count:              i32,
   pub inner_count:              i32,
   pub height:                   i32,
   pub width:                    i32,
   pub channels:                 i32,
}

/// Runs one directional recurrence over a pooling line.
fn accumulate_pooling_line(
   coefficients: &TensorAccessor<impl AsRef<[u8]>>,
   source: &TensorAccessor<impl AsRef<[u8]>>,
   output: &mut TensorAccessor<impl AsRef<[u8]> + AsMut<[u8]>>,
   start: usize,
   length: usize,
   stride: usize,
   reverse: bool,
) {
   let boundary = if reverse {
      start + (length - 1) * stride
   } else {
      start
   };
   let boundary_value = source.load_f64(boundary) as f32;
   output.store_f64(boundary, f64::from(boundary_value));

   if reverse {
      for step in (0..length - 1).rev() {
         let current = start + step * stride;
         let previous = current + stride;
         let value = mul_add(
            coefficients.load_f64(current) as f32,
            output.load_f64(previous) as f32,
            source.load_f64(current) as f32,
         );
         output.store_f64(current, f64::from(value));
      }
   } else {
      for step in 1..length {
         let current = start + step * stride;
         let previous = current - stride;
         let value = mul_add(
            coefficients.load_f64(current) as f32,
            output.load_f64(previous) as f32,
            source.load_f64(current) as f32,
         );
         output.store_f64(current, f64::from(value));
      }
   }
}

/// Validates pooling dimensions and computes their product.
fn pooling_dimensions(raw: [i32; 5]) -> Result<([usize; 5], usize), &'static str> {
   let [Ok(outer), Ok(inner), Ok(height), Ok(width), Ok(channels)] = raw.map(usize::try_from)
   else {
      return Err("pooling dimension out of range");
   };
   let dimensions = [outer, inner, height, width, channels];
   let Some(element_count) = dimensions
      .iter()
      .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
   else {
      return Err("pooling dimensions overflow");
   };
   Ok((dimensions, element_count))
}

/// Runs the four directional pooling recurrences over every plane.
fn accumulate_pooling_planes(
   coefficients: &TensorAccessor<&[u8]>,
   source: &TensorAccessor<&[u8]>,
   output: &mut TensorAccessor<&mut [u8]>,
   dimensions: [usize; 5],
) {
   let [outer_count, inner_count, height, width, channels] = dimensions;
   let group = channels / 4;
   let planes = outer_count * inner_count;
   let plane_stride = height * width * channels;

   for plane in 0..planes {
      let plane_start = plane * plane_stride;

      for column in 0..width {
         for channel in 0..group {
            let downward_start = plane_start + column * channels + channel;
            accumulate_pooling_line(
               coefficients,
               source,
               output,
               downward_start,
               height,
               width * channels,
               false,
            );

            let upward_start = plane_start + column * channels + group + channel;
            accumulate_pooling_line(
               coefficients,
               source,
               output,
               upward_start,
               height,
               width * channels,
               true,
            );
         }
      }

      for row in 0..height {
         for channel in 0..group {
            let rightward_start = plane_start + row * width * channels + 2 * group + channel;
            accumulate_pooling_line(
               coefficients,
               source,
               output,
               rightward_start,
               width,
               channels,
               false,
            );

            let leftward_start = plane_start + row * width * channels + 3 * group + channel;
            accumulate_pooling_line(
               coefficients,
               source,
               output,
               leftward_start,
               width,
               channels,
               true,
            );
         }
      }
   }
}

#[inline]
#[must_use]
pub fn integral_pooling_four_way_kernel(context: IntegralPoolingContext<'_>) -> Status {
   const FILE: &str = "integral_pooling_kernel.cc";

   let IntegralPoolingContext {
      coefficients,
      coefficient_element_type,
      source,
      source_element_type,
      output,
      output_element_type,
      outer_count: raw_outer_count,
      inner_count: raw_inner_count,
      height: raw_height,
      width: raw_width,
      channels: raw_channels,
   } = context;

   if coefficients.element_count != source.element_count {
      dive::log("pooling element count mismatch", FILE);
      dive::cease(POOLING_ELEMENT_COUNT_MISMATCH, FILE);
   }
   if coefficients.element_count != output.element_count {
      dive::log("pooling element count mismatch", FILE);
      dive::cease(POOLING_ELEMENT_COUNT_MISMATCH_THIRD, FILE);
   }
   if output.tensor_type != TensorType::Float16 {
      dive::log("pooling expects float16", FILE);
      dive::cease(POOLING_NOT_FLOAT16, FILE);
   }

   let (dimensions, element_count) = match pooling_dimensions([
      raw_outer_count,
      raw_inner_count,
      raw_height,
      raw_width,
      raw_channels,
   ]) {
      Ok(dimensions) => dimensions,
      Err(message) => {
         dive::log(message, FILE);
         return Status::new(Code::InvalidArgument);
      },
   };
   if element_count == 0 {
      return Status::ok();
   }
   if coefficients.element_count < element_count
      || source.element_count < element_count
      || output.element_count < element_count
   {
      dive::log("pooling tensor too small", FILE);
      return Status::new(Code::InvalidArgument);
   }

   if dimensions[4] / 4 == 0 {
      return Status::ok();
   }

   let coefficient_accessor = TensorAccessor::new(coefficient_element_type, coefficients.bytes());
   let source_accessor = TensorAccessor::new(source_element_type, source.bytes());
   let mut output_accessor = TensorAccessor::new(output_element_type, output.bytes_mut());
   if coefficient_accessor.len() < element_count
      || source_accessor.len() < element_count
      || output_accessor.len() < element_count
   {
      dive::log("pooling tensor too small", FILE);
      return Status::new(Code::InvalidArgument);
   }

   accumulate_pooling_planes(
      &coefficient_accessor,
      &source_accessor,
      &mut output_accessor,
      dimensions,
   );

   Status::ok()
}

#[cfg(test)]
mod tests {
   use alloc::{
      vec,
      vec::Vec,
   };

   use crate::{
      accessor::ElementType,
      integral_pooling::{
         IntegralPoolingContext,
         integral_pooling_four_way_kernel,
      },
      tensor::TensorType,
      test_support::{
         f32_bytes,
         tensor,
      },
   };

   #[test]
   fn integral_pooling_traverses_four_channel_groups() {
      let source_values = [
         1.0_f32, 10.0, 1.0, 10.0, 2.0, 20.0, 2.0, 20.0, 3.0, 30.0, 3.0, 30.0, 4.0, 40.0, 4.0, 40.0,
      ];
      let source_bytes = f32_bytes(&source_values);
      let coefficient_bytes = f32_bytes(&[1.0_f32; 16]);
      let coefficients = tensor(TensorType::Float16, vec![16], &coefficient_bytes);
      let source = tensor(TensorType::Float16, vec![16], &source_bytes);
      let mut output = tensor(TensorType::Float16, vec![16], &[0_u8; 64]);

      assert!(
         integral_pooling_four_way_kernel(IntegralPoolingContext {
            coefficients:             &coefficients,
            coefficient_element_type: ElementType::Float32,
            source:                   &source,
            source_element_type:      ElementType::Float32,
            output:                   &mut output,
            output_element_type:      ElementType::Float32,
            outer_count:              1,
            inner_count:              1,
            height:                   2,
            width:                    2,
            channels:                 4,
         })
         .is_ok(),
         "four way pooling should succeed"
      );
      let values: Vec<f32> = output
         .bytes()
         .as_chunks::<4>()
         .0
         .iter()
         .map(|bytes| f32::from_le_bytes(*bytes))
         .collect();
      assert_eq!(values, [
         1.0_f32, 40.0, 1.0, 30.0, 2.0, 60.0, 3.0, 20.0, 4.0, 30.0, 3.0, 70.0, 6.0, 40.0, 7.0,
         40.0,
      ]);
   }
}
