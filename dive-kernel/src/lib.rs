#![no_std]
#![feature(core_float_math)]

extern crate alloc;

pub mod accessor;
pub mod bfloat;
pub mod cast_ops;
pub mod gather;
pub mod integral_pooling;
pub mod mask_indices;
pub mod multinomial;
pub mod pad_fill;
pub mod rotate_crop_resize;
pub mod scalar_op;
pub mod slice_mem;
pub mod tensor;
pub mod tensor_ops;
pub mod topk_reduce;
pub mod warp_affine;

#[cfg(test)]
mod test_support {
   use alloc::vec::Vec;

   use crate::tensor::{
      ResolvedTensor,
      Shape,
      TensorType,
   };

   pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
      values
         .iter()
         .flat_map(|value| value.to_le_bytes())
         .collect()
   }

   pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
      values
         .iter()
         .flat_map(|value| value.to_le_bytes())
         .collect()
   }

   pub fn read_f32(tensor: &ResolvedTensor) -> Vec<f32> {
      tensor
         .bytes()
         .as_chunks::<4>()
         .0
         .iter()
         .map(|chunk| f32::from_le_bytes(*chunk))
         .collect()
   }

   pub fn read_i32(tensor: &ResolvedTensor) -> Vec<i32> {
      tensor
         .bytes()
         .as_chunks::<4>()
         .0
         .iter()
         .map(|chunk| i32::from_le_bytes(*chunk))
         .collect()
   }

   pub fn tensor(tensor_type: TensorType, dims: Vec<i64>, bytes: &[u8]) -> ResolvedTensor {
      let mut tensor = ResolvedTensor::new(tensor_type, Shape::new(dims));
      tensor.set_backing_buffer(bytes.to_vec());
      tensor
   }

   pub fn zero_tensor(tensor_type: TensorType, dims: Vec<i64>) -> ResolvedTensor {
      let mut tensor = ResolvedTensor::new(tensor_type, Shape::new(dims));
      assert!(
         tensor.allocate_backing_buffer().is_ok(),
         "tensor allocation should succeed"
      );
      tensor
   }
}
