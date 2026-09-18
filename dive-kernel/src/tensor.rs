use alloc::{
   vec,
   vec::Vec,
};

use dive_abi::{
   dive,
   status::Status,
};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
   Bool    = 0,
   Int8    = 1,
   UInt8   = 2,
   Int16   = 3,
   Int32   = 4,
   Float16 = 5,
   Bfloat  = 6,
   Float32 = 7,
   UInt16  = 8,
   UInt32  = 9,
   Int64   = 10,
   UInt64  = 11,
   Float64 = 12,
}

impl TensorType {
   #[inline]
   #[must_use]
   pub const fn from_raw(raw: u32) -> Option<Self> {
      match raw {
         0 => Some(Self::Bool),
         1 => Some(Self::Int8),
         2 => Some(Self::UInt8),
         3 => Some(Self::Int16),
         4 => Some(Self::Int32),
         5 => Some(Self::Float16),
         6 => Some(Self::Bfloat),
         7 => Some(Self::Float32),
         8 => Some(Self::UInt16),
         9 => Some(Self::UInt32),
         10 => Some(Self::Int64),
         11 => Some(Self::UInt64),
         12 => Some(Self::Float64),
         _ => None,
      }
   }
}

#[inline]
#[must_use]
pub const fn type_size_bytes(ty: TensorType) -> usize {
   match ty {
      TensorType::Bool | TensorType::Int8 | TensorType::UInt8 => 1,
      TensorType::Int16 | TensorType::Float16 | TensorType::Bfloat | TensorType::UInt16 => 2,
      TensorType::Int32 | TensorType::Float32 | TensorType::UInt32 => 4,
      TensorType::Int64 | TensorType::UInt64 | TensorType::Float64 => 8,
   }
}

#[inline]
#[must_use]
pub fn type_size_bytes_raw(raw: u32) -> usize {
   TensorType::from_raw(raw).map_or_else(
      || {
         dive::log("Unexpected vmir::TensorType.", "tensor_util.cc");
         dive::cease(54, "tensor_util.cc")
      },
      type_size_bytes,
   )
}

#[inline]
#[must_use]
pub const fn is_integer_type(ty: TensorType) -> bool {
   match ty {
      TensorType::Int8
      | TensorType::UInt8
      | TensorType::Int16
      | TensorType::Int32
      | TensorType::UInt16
      | TensorType::UInt32
      | TensorType::Int64
      | TensorType::UInt64 => true,
      TensorType::Bool
      | TensorType::Float16
      | TensorType::Bfloat
      | TensorType::Float32
      | TensorType::Float64 => false,
   }
}

#[inline]
#[must_use]
pub const fn is_signed_integer_type(ty: TensorType) -> bool {
   match ty {
      TensorType::Int8 | TensorType::Int16 | TensorType::Int32 | TensorType::Int64 => true,
      TensorType::Bool
      | TensorType::UInt8
      | TensorType::UInt16
      | TensorType::UInt32
      | TensorType::UInt64
      | TensorType::Float16
      | TensorType::Bfloat
      | TensorType::Float32
      | TensorType::Float64 => false,
   }
}

/// Clamps a dim to a non-negative element count.
#[inline]
#[must_use]
pub fn extent(dim: i64) -> usize {
   usize::try_from(dim).unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
   pub dims: Vec<i64>,
}

impl Shape {
   #[inline]
   #[must_use]
   pub const fn new(dims: Vec<i64>) -> Self {
      Self { dims }
   }

   #[inline]
   #[must_use]
   pub const fn rank(&self) -> usize {
      self.dims.len()
   }

   #[inline]
   #[must_use]
   pub fn num_elements(&self) -> usize {
      self.dims.iter().fold(1_usize, |acc, dim| {
         acc.saturating_mul(usize::try_from(*dim).unwrap_or(0))
      })
   }
}

#[expect(
   clippy::partial_pub_fields,
   reason = "the byte buffer stays private so its length stays the single backing size"
)]
pub struct Tensor {
   pub tensor_type: TensorType,
   pub shape:       Shape,
   /// Owned element bytes in little endian order.
   backing:         Option<Vec<u8>>,
}

impl Tensor {
   #[inline]
   #[must_use]
   pub const fn new(tensor_type: TensorType, shape: Shape) -> Self {
      Self {
         tensor_type,
         shape,
         backing: None,
      }
   }

   #[inline]
   pub fn set_backing_buffer(&mut self, bytes: Vec<u8>) {
      if self.backing.is_some() {
         dive::cease(20, "tensor.cc");
      }
      self.backing = Some(bytes);
   }

   #[inline]
   #[must_use]
   pub const fn has_backing(&self) -> bool {
      self.backing.is_some()
   }

   #[inline]
   #[must_use]
   pub fn bytes(&self) -> &[u8] {
      self.backing.as_deref().unwrap_or_default()
   }

   #[inline]
   pub fn bytes_mut(&mut self) -> &mut [u8] {
      self.backing.as_deref_mut().unwrap_or_default()
   }
}

#[expect(
   clippy::partial_pub_fields,
   reason = "the byte buffer stays private so its length stays the single backing size"
)]
pub struct ResolvedTensor {
   pub tensor_type:   TensorType,
   pub shape:         Shape,
   pub element_count: usize,
   /// Owned element bytes in little endian order.
   backing:           Vec<u8>,
}

impl ResolvedTensor {
   #[inline]
   #[must_use]
   pub fn new(tensor_type: TensorType, shape: Shape) -> Self {
      let element_count = shape.num_elements();
      Self {
         tensor_type,
         shape,
         element_count,
         backing: Vec::new(),
      }
   }

   #[inline]
   #[must_use]
   pub const fn byte_size(&self) -> usize {
      self.element_count * type_size_bytes(self.tensor_type)
   }

   #[inline]
   #[must_use]
   pub const fn backing_size(&self) -> usize {
      self.backing.len()
   }

   #[inline]
   #[must_use]
   pub const fn has_backing(&self) -> bool {
      !self.backing.is_empty()
   }

   #[inline]
   pub fn set_backing_buffer(&mut self, bytes: Vec<u8>) {
      if !self.backing.is_empty() {
         dive::cease(20, "tensor.cc");
      }
      self.backing = bytes;
   }

   #[inline]
   pub fn allocate_backing_buffer(&mut self) -> Status {
      if !self.backing.is_empty() {
         dive::cease(20, "tensor.cc");
      }
      self.backing = vec![0_u8; self.byte_size()];
      Status::ok()
   }

   #[inline]
   #[must_use]
   pub fn bytes(&self) -> &[u8] {
      &self.backing
   }

   #[inline]
   pub fn bytes_mut(&mut self) -> &mut [u8] {
      &mut self.backing
   }

   /// Ceases when the backing holds fewer than `need` bytes.
   #[inline]
   pub fn require_bytes(&self, need: usize, file: &'static str) {
      if self.backing.len() < need {
         dive::log("backing buffer too small", file);
         dive::cease(20, file);
      }
   }
}

pub struct UnresolvedTensor {
   pub tensor_type:    TensorType,
   pub expected:       Shape,
   pub resolved_shape: Option<Shape>,
}

impl UnresolvedTensor {
   #[inline]
   #[must_use]
   pub const fn new(tensor_type: TensorType, expected: Shape) -> Self {
      Self {
         tensor_type,
         expected,
         resolved_shape: None,
      }
   }

   #[inline]
   #[must_use]
   pub const fn is_resolved(&self) -> bool {
      self.resolved_shape.is_some()
   }

   #[inline]
   pub fn resolve(&mut self, shape: Shape) -> ResolvedTensor {
      if let Some(done) = self.resolved_shape.as_ref() {
         if *done != shape {
            dive::cease(406, "tensor.h");
         }
         return ResolvedTensor::new(self.tensor_type, shape);
      }
      if self.expected.rank() != shape.rank() {
         dive::cease(406, "tensor.h");
      }
      for (want, got) in self.expected.dims.iter().zip(shape.dims.iter()) {
         if *got <= 0 || (*want != -1 && *want != *got) {
            dive::cease(406, "tensor.h");
         }
      }
      self.resolved_shape = Some(shape.clone());
      ResolvedTensor::new(self.tensor_type, shape)
   }
}
