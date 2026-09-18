//! Typed element access over tensor backing bytes, with affine quantisation.

use dive_abi::dive;

use crate::{
   bfloat::{
      bf16_to_f32,
      f32_to_bf16,
   },
   tensor::{
      TensorType,
      is_integer_type,
      type_size_bytes,
   },
};

/// Affine quantisation carried in `ElementType` beside the storage dtype.
///
/// The C++ struct sits at `TensorRef + 104` and lays out the primitive kind at
/// +0, the bit width at +4, the signedness at +8, `scale` at +12, and
/// `zero_point` at +16. `TensorAccessor::TensorAccessor` installs a separate
/// closure set when `scale == 1.0`, which is the unquantised fast path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantization {
   pub scale:      f32,
   pub zero_point: i32,
}

/// Matches the compiler runtime conversion used by the original kernel.
fn fix_double_to_i64(value: f64) -> i64 {
   if value.is_nan() {
      return if value.is_sign_negative() { 0 } else { -1 };
   }
   if value < 0.0_f64 {
      return ((-value) as u64).wrapping_neg() as i64;
   }
   (value as u64) as i64
}

impl Quantization {
   /// The identity the `scale == 1.0` closure set implements.
   pub const IDENTITY: Self = Self {
      scale:      1.0,
      zero_point: 0,
   };

   #[inline]
   #[must_use]
   pub const fn new(scale: f32, zero_point: i32) -> Self {
      Self { scale, zero_point }
   }

   /// The constructor branches on an exact `scale == 1.0`, so this comparison
   /// has to be exact too.
   #[inline]
   #[must_use]
   pub fn is_identity(self) -> bool {
      self.scale == 1.0 && self.zero_point == 0
   }

   /// `lw zp, 0x10(elem)` then `sub`, widened and scaled by `flw 0xc(elem)`.
   #[inline]
   #[must_use]
   pub fn dequantize(self, stored: i64) -> f64 {
      stored.wrapping_sub(i64::from(self.zero_point)) as f64 * f64::from(self.scale)
   }

   /// The inverse, ending in `add zp` before the narrowing store.
   #[inline]
   #[must_use]
   pub fn quantize(self, value: f64) -> i64 {
      let scaled = value / f64::from(self.scale);
      fix_double_to_i64(scaled).wrapping_add(i64::from(self.zero_point))
   }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementType {
   Int8(Quantization),
   UInt8(Quantization),
   Int16(Quantization),
   UInt16(Quantization),
   Int32(Quantization),
   UInt32(Quantization),
   Int64(Quantization),
   UInt64(Quantization),
   Bfloat,
   Float32,
   Float64,
}

impl ElementType {
   #[inline]
   #[must_use]
   pub const fn dtype(self) -> TensorType {
      match self {
         Self::Int8(_) => TensorType::Int8,
         Self::UInt8(_) => TensorType::UInt8,
         Self::Int16(_) => TensorType::Int16,
         Self::UInt16(_) => TensorType::UInt16,
         Self::Int32(_) => TensorType::Int32,
         Self::UInt32(_) => TensorType::UInt32,
         Self::Int64(_) => TensorType::Int64,
         Self::UInt64(_) => TensorType::UInt64,
         Self::Bfloat => TensorType::Bfloat,
         Self::Float32 => TensorType::Float32,
         Self::Float64 => TensorType::Float64,
      }
   }

   #[inline]
   #[must_use]
   pub const fn quantization(self) -> Quantization {
      match self {
         Self::Int8(quant)
         | Self::UInt8(quant)
         | Self::Int16(quant)
         | Self::UInt16(quant)
         | Self::Int32(quant)
         | Self::UInt32(quant)
         | Self::Int64(quant)
         | Self::UInt64(quant) => quant,
         Self::Bfloat | Self::Float32 | Self::Float64 => Quantization::IDENTITY,
      }
   }
}

pub struct TensorAccessor<Buffer> {
   /// Storage and quantization descriptor for each element.
   element_type: ElementType,
   /// Borrowed or owned backing bytes.
   buffer:       Buffer,
}

impl<Buffer> TensorAccessor<Buffer> {
   #[inline]
   #[must_use]
   pub const fn new(element_type: ElementType, buffer: Buffer) -> Self {
      Self {
         element_type,
         buffer,
      }
   }

   #[inline]
   #[must_use]
   pub const fn dtype(&self) -> TensorType {
      self.element_type.dtype()
   }

   #[inline]
   #[must_use]
   pub const fn quantization(&self) -> Quantization {
      self.element_type.quantization()
   }
}

impl<Buffer: AsRef<[u8]>> TensorAccessor<Buffer> {
   #[inline]
   #[must_use]
   pub fn as_bytes(&self) -> &[u8] {
      self.buffer.as_ref()
   }

   #[inline]
   #[must_use]
   pub fn len(&self) -> usize {
      self.as_bytes().len() / type_size_bytes(self.dtype())
   }

   #[inline]
   #[must_use]
   pub fn is_empty(&self) -> bool {
      self.len() == 0
   }

   /// Ceases simulation when the element index is out of bounds.
   fn check(&self, index: usize) {
      if index >= self.len() {
         dive::cease(11, "tensor_accessor.h");
      }
   }

   #[inline]
   #[must_use]
   /// Reads one element at its storage dtype with no affine transform.
   fn load_raw_f64(&self, index: usize) -> f64 {
      self.check(index);
      let dtype = self.dtype();
      let size = type_size_bytes(dtype);
      let base = index * size;
      let bytes = &self.as_bytes()[base..base + size];
      match self.element_type {
         ElementType::UInt8(_) => f64::from(bytes[0]),
         ElementType::Int8(_) => f64::from(bytes[0].cast_signed()),
         ElementType::UInt16(_) => f64::from(u16::from_le_bytes(dive::lane::<2>(bytes))),
         ElementType::Int16(_) => f64::from(i16::from_le_bytes(dive::lane::<2>(bytes))),
         ElementType::Bfloat => f64::from(bf16_to_f32(u16::from_le_bytes(dive::lane::<2>(bytes)))),
         ElementType::UInt32(_) => f64::from(u32::from_le_bytes(dive::lane::<4>(bytes))),
         ElementType::Int32(_) => f64::from(i32::from_le_bytes(dive::lane::<4>(bytes))),
         ElementType::Float32 => f64::from(f32::from_le_bytes(dive::lane::<4>(bytes))),
         ElementType::UInt64(_) => {
            let raw = u64::from_le_bytes(dive::lane::<8>(bytes));
            raw as i64 as f64
         },
         ElementType::Int64(_) => i64::from_le_bytes(dive::lane::<8>(bytes)) as f64,
         ElementType::Float64 => f64::from_le_bytes(dive::lane::<8>(bytes)),
      }
   }

   #[inline]
   /// Writes one element at its storage dtype with no affine transform.
   fn store_raw_f64(&mut self, index: usize, value: f64)
   where
      Buffer: AsMut<[u8]>,
   {
      self.check(index);
      let dtype = self.dtype();
      let size = type_size_bytes(dtype);
      let base = index * size;
      let bytes = &mut self.buffer.as_mut()[base..base + size];
      match self.element_type {
         ElementType::UInt8(_) | ElementType::Int8(_) => {
            bytes[0] = fix_double_to_i64(value) as u8;
         },
         ElementType::UInt16(_) => {
            let bits = (fix_double_to_i64(value) as u16).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int16(_) => {
            let bits = (fix_double_to_i64(value) as i16).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Bfloat => {
            let bits = f32_to_bf16(value as f32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::UInt32(_) => {
            let bits = (fix_double_to_i64(value) as u32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int32(_) => {
            let bits = (fix_double_to_i64(value) as i32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Float32 => {
            let bits = (value as f32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::UInt64(_) => {
            let bits = (fix_double_to_i64(value) as u64).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int64(_) => {
            let bits = fix_double_to_i64(value).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Float64 => {
            let bits = value.to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
      }
   }

   /// Only the integer closure sets carry `scale` and `zero_point`, so float
   /// storage passes straight through.
   #[inline]
   #[must_use]
   pub fn load_f64(&self, index: usize) -> f64 {
      let quant = self.quantization();
      if quant.is_identity() || !is_integer_type(self.dtype()) {
         return self.load_raw_f64(index);
      }
      quant.dequantize(self.load_i64(index))
   }

   #[inline]
   pub fn store_f64(&mut self, index: usize, value: f64)
   where
      Buffer: AsMut<[u8]>,
   {
      let quant = self.quantization();
      if quant.is_identity() || !is_integer_type(self.dtype()) {
         self.store_raw_f64(index, value);
         return;
      }
      let stored = quant.quantize(value);
      self.store_i64(index, stored);
   }

   #[inline]
   #[must_use]
   pub fn load_i64(&self, index: usize) -> i64 {
      self.check(index);
      let dtype = self.dtype();
      let size = type_size_bytes(dtype);
      let base = index * size;
      let bytes = &self.as_bytes()[base..base + size];
      match self.element_type {
         ElementType::UInt8(_) => i64::from(bytes[0]),
         ElementType::Int8(_) => i64::from(bytes[0].cast_signed()),
         ElementType::UInt16(_) => i64::from(u16::from_le_bytes(dive::lane::<2>(bytes))),
         ElementType::Int16(_) => i64::from(i16::from_le_bytes(dive::lane::<2>(bytes))),
         ElementType::Bfloat => {
            fix_double_to_i64(f64::from(bf16_to_f32(u16::from_le_bytes(dive::lane::<2>(
               bytes,
            )))))
         },
         ElementType::UInt32(_) => i64::from(u32::from_le_bytes(dive::lane::<4>(bytes))),
         ElementType::Int32(_) => i64::from(i32::from_le_bytes(dive::lane::<4>(bytes))),
         ElementType::Float32 => {
            fix_double_to_i64(f64::from(f32::from_le_bytes(dive::lane::<4>(bytes))))
         },
         ElementType::UInt64(_) => {
            let raw = u64::from_le_bytes(dive::lane::<8>(bytes));
            raw as i64
         },
         ElementType::Int64(_) => i64::from_le_bytes(dive::lane::<8>(bytes)),
         ElementType::Float64 => fix_double_to_i64(f64::from_le_bytes(dive::lane::<8>(bytes))),
      }
   }

   #[inline]
   pub fn store_i64(&mut self, index: usize, value: i64)
   where
      Buffer: AsMut<[u8]>,
   {
      self.check(index);
      let dtype = self.dtype();
      let size = type_size_bytes(dtype);
      let base = index * size;
      let bytes = &mut self.buffer.as_mut()[base..base + size];
      match self.element_type {
         ElementType::UInt8(_) | ElementType::Int8(_) => {
            bytes[0] = value as u8;
         },
         ElementType::UInt16(_) => {
            let bits = (value as u16).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int16(_) => {
            let bits = (value as i16).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Bfloat => {
            let bits = f32_to_bf16(value as f32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::UInt32(_) => {
            let bits = (value as u32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int32(_) => {
            let bits = (value as i32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Float32 => {
            let bits = (value as f32).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::UInt64(_) => {
            let bits = (value as u64).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Int64(_) => {
            let bits = value.to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
         ElementType::Float64 => {
            let bits = (value as f64).to_le_bytes();
            bytes.copy_from_slice(&bits);
         },
      }
   }

   #[inline]
   pub fn loader(&self) -> impl Fn(usize) -> f64 + '_ {
      move |index| self.load_f64(index)
   }

   #[inline]
   pub fn storer(&mut self) -> impl FnMut(usize, f64) + '_
   where
      Buffer: AsMut<[u8]>,
   {
      move |index, value| self.store_f64(index, value)
   }
}

#[cfg(test)]
mod tests {
   use crate::accessor::{
      ElementType,
      Quantization,
      TensorAccessor,
   };

   /// `ElementType` carries `scale` at +12 and `zero_point` at +16. The int
   /// closure sets load `(stored - zero_point) * scale` and store the inverse,
   /// while `scale == 1.0` selects a separate unquantised closure set.
   #[test]
   fn quantized_accessor_round_trips_through_scale_and_zero_point() {
      let quantization = Quantization::new(0.5, -7);
      let mut bytes = [0_u8; 4];
      let mut accessor = TensorAccessor::new(ElementType::Int8(quantization), &mut bytes);

      accessor.store_f64(0, 3.0_f64);
      assert_eq!(accessor.load_i64(0), -1, "3.0 / 0.5 + -7 == -1");
      assert_eq!(accessor.load_f64(0), 3.0_f64);

      accessor.store_f64(1, -2.0_f64);
      assert_eq!(accessor.load_i64(1), -11);
      assert_eq!(accessor.load_f64(1), -2.0_f64);

      accessor.store_f64(2, 3.3_f64);
      assert_eq!(accessor.load_i64(2), -1);
      assert_eq!(accessor.load_f64(2), 3.0_f64);
   }

   #[test]
   fn identity_quantization_leaves_the_raw_path() {
      let mut bytes = [0_u8; 4];
      let mut accessor = TensorAccessor::new(ElementType::Int8(Quantization::IDENTITY), &mut bytes);
      assert!(
         accessor.quantization().is_identity(),
         "identity quantization should select the raw path"
      );
      accessor.store_f64(0, 5.0_f64);
      assert_eq!(accessor.load_i64(0), 5);
      assert_eq!(accessor.load_f64(0), 5.0_f64);

      accessor.store_i64(1, 300);
      assert_eq!(accessor.load_i64(1), 44);

      accessor.store_f64(2, f64::NEG_INFINITY);
      assert_eq!(accessor.load_i64(2), 1);
   }
}
