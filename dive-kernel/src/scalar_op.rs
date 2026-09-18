//! The `DiveVm_LegacyScalarOp` interpreter.

use core::f32::math::{
   round_ties_even,
   trunc,
};

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

/// One operand of `DiveVm_LegacyScalarOp`.
///
/// The C++ passes eight parallel arrays rather than a struct. This groups one
/// index across all of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarOperand<'buf> {
   /// Element bytes, from `unsigned char* const*`.
   pub bytes:      &'buf [u8],
   /// Byte size, from `const long*`.
   pub size:       i64,
   /// `DiveVmPrimitiveType` tag.
   pub primitive:  i32,
   /// Element width in bits.
   pub width_bits: i32,
   /// Whether integer storage is signed.
   pub signed:     bool,
   /// Quantisation scale, from the first `const float*`.
   pub scale:      f32,
   /// Quantisation zero point, from the second `const float*`.
   pub zero_point: i32,
}

impl ScalarOperand<'_> {
   /// The descriptor equality the fast path and the duplicate check both test.
   /// It compares every field except the buffer and the size.
   #[inline]
   #[must_use]
   pub const fn descriptor_eq(&self, other: &Self) -> bool {
      self.primitive == other.primitive
         && self.width_bits == other.width_bits
         && self.signed == other.signed
         && self.scale.to_bits() == other.scale.to_bits()
         && self.zero_point == other.zero_point
   }
}

/// The op value whose matching-descriptor case degenerates to a byte copy.
pub const SCALAR_OP_COPY: i32 = 3;

/// Scalar value after descriptor decoding and dequantization.
#[derive(Clone, Copy)]
enum ScalarValue {
   /// Signed or unsigned storage widened into one integer lane.
   Integer(i64),
   /// Float storage or a dequantized integer lane.
   Float(f32),
}

impl ScalarValue {
   /// Converts either lane into the floating point operation domain.
   const fn as_float(self) -> f32 {
      match self {
         Self::Integer(value) => value as f32,
         Self::Float(value) => value,
      }
   }

   /// Applies the scalar operation truth test.
   const fn truthy(self) -> bool {
      match self {
         Self::Integer(value) => value != 0_i64,
         Self::Float(value) => value != 0.0_f32,
      }
   }
}

/// Returns the storage width supported by the legacy scalar kernel.
const fn scalar_width(operand: &ScalarOperand<'_>) -> Option<usize> {
   match (operand.primitive, operand.width_bits) {
      (0_i32, 8_i32) => Some(1_usize),
      (0_i32, 16_i32) => Some(2_usize),
      (0_i32 | 1_i32, 32_i32) => Some(4_usize),
      _ => None,
   }
}

/// Derives an element count only when the byte size is width aligned.
fn scalar_element_count(operand: &ScalarOperand<'_>) -> Option<usize> {
   let size = usize::try_from(operand.size).ok()?;
   let width = scalar_width(operand)?;
   size.is_multiple_of(width).then_some(size / width)
}

/// Checks broadcast shape and backing bytes for one scalar input.
fn scalar_input_fits(operand: &ScalarOperand<'_>, count: usize, output_count: usize) -> bool {
   (count == 1 || count == output_count)
      && scalar_width(operand).is_some_and(|width| operand.bytes.len() >= count * width)
}

/// Loads and dequantizes one validated legacy scalar operand.
fn load_scalar(operand: &ScalarOperand<'_>, index: usize) -> Option<ScalarValue> {
   let width = scalar_width(operand)?;
   let bytes = &operand.bytes[index * width..][..width];

   if operand.primitive == 1_i32 {
      return Some(ScalarValue::Float(f32::from_le_bytes(dive::lane::<4>(
         bytes,
      ))));
   }

   let raw = match (operand.width_bits, operand.signed) {
      (8_i32, true) => i64::from(bytes[0].cast_signed()),
      (8_i32, false) => i64::from(bytes[0]),
      (16_i32, true) => i64::from(i16::from_le_bytes(dive::lane::<2>(bytes))),
      (16_i32, false) => i64::from(u16::from_le_bytes(dive::lane::<2>(bytes))),
      (32_i32, true) => i64::from(i32::from_le_bytes(dive::lane::<4>(bytes))),
      (32_i32, false) => i64::from(u32::from_le_bytes(dive::lane::<4>(bytes))),
      _ => return None,
   };

   Some(if operand.scale == 1.0_f32 && operand.zero_point == 0_i32 {
      ScalarValue::Integer(raw)
   } else {
      ScalarValue::Float((raw - i64::from(operand.zero_point)) as f32 * operand.scale)
   })
}

/// Evaluates an operation that remains in the integer domain.
fn integer_result(op: i32, left_value: i64, right_value: i64) -> Result<ScalarValue, Code> {
   let left_word = left_value as i32;
   let right_word = right_value as i32;
   let value = match op {
      4_i32 => i64::from(left_word.wrapping_add(right_word)),
      5_i32 => i64::from(left_word.wrapping_mul(right_word)),
      6_i32 if right_value == 0_i64 => return Err(Code::InvalidArgument),
      6_i32 => i64::from(trunc(left_value as f32 / right_value as f32) as i32),
      7_i32 => i64::from(left_word.wrapping_sub(right_word)),
      8_i32 => left_value & right_value,
      9_i32 => left_value | right_value,
      10_i32 => left_value ^ right_value,
      13_i32 => left_value.min(right_value),
      14_i32 => left_value.max(right_value),
      _ => return Err(Code::Unimplemented),
   };
   Ok(ScalarValue::Integer(value))
}

/// Evaluates one decoded scalar operation.
fn evaluate_scalar(
   op: i32,
   left_value: ScalarValue,
   right_value: ScalarValue,
) -> Result<ScalarValue, Code> {
   let either_float =
      matches!(left_value, ScalarValue::Float(_)) || matches!(right_value, ScalarValue::Float(_));

   match op {
      0_i32 => {
         Ok(ScalarValue::Integer(i64::from(if either_float {
            left_value.as_float() != right_value.as_float()
         } else {
            let (ScalarValue::Integer(left_integer), ScalarValue::Integer(right_integer)) =
               (left_value, right_value)
            else {
               return Err(Code::Internal);
            };
            left_integer != right_integer
         })))
      },
      1_i32 => {
         Ok(ScalarValue::Integer(i64::from(if either_float {
            left_value.as_float() == right_value.as_float()
         } else {
            let (ScalarValue::Integer(left_integer), ScalarValue::Integer(right_integer)) =
               (left_value, right_value)
            else {
               return Err(Code::Internal);
            };
            left_integer == right_integer
         })))
      },
      2_i32 => {
         Ok(ScalarValue::Integer(i64::from(if either_float {
            left_value.as_float() < right_value.as_float()
         } else {
            let (ScalarValue::Integer(left_integer), ScalarValue::Integer(right_integer)) =
               (left_value, right_value)
            else {
               return Err(Code::Internal);
            };
            left_integer < right_integer
         })))
      },
      4_i32 | 5_i32 | 6_i32 | 7_i32 | 13_i32 | 14_i32 if either_float => {
         let left_float = left_value.as_float();
         let right_float = right_value.as_float();
         let value = match op {
            4_i32 => left_float + right_float,
            5_i32 => left_float * right_float,
            6_i32 => left_float / right_float,
            7_i32 => left_float - right_float,
            13_i32 => left_float.min(right_float),
            14_i32 => left_float.max(right_float),
            _ => return Err(Code::Internal),
         };
         Ok(ScalarValue::Float(value))
      },
      4_i32..=10_i32 | 13_i32 | 14_i32 => {
         let (ScalarValue::Integer(left_integer), ScalarValue::Integer(right_integer)) =
            (left_value, right_value)
         else {
            return Err(Code::Unimplemented);
         };
         integer_result(op, left_integer, right_integer)
      },
      11_i32 => {
         Ok(ScalarValue::Integer(i64::from(
            left_value.truthy() && right_value.truthy(),
         )))
      },
      12_i32 => {
         Ok(ScalarValue::Integer(i64::from(
            left_value.truthy() || right_value.truthy(),
         )))
      },
      _ => Err(Code::Unimplemented),
   }
}

/// Quantizes and stores one legacy scalar result.
fn store_scalar(
   operand: &ScalarOperand<'_>,
   bytes: &mut [u8],
   scalar_value: ScalarValue,
) -> Option<()> {
   if operand.primitive == 1_i32 {
      bytes.copy_from_slice(&scalar_value.as_float().to_le_bytes());
      return Some(());
   }

   let raw = if operand.scale == 1.0_f32 && operand.zero_point == 0_i32 {
      match scalar_value {
         ScalarValue::Integer(integer_value) => integer_value,
         ScalarValue::Float(float_value) => round_ties_even(float_value) as i64,
      }
   } else {
      let scaled = round_ties_even(scalar_value.as_float() / operand.scale);
      (scaled as i64).saturating_add(i64::from(operand.zero_point))
   };

   match operand.width_bits {
      8_i32 => bytes[0] = raw as u8,
      16_i32 => bytes.copy_from_slice(&(raw as u16).to_le_bytes()),
      32_i32 => bytes.copy_from_slice(&(raw as u32).to_le_bytes()),
      _ => return None,
   }
   Some(())
}

/// Writes a scalar immediate into every output element.
fn fill_scalar_immediate(
   output: &ScalarOperand<'_>,
   dst: &mut [u8],
   output_count: usize,
   output_width: usize,
   immediate_value: i32,
) -> Option<()> {
   for index in 0..output_count {
      let at = index * output_width;
      store_scalar(
         output,
         &mut dst[at..at + output_width],
         ScalarValue::Integer(i64::from(immediate_value)),
      )?;
   }
   Some(())
}

#[inline]
pub fn legacy_scalar_op(
   op: i32,
   operands: &[ScalarOperand<'_>],
   dst: &mut [u8],
   immediate: Option<i32>,
) -> Status {
   const FILE: &str = "legacy_scalar_op.cc";

   if operands.is_empty() || operands.len() > 3 {
      dive::log("Unsupported number of operands", FILE);
      return Status::new(Code::InvalidArgument);
   }
   let (Some(first), Some(last)) = (operands.first(), operands.last()) else {
      dive::log("Unsupported number of operands", FILE);
      return Status::new(Code::InvalidArgument);
   };

   if !(0_i32..=14_i32).contains(&op) {
      dive::log("unsupported legacy scalar operation", FILE);
      return Status::new(Code::Unimplemented);
   }
   if operands
      .iter()
      .any(|operand| scalar_width(operand).is_none())
   {
      dive::log("unsupported legacy scalar operand type", FILE);
      return Status::new(Code::Unimplemented);
   }
   if operands.len() == 3 && !first.descriptor_eq(&operands[1]) {
      dive::log("operand descriptor mismatch", FILE);
      return Status::new(Code::InvalidArgument);
   }

   let Some(output_count) = scalar_element_count(last) else {
      dive::log("invalid legacy scalar output size", FILE);
      return Status::new(Code::InvalidArgument);
   };
   let output_width = scalar_width(last).unwrap_or(1);
   let Some(output_size) = output_count.checked_mul(output_width) else {
      return Status::new(Code::InvalidArgument);
   };
   if dst.len() < output_size || (last.primitive == 0_i32 && last.scale == 0.0_f32) {
      dive::log("invalid legacy scalar output", FILE);
      return Status::new(Code::InvalidArgument);
   }

   if op == SCALAR_OP_COPY
      && immediate.is_none()
      && operands.len() == 2
      && first.descriptor_eq(last)
      && first.bytes.len() >= output_size
   {
      dst[..output_size].copy_from_slice(&first.bytes[..output_size]);
      return Status::ok();
   }

   if op == SCALAR_OP_COPY {
      let Some(immediate_value) = immediate else {
         return Status::new(Code::InvalidArgument);
      };
      if fill_scalar_immediate(last, dst, output_count, output_width, immediate_value).is_none() {
         return Status::new(Code::Unimplemented);
      }
      return Status::ok();
   }

   if operands.len() < 2 {
      return Status::new(Code::InvalidArgument);
   }
   let Some(left_count) = scalar_element_count(first) else {
      return Status::new(Code::InvalidArgument);
   };
   if !scalar_input_fits(first, left_count, output_count) {
      return Status::new(Code::InvalidArgument);
   }

   let right_operand = (immediate.is_none() && operands.len() == 3).then(|| &operands[1]);
   let right_count = right_operand.and_then(scalar_element_count);
   if immediate.is_none() && right_count.is_none() {
      return Status::new(Code::InvalidArgument);
   }
   if right_operand
      .zip(right_count)
      .is_some_and(|(operand, count)| !scalar_input_fits(operand, count, output_count))
   {
      return Status::new(Code::InvalidArgument);
   }

   for index in 0..output_count {
      let left_index = if left_count == 1 { 0 } else { index };
      let Some(left_value) = load_scalar(first, left_index) else {
         return Status::new(Code::Unimplemented);
      };
      let right_value = match (immediate, right_operand, right_count) {
         (Some(immediate_value), ..) => ScalarValue::Integer(i64::from(immediate_value)),
         (None, Some(operand), Some(count)) => {
            let Some(value) = load_scalar(operand, if count == 1 { 0 } else { index }) else {
               return Status::new(Code::Unimplemented);
            };
            value
         },
         _ => return Status::new(Code::InvalidArgument),
      };
      let value = match evaluate_scalar(op, left_value, right_value) {
         Ok(value) => value,
         Err(code) => return Status::new(code),
      };
      let at = index * output_width;
      if store_scalar(last, &mut dst[at..at + output_width], value).is_none() {
         return Status::new(Code::Unimplemented);
      }
   }

   Status::ok()
}
