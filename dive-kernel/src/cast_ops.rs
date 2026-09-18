use core::hint::spin_loop;

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

/// Reads one element as the f64 staging value the C++ cast path uses.
fn read_as_f64(ty: TensorType, cell: &[u8]) -> Option<f64> {
   Some(match ty {
      TensorType::Bool => f64::from(u8::from(cell[0] != 0)),
      TensorType::Int8 => f64::from(cell[0].cast_signed()),
      TensorType::UInt8 => f64::from(cell[0]),
      TensorType::Int16 => f64::from(i16::from_le_bytes(dive::lane::<2>(cell))),
      TensorType::UInt16 => f64::from(u16::from_le_bytes(dive::lane::<2>(cell))),
      TensorType::Int32 => f64::from(i32::from_le_bytes(dive::lane::<4>(cell))),
      TensorType::UInt32 => f64::from(u32::from_le_bytes(dive::lane::<4>(cell))),
      TensorType::Int64 => i64::from_le_bytes(dive::lane::<8>(cell)) as f64,
      TensorType::UInt64 => u64::from_le_bytes(dive::lane::<8>(cell)) as f64,
      TensorType::Float32 => f64::from(f32::from_le_bytes(dive::lane::<4>(cell))),
      TensorType::Float64 => f64::from_le_bytes(dive::lane::<8>(cell)),
      TensorType::Float16 | TensorType::Bfloat => return None,
   })
}

/// RISC-V `fcvt.l.s rtz` maps NaN to [`i64::MAX`] and saturates the finite
/// range, then the narrow store keeps only the low bits with no clamp to the
/// dtype.
const fn f64_to_i64_rtz_sat(value: f64) -> i64 {
   if value.is_nan() {
      i64::MAX
   } else {
      value as i64
   }
}

/// Writes the f64 staging value back at the destination dtype width.
fn write_from_f64(ty: TensorType, cell: &mut [u8], value: f64) -> Option<()> {
   let narrow = f64_to_i64_rtz_sat(value);
   let mut out = [0_u8; 8];
   let width = match ty {
      TensorType::Bool => {
         out[0] = u8::from(value != 0.0_f64 && !value.is_nan());
         1
      },
      TensorType::Int8 => {
         out[0] = (narrow as i8).cast_unsigned();
         1
      },
      TensorType::UInt8 => {
         out[0] = narrow as u8;
         1
      },
      TensorType::Int16 => dive::stage(&mut out, &(narrow as i16).to_le_bytes()),
      TensorType::UInt16 => dive::stage(&mut out, &(narrow as u16).to_le_bytes()),
      TensorType::Int32 => dive::stage(&mut out, &(narrow as i32).to_le_bytes()),
      TensorType::UInt32 => dive::stage(&mut out, &(narrow as u32).to_le_bytes()),
      TensorType::Int64 => dive::stage(&mut out, &narrow.to_le_bytes()),
      TensorType::UInt64 => {
         let wide = if value.is_nan() {
            u64::MAX
         } else {
            value as u64
         };
         dive::stage(&mut out, &wide.to_le_bytes())
      },
      TensorType::Float32 => dive::stage(&mut out, &(value as f32).to_le_bytes()),
      TensorType::Float64 => dive::stage(&mut out, &value.to_le_bytes()),
      TensorType::Float16 | TensorType::Bfloat => return None,
   };
   cell[..width].copy_from_slice(&out[..width]);
   Some(())
}

/// Ceases unless both tensors share an element count and hold their bytes.
fn require_cast_shapes(input: &ResolvedTensor, output: &ResolvedTensor, file: &'static str) {
   if input.element_count != output.element_count {
      dive::log("cast element count mismatch", file);
      dive::cease(23, file);
   }
   if input.bytes().len() < input.byte_size() || output.bytes().len() < output.byte_size() {
      dive::log("cast tensor without backing", file);
      dive::cease(19, file);
   }
}

#[inline]
#[must_use]
pub fn cast_fallback(input: &ResolvedTensor, output: &mut ResolvedTensor) -> Status {
   if matches!(input.tensor_type, TensorType::Float16 | TensorType::Bfloat)
      || matches!(output.tensor_type, TensorType::Float16 | TensorType::Bfloat)
   {
      return Status::new(Code::Unimplemented);
   }

   require_cast_shapes(input, output, "cast.cc");
   if input.tensor_type == output.tensor_type {
      let bytes = input.byte_size();
      output.bytes_mut()[..bytes].copy_from_slice(&input.bytes()[..bytes]);
      return Status::ok();
   }
   let in_size = type_size_bytes(input.tensor_type);
   let out_size = type_size_bytes(output.tensor_type);
   let in_type = input.tensor_type;
   let out_type = output.tensor_type;
   let count = input.element_count;
   let src = input.bytes();
   let dst = output.bytes_mut();
   for idx in 0..count {
      let Some(value) = read_as_f64(in_type, &src[idx * in_size..]) else {
         return Status::new(Code::Unimplemented);
      };
      if write_from_f64(out_type, &mut dst[idx * out_size..], value).is_none() {
         return Status::new(Code::Unimplemented);
      }
   }

   Status::ok()
}

/// Casts between dtypes, vectorising only the f32 to i32 pair.
///
/// `vfcvt.rtz.x.f.v` saturates the finite range and maps NaN to [`i32::MAX`].
/// Other supported pairs reach the scalar `Fallback`, which truncates the low
/// bits instead of clamping. Float16 and Bfloat pairs return unimplemented.
#[inline]
#[must_use]
pub fn cast(input: &ResolvedTensor, output: &mut ResolvedTensor) -> Status {
   if input.tensor_type != TensorType::Float32 || output.tensor_type != TensorType::Int32 {
      return cast_fallback(input, output);
   }

   require_cast_shapes(input, output, "cast.cc");
   let count = input.element_count;
   let src = input.bytes();
   let dst = output.bytes_mut();
   for idx in 0..count {
      let cell = &src[idx * 4..];
      let value = f32::from_le_bytes(dive::lane::<4>(cell));
      let out = if value.is_nan() {
         i32::MAX
      } else {
         value as i32
      };
      dst[idx * 4..idx * 4 + 4].copy_from_slice(&out.to_le_bytes());
   }

   Status::ok()
}

#[inline]
#[must_use]
pub const fn us_to_ticks(us: u64) -> u64 {
   ((us as u128 * 96).div_ceil(5)) as u64
}

#[inline]
#[must_use]
pub const fn ns_to_ticks(ns: u64) -> u64 {
   ((ns as u128 * 12).div_ceil(625)) as u64
}

#[inline]
#[must_use]
pub const fn ms_to_ticks(ms: u64) -> u64 {
   ms.saturating_mul(19200)
}

#[inline]
#[must_use]
pub const fn ticks_to_ns(ticks: u64) -> u64 {
   ((ticks as u128 * 625) / 12) as u64
}

#[inline]
#[must_use]
pub const fn ticks_to_us(ticks: u64) -> u64 {
   ((ticks as u128 * 5) / 96) as u64
}

#[inline]
#[must_use]
pub const fn ticks_to_ms(ticks: u64) -> u64 {
   ticks / 19200
}

#[cfg(target_arch = "riscv64")]
fn read_ticks() -> u64 {
   // SAFETY: the Darwin firmware exposes its read-only CLINT mtime register
   // here.
   unsafe { core::ptr::read_volatile(core::ptr::with_exposed_provenance(0x0200_BFF8)) }
}

/// Waits against CLINT on device and uses deterministic spin work elsewhere.
fn spin_ticks(ticks: u64) {
   #[cfg(target_arch = "riscv64")]
   {
      let start = read_ticks();
      while read_ticks().wrapping_sub(start) < ticks {
         spin_loop();
      }
   }

   #[cfg(not(target_arch = "riscv64"))]
   {
      for _tick in 0..ticks {
         spin_loop();
      }
   }
}

#[inline]
pub fn delay_us(us: u32) {
   spin_ticks(us_to_ticks(u64::from(us)));
}

#[inline]
pub fn delay_ns(ns: u32) {
   spin_ticks(ns_to_ticks(u64::from(ns)));
}

#[inline]
pub fn delay_ms(ms: u32) {
   spin_ticks(ms_to_ticks(u64::from(ms)));
}

#[inline]
pub fn custom_absolute_kernel(input: &ResolvedTensor, output: &mut ResolvedTensor) {
   const FILE: &str = "custom_absolute_kernel.cc";

   if input.element_count != output.element_count {
      dive::log("absolute element count mismatch", FILE);
      dive::cease(23, FILE);
   }
   if input.tensor_type != TensorType::Int32 || output.tensor_type != TensorType::Int32 {
      dive::log("absolute expects int32", FILE);
      dive::cease(18, FILE);
   }
   let need = input.element_count * 4;
   if input.bytes().len() < need || output.bytes().len() < need {
      dive::log("absolute tensor without backing", FILE);
      dive::cease(19, FILE);
   }
   let count = input.element_count;
   let src = input.bytes();
   let dst = output.bytes_mut();
   for idx in 0..count {
      let cell = &src[idx * 4..];
      let value = i32::from_le_bytes(dive::lane::<4>(cell));
      dst[idx * 4..idx * 4 + 4].copy_from_slice(&value.wrapping_abs().to_le_bytes());
   }
}

#[inline]
#[must_use]
pub const fn custom_sync_op_kernel() -> Status {
   Status::new(Code::Ok)
}

#[inline]
#[must_use]
pub const fn get_model_firmware_size() -> usize {
   0
}

#[cfg(test)]
mod tests {
   use alloc::{
      vec,
      vec::Vec,
   };

   use dive_abi::status::Code;

   use crate::{
      cast_ops::cast,
      tensor::TensorType,
      test_support::{
         f32_bytes,
         read_i32,
         tensor,
         zero_tensor,
      },
   };

   #[test]
   fn identity_pair_is_byte_copy() {
      let payload: Vec<u8> = (0..16_u8).collect();
      let input = tensor(TensorType::Int32, vec![4], &payload);
      let mut output = zero_tensor(TensorType::Int32, vec![4]);
      assert!(
         cast(&input, &mut output).is_ok(),
         "identity cast should succeed"
      );
      assert_eq!(output.bytes(), payload);

      let half = tensor(TensorType::Float16, vec![1], &[0_u8; 2]);
      let mut float_output = tensor(TensorType::Float32, vec![1], &[0xA5_u8; 4]);
      assert_eq!(cast(&half, &mut float_output).code(), Code::Unimplemented);
      assert_eq!(float_output.bytes(), &[0xA5_u8; 4]);

      let float_input = tensor(TensorType::Float32, vec![1], &[0_u8; 4]);
      let mut bfloat = tensor(TensorType::Bfloat, vec![1], &[0x5A_u8; 2]);
      assert_eq!(cast(&float_input, &mut bfloat).code(), Code::Unimplemented);
      assert_eq!(bfloat.bytes(), &[0x5A_u8; 2]);
   }

   /// `Cast` routes f32 to i32 through `vfcvt.rtz.x.f.v`, which clamps to the
   /// i32 range and maps NaN to `i32::MAX`.
   #[test]
   fn f32_to_i32_saturates_and_maps_nan_to_max() {
      let input = tensor(
         TensorType::Float32,
         vec![4],
         &f32_bytes(&[1.0e30, -1.0e30, f32::NAN, f32::NEG_INFINITY]),
      );
      let mut output = zero_tensor(TensorType::Int32, vec![4]);
      assert!(
         cast(&input, &mut output).is_ok(),
         "saturating f32 to i32 cast should succeed"
      );
      assert_eq!(read_i32(&output), vec![
         i32::MAX,
         i32::MIN,
         i32::MAX,
         i32::MIN
      ]);
   }

   /// The scalar `cast::Fallback` does a 64-bit saturating `fcvt.l.s rtz` and
   /// then keeps the low bits on the narrow store, with no clamp to the
   /// destination.
   #[test]
   fn f32_huge_to_i8_truncates_low_bits() {
      let input = tensor(TensorType::Float32, vec![2], &f32_bytes(&[1.0e30, -1.9]));
      let mut output = zero_tensor(TensorType::Int8, vec![2]);
      assert!(
         cast(&input, &mut output).is_ok(),
         "f32 to i8 cast should succeed"
      );
      let raw: Vec<i8> = output
         .bytes()
         .iter()
         .map(|byte| byte.cast_signed())
         .collect();
      assert_eq!(
         raw,
         vec![-1_i8, -1_i8],
         "1e30 saturates to i64::MAX, low byte 0xFF"
      );
   }
}
