//! Bfloat16 conversions using round to nearest, ties to even.

/// Widens bfloat16 bits to binary32.
#[inline]
#[must_use]
pub const fn bf16_to_f32(bits: u16) -> f32 {
   f32::from_bits((bits as u32) << 16_i32)
}

/// Narrows binary32 to bfloat16 bits, rounding half to even.
#[inline]
#[must_use]
pub const fn f32_to_bf16(value: f32) -> u16 {
   let bits = value.to_bits();
   let rounding = 0x7FFF + ((bits >> 16_i32) & 1);
   bits.wrapping_add(rounding).wrapping_shr(16) as u16
}
