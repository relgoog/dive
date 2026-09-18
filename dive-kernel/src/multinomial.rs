//! Multinomial sampling kernels.
//!
//! The device kernels exponentiate the logits in place, prefix-sum them in
//! place, then scan for the first partial sum above `uniform * total`. `exp` is
//! the product form `2^k * (1 + r(1 + (r/2)(1 + (r/3)(1 + r/4))))` with
//! `k = rne(x * log2e)` and `r = x - k * ln2`, evaluated with `vfmadd.vv`
//! chains, and the scale collapses to zero at `k <= -128`.

use core::f32::{
   consts::{
      LN_2,
      LOG2_E,
   },
   math::{
      mul_add,
      round_ties_even,
   },
};

use dive_abi::dive;

/// Seed of the `vfredmax.vs` reduction, so an all-NaN row still reduces.
const MAX_SEED: f32 = -f32::MAX;

/// Approximates exponential logits using a polynomial expansion.
fn exp_approx(value: f32) -> f32 {
   let exponent = round_ties_even(value * LOG2_E);
   let steps = exponent as i32;
   if steps <= -128_i32 {
      return 0.0;
   }
   let scale = f32::from_bits(((steps + 127_i32).cast_unsigned()) << 23);
   let rem = mul_add(exponent, -LN_2, value);
   let mut poly = mul_add(rem, 0.25, 1.0);
   poly = mul_add(rem * (1.0 / 3.0), poly, 1.0);
   poly = mul_add(rem * 0.5, poly, 1.0);
   poly = mul_add(rem, poly, 1.0);
   poly * scale
}

/// `vfredmax.vs` ignores NaN operands, which is what `f32::max` does.
fn row_max(logits: &[f32]) -> f32 {
   logits.iter().fold(MAX_SEED, |acc, logit| acc.max(*logit))
}

/// Rewrites `logits` with the running softmax numerator sum.
///
/// Returns the index of the first partial sum above the threshold, or
/// `logits.len()` when the scan falls off the end.
fn sample_in_place(logits: &mut [f32], uniform: f32) -> usize {
   let max = row_max(logits);
   let mut acc = 0.0_f32;
   for logit in logits.iter_mut() {
      acc += exp_approx(*logit - max);
      *logit = acc;
   }
   let threshold = uniform * acc;
   logits
      .iter()
      .position(|partial| threshold < *partial)
      .unwrap_or(logits.len())
}

/// Maps the sampled index onto the normalised output coordinate.
///
/// The kernels apply this when the remap flag is set, rounding half to even.
fn remap_index(index: usize, depth: usize, scale: f32, offset: i32) -> i32 {
   let span = 2.0_f32 / depth as f32;
   let point = mul_add(span, index as f32, -1.0) / scale;
   offset.wrapping_add(round_ties_even(point) as i32)
}

/// Samples an outcome from logits with optional index remapping.
fn sample(logits: &mut [f32], uniform: f32, remap: bool, scale: f32, offset: i32) -> i32 {
   let depth = logits.len();
   if depth == 0 {
      dive::cease(3, "multinomial.cc");
   }
   let index = sample_in_place(logits, uniform);
   let Ok(narrow) = i32::try_from(index) else {
      dive::cease(3, "multinomial.cc");
   };
   if index == depth || !remap {
      return narrow;
   }
   remap_index(index, depth, scale, offset)
}

/// `ComputeMultinomialScalar(float*, float*, int, bool, float, int, int,
/// void*)`.
#[inline]
pub fn compute_multinomial_scalar(
   logits: &mut [f32],
   uniform: f32,
   remap: bool,
   scale: f32,
   offset: i32,
) -> i32 {
   sample(logits, uniform, remap, scale, offset)
}

/// `ComputeMultinomialVectorized` shares the scalar signature and result.
///
/// It differs only in using `vfredmax.vs` and an eight-lane `vle32.v` body,
/// which keeps the same sequential accumulation order through `vfredosum.vs`.
#[inline]
pub fn compute_multinomial_vectorized(
   logits: &mut [f32],
   uniform: f32,
   remap: bool,
   scale: f32,
   offset: i32,
) -> i32 {
   sample(logits, uniform, remap, scale, offset)
}

/// `ComputeMultinomialVectorizedSize3(float*, float*, short*)`.
///
/// Fixed at three logits, no remap, and no clamp on `uniform`. Returns 3 when
/// the scan falls off the end, which is one past the last valid index.
///
/// # Panics
///
/// Panics when fewer than three logits are provided.
#[inline]
pub fn compute_multinomial_vectorized_size3(logits: &mut [f32], uniform: f32) -> i16 {
   if logits.len() < 3 {
      dive::cease(3, "multinomial.cc");
   }
   i16::try_from(sample_in_place(&mut logits[..3], uniform)).unwrap_or(3)
}

/// `ComputeMultinomialBatch4Size3Padding1(float*, float*, short*)`.
///
/// # Panics
///
/// Panics when the batch does not hold four padded rows of three logits.
#[inline]
pub fn compute_multinomial_batch4_size3_padding1(
   logits: &mut [f32],
   uniforms: &[f32],
   outputs: &mut [i16],
) {
   if logits.len() < 12 || uniforms.len() < 4 || outputs.len() < 4 {
      dive::cease(3, "multinomial.cc");
   }
   for row in 0..4 {
      outputs[row] =
         compute_multinomial_vectorized_size3(&mut logits[row * 3..row * 3 + 3], uniforms[row]);
   }
}

#[cfg(test)]
mod tests {
   use crate::multinomial::{
      compute_multinomial_scalar,
      compute_multinomial_vectorized_size3,
   };

   /// The kernels overwrite the logit buffer with the running softmax sum
   /// before scanning it, so the caller sees a prefix-summed row on return.
   #[test]
   fn rewrites_logits_with_prefix_sums() {
      let mut logits = [1.0_f32, 5.0, 2.0, 4.0];
      let pick = compute_multinomial_scalar(&mut logits, 0.0, false, 1.0, 0);
      assert_eq!(pick, 0_i32);
      assert!(logits.is_sorted(), "{logits:?}");
      assert!(
         logits.last().is_some_and(|total| *total > 0.0),
         "softmax prefix total should be positive"
      );
   }

   #[test]
   fn uniform_one_falls_off_the_end() {
      let mut logits = [1.0_f32, 5.0, 2.0, 4.0];
      let pick = compute_multinomial_scalar(&mut logits, 1.0, false, 1.0, 0);
      assert_eq!(pick, 4_i32);
   }

   #[test]
   fn remap_shifts_onto_the_normalised_axis() {
      let mut logits = [1.0_f32, 5.0, 2.0, 4.0];
      let pick = compute_multinomial_scalar(&mut logits, 0.0, true, 1.0, 10);
      assert_eq!(pick, 9_i32, "index 0 maps to -1.0 then offsets by 10");
   }

   #[test]
   fn size3_returns_three_when_the_scan_falls_off() {
      let mut logits = [1.0_f32, 2.0, 3.0];
      assert_eq!(compute_multinomial_vectorized_size3(&mut logits, 1.0), 3);
   }
}
