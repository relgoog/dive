#[inline]
pub const fn log(message: &str, file: &str) {
   let _: (&str, &str) = (message, file);
}

/// Ceases execution with the given code.
///
/// # Panics
///
/// Always panics to model the `DiveRuntime_Cease` trap.
#[inline]
#[expect(
   clippy::panic,
   reason = "models DiveRuntime_Cease trap verified by runtime"
)]
pub fn cease(code: i32, file: &str) -> ! {
   panic!("DiveRuntime_Cease {code} at {file}");
}

/// Reads a fixed-size byte lane, ceasing when the backing runs short.
#[inline]
#[must_use]
pub fn lane<const N: usize>(bytes: &[u8]) -> [u8; N] {
   bytes
      .first_chunk::<N>()
      .map_or_else(|| cease(19, "tensor.cc"), |chunk| *chunk)
}

/// Stages one little-endian value into the eight byte scratch, returning its
/// width.
#[inline]
pub fn stage(out: &mut [u8; 8], bytes: &[u8]) -> usize {
   out[..bytes.len()].copy_from_slice(bytes);
   bytes.len()
}

#[inline]
pub fn memcpy(dst: &mut [u8], src: &[u8]) {
   let len = dst.len().min(src.len());
   dst[..len].copy_from_slice(&src[..len]);
}

#[inline]
pub fn memset(dst: &mut [u8], value: u8) {
   dst.fill(value);
}
