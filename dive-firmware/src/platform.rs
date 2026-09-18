//! Host shims for the Darwin tile SRAM, TMU CSR, and UART platform calls.

use alloc::string::String;
use core::fmt;

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

pub const TILE_SRAM_BYTES: usize = 786_432;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMappingTable {
   pub extents: [i64; 4],
   pub coef_a:  [i64; 4],
   pub coef_b:  [i64; 4],
   pub window:  [i64; 4],
}

impl TensorMappingTable {
   #[inline]
   #[must_use]
   pub const fn new(
      extents: [i64; 4],
      coef_a: [i64; 4],
      coef_b: [i64; 4],
      window: [i64; 4],
   ) -> Self {
      Self {
         extents,
         coef_a,
         coef_b,
         window,
      }
   }

   #[inline]
   #[must_use]
   pub fn validate(&self) -> Status {
      for i in 0..4 {
         if self.extents[i] <= 0
            || self.window[i] <= 0
            || self.window[i] > self.extents[i]
            || self.coef_a[i] < 0
            || self.coef_b[i] < 0
         {
            return Status::new(Code::InvalidArgument);
         }
      }
      Status::ok()
   }

   #[inline]
   #[must_use]
   pub fn element_count(&self) -> usize {
      self
         .window
         .iter()
         .fold(1_usize, |acc, w| acc.saturating_mul(*w as usize))
   }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileTmuCsrOffsets {
   pub offsets: [u32; 8],
}

impl TileTmuCsrOffsets {
   #[inline]
   #[must_use]
   pub const fn new(offsets: [u32; 8]) -> Self {
      Self { offsets }
   }

   #[inline]
   #[must_use]
   pub fn validate(&self) -> Status {
      for offset in self.offsets {
         if offset % 4 != 0 {
            return Status::new(Code::InvalidArgument);
         }
      }
      Status::ok()
   }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerformanceCounters {
   pub cycles:       u64,
   pub instructions: u64,
}

impl PerformanceCounters {
   #[inline]
   #[must_use]
   pub const fn new(cycles: u64, instructions: u64) -> Self {
      Self {
         cycles,
         instructions,
      }
   }
}

#[inline]
pub fn mem_cpy_between_tile_sram_bypass_mode(dst: &mut [u8], src: &[u8], len: usize) -> Status {
   if len > dst.len() || len > src.len() {
      return Status::new(Code::OutOfRange);
   }
   dst[..len].copy_from_slice(&src[..len]);
   Status::ok()
}

#[inline]
pub fn mem_cpy_from_tile_sram_to_dram_bypass_mode(
   dst: &mut [u8],
   src: &[u8],
   len: usize,
) -> Status {
   if len > dst.len() || len > src.len() {
      return Status::new(Code::OutOfRange);
   }
   dst[..len].copy_from_slice(&src[..len]);
   Status::ok()
}

#[inline]
pub fn write_tile_sram_bypass_mode(dst: &mut [u8], dst_offset: usize, src: &[u8]) -> Status {
   let Some(end) = dst_offset.checked_add(src.len()) else {
      return Status::new(Code::OutOfRange);
   };
   if end > dst.len() {
      return Status::new(Code::OutOfRange);
   }
   dst[dst_offset..end].copy_from_slice(src);
   Status::ok()
}

#[inline]
pub fn copy_sliced_tensor(
   dst: &mut [u8],
   src: &[u8],
   dim0: i32,
   dim1: i32,
   tile: i32,
   table: &TensorMappingTable,
) -> Status {
   if dim0 < 0_i32 || dim1 < 0_i32 || tile < 0_i32 {
      return Status::new(Code::InvalidArgument);
   }
   let status = table.validate();
   if !status.is_ok() {
      return status;
   }
   let count = table.element_count();
   if count > dst.len() || count > src.len() {
      return Status::new(Code::OutOfRange);
   }
   dst[..count].copy_from_slice(&src[..count]);
   Status::ok()
}

#[inline]
#[must_use]
pub fn program_tensor_mapping_table(table: &TensorMappingTable, unit: i32, tile: i32) -> Status {
   if unit < 0_i32 || tile < 0_i32 {
      return Status::new(Code::InvalidArgument);
   }
   let status = table.validate();
   if !status.is_ok() {
      return status;
   }
   Status::ok()
}

#[inline]
#[must_use]
pub fn program_tile_tmu_csrs(
   offsets: &TileTmuCsrOffsets,
   table: &TensorMappingTable,
   tile: i32,
) -> Status {
   if tile < 0_i32 {
      return Status::new(Code::InvalidArgument);
   }
   let status = offsets.validate();
   if !status.is_ok() {
      return status;
   }
   let table_status = table.validate();
   if !table_status.is_ok() {
      return table_status;
   }
   Status::ok()
}

#[inline]
pub const fn log_performance_counters(counters: PerformanceCounters) {
   let _: PerformanceCounters = counters;
}

#[inline]
pub fn terminate_simulation() -> ! {
   dive::cease(0, "platform.cc");
}

#[inline]
pub const fn configure_uart() {}

#[inline]
pub fn printf_info(args: fmt::Arguments<'_>) {
   let mut out = String::new();
   let _result: fmt::Result = fmt::Write::write_fmt(&mut out, args);
   dive::log(&out, "uart.cc");
}
