//! Shared firmware primitives for the Dive kernel rewrite.
//!
//! Volatile `MmioU64` cells, `CacheLineOps` immediates, `FenceGuard`
//! nesting, `PowerHint` lanes, `AllocatorPtr` addresses, `TraceId`
//! values, and `DtcMode` requests. Uses only `core` so every firmware
//! module can depend on it.

use core::{
   cell::UnsafeCell,
   ptr,
};

/// Volatile `u64` cell over one MMIO register.
///
/// Transparent over `UnsafeCell` so a reference shares the register address,
/// and every access uses a volatile intrinsic so the compiler keeps each
/// read and write the firmware issued.
#[derive(Debug)]
#[repr(transparent)]
pub struct MmioU64 {
   /// Register storage behind the MMIO address.
   cell: UnsafeCell<u64>,
}

impl MmioU64 {
   /// Wraps a reset value for host-side tests.
   #[inline]
   #[must_use]
   pub const fn new(value: u64) -> Self {
      Self {
         cell: UnsafeCell::new(value),
      }
   }

   /// Returns the register address.
   #[inline]
   #[must_use]
   pub const fn addr(&self) -> *mut u64 {
      self.cell.get()
   }

   /// Performs one volatile register read.
   #[inline]
   #[must_use]
   pub fn read(&self) -> u64 {
      // SAFETY: `addr` is one mapped MMIO register owned by this cell.
      unsafe { ptr::read_volatile(self.addr()) }
   }

   /// Performs one volatile register write.
   #[inline]
   pub fn write(&self, value: u64) {
      // SAFETY: `addr` is one mapped MMIO register owned by this cell.
      unsafe { ptr::write_volatile(self.addr(), value) }
   }

   /// Borrows the register living at `addr`.
   ///
   /// # Safety
   ///
   /// The caller must ensure `addr` names one mapped readable and writable
   /// `u64` MMIO register for the whole borrow.
   #[inline]
   #[must_use]
   pub const unsafe fn at(addr: usize) -> &'static Self {
      // SAFETY: upheld by the caller, and `MmioU64` is transparent over `u64`.
      unsafe { &*(addr as *const Self) }
   }
}

// SAFETY: MMIO registers are process-wide hardware cells shared by design.
unsafe impl Sync for MmioU64 {}

/// Cache maintenance immediates from `DiveSystem_CacheFlush` and
/// `DiveSystem_CacheFlushFull`.
pub struct CacheLineOps;

impl CacheLineOps {
   /// One cache line, the `0xC0` alignment granularity.
   pub const LINE_LEN: usize = 64;
   /// Outer loop stride, eight lines per chunk.
   pub const STRIDE: usize = 512;
   /// Lines covered per stride.
   pub const LINES_PER_STRIDE: usize = 8;
   /// Command bits `ORed` with each line address, `0x3 << 56`.
   pub const CMD_OR: u64 = 0x03_u64 << 56;
   /// Full-flush command word, `0x13 << 56`.
   pub const FULL_FLUSH_CMD: u64 = 0x13_u64 << 56;
   /// Cache controller region base reported by the full-flush path.
   pub const BASE_ADDR: usize = 0x1010_4000;
   /// Command register written once per line.
   pub const CMD_ADDR: usize = 0x1010_4200;
   /// Status register polled until it reads zero.
   pub const STATUS_ADDR: usize = 0x1010_4208;

   /// Aligns `addr` down to a line boundary.
   #[inline]
   #[must_use]
   pub const fn align_down(addr: usize) -> usize {
      addr & !(Self::LINE_LEN - 1)
   }

   /// Builds the command word for one line address.
   #[inline]
   #[must_use]
   pub const fn line_command(line_addr: usize) -> u64 {
      (line_addr as u64) | Self::CMD_OR
   }
}

/// Nesting counter behind `DiveSystem_EnterCriticalSection` and
/// `DiveSystem_ExitCriticalSection`.
///
/// While `is_locked` reports held, fence waiters keep spinning, and the
/// outermost `exit` restores `MIE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FenceGuard {
   /// Nesting depth, zero means outside any critical section.
   depth: u32,
}

impl FenceGuard {
   /// Starts outside any critical section.
   #[inline]
   #[must_use]
   pub const fn new() -> Self {
      Self { depth: 0 }
   }

   /// Current nesting depth.
   #[inline]
   #[must_use]
   pub const fn depth(&self) -> u32 {
      self.depth
   }

   /// Whether the fence is held, so waiters keep spinning.
   #[inline]
   #[must_use]
   pub const fn is_locked(&self) -> bool {
      self.depth != 0
   }

   /// Enters one critical section.
   ///
   /// Matches `DiveSystem_EnterCriticalSection`, the depth grows when
   /// already nested or `mie_was_set` reports interrupts were enabled,
   /// and the caller clears `MIE` in exactly that case.
   #[inline]
   #[expect(
      clippy::missing_const_for_fn,
      reason = "fence entry must stay a runtime transition so const contexts cannot reorder \
                interrupt masking"
   )]
   pub fn enter(&mut self, mie_was_set: bool) {
      if self.depth != 0 || mie_was_set {
         self.depth += 1;
      }
   }

   /// Exits one critical section.
   ///
   /// Matches `DiveSystem_ExitCriticalSection`, the depth falls only when
   /// nonzero, and the return tells the caller to restore `MIE` on the
   /// outermost exit.
   #[inline]
   #[must_use]
   #[expect(
      clippy::missing_const_for_fn,
      reason = "fence exit must stay a runtime transition so const contexts cannot reorder \
                interrupt masking"
   )]
   pub fn exit(&mut self) -> bool {
      if self.depth == 0 {
         false
      } else {
         self.depth -= 1;
         self.depth == 0
      }
   }
}

/// Two lane power hint from `DivePower_UpdateHint`.
///
/// Kept as the raw `i16` pair the firmware forwards to the power handler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerHint {
   /// First hint lane, `a1` in the pseudocode.
   pub first:  i16,
   /// Second hint lane, `a2` in the pseudocode.
   pub second: i16,
}

impl PowerHint {
   /// Packs both hint lanes.
   #[inline]
   #[must_use]
   pub const fn new(first: i16, second: i16) -> Self {
      Self { first, second }
   }
}

/// Opaque allocator address installed by `DiveRuntime_SetAllocator`.
///
/// Stored as a plain address so host builds never fabricate a reference,
/// and the runtime rejects the null address exactly like the firmware trap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocatorPtr(usize);

impl AllocatorPtr {
   /// Null installer address, always rejected by the runtime.
   pub const NULL: Self = Self(0);

   /// Wraps a raw allocator address.
   #[inline]
   #[must_use]
   pub const fn new(addr: usize) -> Self {
      Self(addr)
   }

   /// Returns the wrapped address.
   #[inline]
   #[must_use]
   pub const fn addr(self) -> usize {
      self.0
   }

   /// Whether this is the rejected null address.
   #[inline]
   #[must_use]
   pub const fn is_null(self) -> bool {
      self.0 == 0
   }
}

/// Operation trace identifier from the tracing path.
///
/// Stored narrow because `StartOperationTrace` keeps the id in one `_WORD`
/// slot of the trace entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraceId(u16);

impl TraceId {
   /// Wraps a trace identifier.
   #[inline]
   #[must_use]
   pub const fn new(id: u16) -> Self {
      Self(id)
   }

   /// Returns the wrapped identifier.
   #[inline]
   #[must_use]
   pub const fn value(self) -> u16 {
      self.0
   }
}

/// DTC mode requested through `DiveDtc_SetDtcMode`.
///
/// The firmware traps on every path here because this target wires no
/// `DTC`, so the enum only names the request for logging and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DtcMode {
   /// DTC disabled.
   #[default]
   Disabled = 0,
   /// DTC enabled.
   Enabled  = 1,
}

impl DtcMode {
   /// Whether the request enables the `DTC`.
   #[inline]
   #[must_use]
   pub const fn is_enabled(self) -> bool {
      matches!(self, Self::Enabled)
   }
}
