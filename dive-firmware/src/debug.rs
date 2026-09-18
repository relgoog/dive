//! Debug, fence, cache, critical-section, doorbell, and tracing shims.
//!
//! Host-testable rewrite of the firmware `DiveDebug_*` trap and fence paths,
//! the `DiveTpu_WaitForFence*` waits, the `DiveSystem_Cache*` maintenance ops,
//! the `DiveSystem_EnterCriticalSection` / `DiveSystem_ExitCriticalSection`
//! nesting guard, the `DiveSystem_SendControlClusterInterrupt` doorbell, and
//! the `DiveTracing_*` / `DiveItcTracing_*` capture paths.
//!
//! Callers own every hardware cell. MMIO blocks arrive as [`MmioU64`]
//! references, fence nesting as [`FenceGuard`], and trace ids as [`TraceId`],
//! so this module keeps no globals and stays testable on the host.

use core::{
   hint::spin_loop,
   ptr::{
      read_volatile,
      write_volatile,
   },
   sync::atomic::{
      Ordering,
      fence,
   },
};

#[cfg(not(target_arch = "riscv64"))] use dive_abi::dive;

use crate::types::{
   CacheLineOps,
   FenceGuard,
   MmioU64,
   TraceId,
};

/// Byte offset of the fence-completion flag from the fence state base.
pub const FENCE_FLAG_OFFSET: usize = 0x188;

/// Token the multicore fence wait reports when the incoming fence was already
/// complete, preserving the firmware `0x1B2AA` default.
pub const MULTICORE_IDLE_FENCE_TOKEN: u32 = 0x1_B2AA;

/// Software trace ring capacity, matching the firmware `0x20` entry ring.
pub const TRACE_CAP: usize = 32;

/// `MIE` bit in `mstatus`.
#[cfg(target_arch = "riscv64")]
const MIE_BIT: u64 = 8;

/// Reads `MIE`, clearing it on device.
///
/// The firmware enter path issues one `read_clear_csr(mstatus, 8)` and treats
/// a set bit as proof interrupts were enabled, so the read and the clear stay
/// a single instruction on device. Host builds have no `mstatus` and report
/// enabled so the [`FenceGuard`] depth still nests.
#[inline]
#[cfg_attr(
   not(target_arch = "riscv64"),
   expect(
      clippy::missing_const_for_fn,
      reason = "host fallback reports enabled so fence nesting still exercises the guard"
   )
)]
fn mie_was_set() -> bool {
   #[cfg(target_arch = "riscv64")]
   {
      let status: u64;
      // SAFETY, `csrrci` only reads and clears the `MIE` bit of `mstatus`.
      unsafe {
         core::arch::asm!(
            "csrrci {old}, mstatus, 8",
            old = out(reg) status,
            options(nomem, nostack)
         );
      }
      (status & MIE_BIT) != 0
   }
   #[cfg(not(target_arch = "riscv64"))]
   {
      true
   }
}

/// Restores `MIE` after the outermost critical-section exit.
#[inline]
#[cfg_attr(
   not(target_arch = "riscv64"),
   expect(
      clippy::missing_const_for_fn,
      reason = "host fallback keeps the critical-section shape without device CSR access"
   )
)]
fn restore_mie() {
   #[cfg(target_arch = "riscv64")]
   {
      // SAFETY, `csrrsi` only sets the `MIE` bit of `mstatus`.
      unsafe {
         core::arch::asm!("csrrsi zero, mstatus, 8", options(nomem, nostack));
      }
   }
}

/// Waits for the next interrupt on device, spins arms-length elsewhere.
#[inline]
fn wait_for_interrupt() {
   #[cfg(target_arch = "riscv64")]
   {
      // SAFETY, `wfi` takes no operands and only sleeps until an interrupt.
      unsafe {
         core::arch::asm!("wfi", options(nomem, nostack));
      }
   }
   #[cfg(not(target_arch = "riscv64"))]
   {
      spin_loop();
   }
}

/// Reads the fence-completion flag byte at `state + FENCE_FLAG_OFFSET`.
///
/// # Safety
///
/// `state` must point at live fence state whose flag byte the hardware sets
/// on completion.
#[inline]
unsafe fn flag_is_set(state: *const u8) -> bool {
   let flag = state.wrapping_byte_add(FENCE_FLAG_OFFSET);
   // SAFETY: upheld by the caller contract above.
   unsafe { read_volatile(flag) != 0 }
}

/// Zero-clears the fence-completion flag byte at `state + FENCE_FLAG_OFFSET`.
///
/// # Safety
///
/// `state` must point at live fence state.
#[inline]
unsafe fn clear_flag(state: *mut u8) {
   // SAFETY: upheld by the caller contract above.
   unsafe {
      write_volatile(state.wrapping_byte_add(FENCE_FLAG_OFFSET), 0);
   }
}

/// Spins until the fence flag reads set, then zero-clears it.
///
/// Shares one `wfi` loop across every fence wait. The guard opens around each
/// sleep so the completion interrupt can arrive, then closes around the
/// clear, matching the firmware nesting count exactly.
///
/// # Safety
///
/// `state` must point at live fence state whose flag byte the hardware sets
/// on completion.
unsafe fn poll_fence_flag(state: *mut u8, guard: &mut FenceGuard) {
   guard.enter(mie_was_set());
   loop {
      fence(Ordering::SeqCst);
      // SAFETY: live fence state per the caller contract above.
      if unsafe { flag_is_set(state) } {
         break;
      }
      fence(Ordering::SeqCst);
      if guard.exit() {
         restore_mie();
      }
      wait_for_interrupt();
      guard.enter(mie_was_set());
   }
   fence(Ordering::SeqCst);
   #[expect(
      clippy::semicolon_outside_block,
      reason = "outer semicolon keeps the fence clear visibly sequenced with the surrounding \
                fences"
   )]
   // SAFETY: same live fence state the flag read above used.
   unsafe {
      clear_flag(state);
   }
   fence(Ordering::SeqCst);
   if guard.exit() {
      restore_mie();
   }
}

/// Firmware `DiveDebug_WaitForFenceCompletionAndClearMemory`.
///
/// Blocks on the shared flag poll, which zero-clears the flag before the
/// guard closes.
///
/// # Safety
///
/// `state` must point at live fence state whose flag byte at
/// [`FENCE_FLAG_OFFSET`] the hardware sets on completion.
#[inline]
pub unsafe fn debug_wait_for_fence_completion_and_clear_memory(
   state: *mut u8,
   guard: &mut FenceGuard,
) {
   // SAFETY: the caller upholds the fence state contract above.
   unsafe {
      poll_fence_flag(state, guard);
   }
}

/// Firmware `DiveTpu_WaitForFenceCompletion`.
///
/// Single-core completion reduces to the shared wait. The post-fence
/// software-preemption dispatch the firmware issues afterwards belongs to the
/// runtime executor, which owns the scheduler state this module never touches.
///
/// # Safety
///
/// `state` must point at live fence state whose flag byte at
/// [`FENCE_FLAG_OFFSET`] the hardware sets on completion.
#[inline]
pub unsafe fn tpu_wait_for_fence_completion(state: *mut u8, guard: &mut FenceGuard) {
   // SAFETY: the caller upholds the fence state contract above.
   unsafe {
      debug_wait_for_fence_completion_and_clear_memory(state, guard);
   }
}

/// Firmware `DiveTpu_WaitForFenceCompletionMulticore`.
///
/// A nonzero `fence` token means completion was already observed, so it
/// passes through untouched. A zero token waits on the shared poll and
/// reports [`MULTICORE_IDLE_FENCE_TOKEN`].
///
/// # Safety
///
/// `state` must point at live fence state whose flag byte at
/// [`FENCE_FLAG_OFFSET`] the hardware sets on completion.
#[inline]
#[must_use]
pub unsafe fn tpu_wait_for_fence_completion_multicore(
   state: *mut u8,
   guard: &mut FenceGuard,
   fence: u32,
) -> u32 {
   if fence == 0 {
      #[expect(
         clippy::semicolon_outside_block,
         reason = "outer semicolon keeps the fence wait visibly sequenced before the idle token"
      )]
      // SAFETY: the caller upholds the fence state contract above.
      unsafe {
         poll_fence_flag(state, guard);
      }
      MULTICORE_IDLE_FENCE_TOKEN
   } else {
      fence
   }
}

/// Writes one [`CacheLineOps`] line command.
#[inline]
fn issue_line(line_addr: usize) {
   // SAFETY: `CMD_ADDR` names the cache command register for the whole borrow.
   unsafe {
      MmioU64::at(CacheLineOps::CMD_ADDR).write(CacheLineOps::line_command(line_addr));
   }
}

/// Writes the full-flush command word.
#[inline]
fn issue_full_flush() {
   // SAFETY: `CMD_ADDR` names the cache command register for the whole borrow.
   unsafe {
      MmioU64::at(CacheLineOps::CMD_ADDR).write(CacheLineOps::FULL_FLUSH_CMD);
   }
}

/// Spins until the cache controller status register reads zero.
#[inline]
fn poll_cache_status() {
   // SAFETY: `STATUS_ADDR` names the cache status register for the whole
   // borrow.
   let status = unsafe { MmioU64::at(CacheLineOps::STATUS_ADDR) };
   while status.read() != 0 {
      spin_loop();
   }
}

/// Runs one 64-byte-line range op.
///
/// Covers eight lines per `0x200` stride with the `0x3 << 56` encoding, then
/// polls status to zero. Both range ops share this helper because the
/// firmware issues the identical command sequence for flush and
/// clean-invalidate on this target.
#[inline]
fn cache_range_op(base: usize, len: usize) {
   fence(Ordering::SeqCst);
   let last = CacheLineOps::align_down(base + len);
   let mut line = CacheLineOps::align_down(base);

   while last >= line {
      let mut offset = 0;
      while offset < CacheLineOps::STRIDE {
         if last < line + offset {
            break;
         }
         issue_line(line + offset);
         offset += CacheLineOps::LINE_LEN;
      }
      line += CacheLineOps::STRIDE;
   }

   poll_cache_status();
}

/// Runs one full-controller flush.
///
/// Issues the `0x13 << 56` command, polls status to zero, and reports the
/// controller base like the firmware.
#[inline]
fn cache_full_op() -> usize {
   fence(Ordering::SeqCst);
   issue_full_flush();
   poll_cache_status();
   CacheLineOps::BASE_ADDR
}

/// Firmware `DiveSystem_CacheFlush`.
#[inline]
pub fn system_cache_flush(base: usize, len: usize) {
   cache_range_op(base, len);
}

/// Firmware `DiveSystem_CacheCleanInvalidate`.
///
/// Delegates to [`system_cache_flush`] because the firmware issues the
/// identical `0x3 << 56` line sequence for both ops on this target.
#[inline]
pub fn system_cache_clean_invalidate(base: usize, len: usize) {
   system_cache_flush(base, len);
}

/// Firmware `DiveSystem_CacheFlushFull`.
#[inline]
#[must_use]
pub fn system_cache_flush_full() -> usize {
   cache_full_op()
}

/// Firmware `DiveSystem_CacheCleanInvalidateFull`.
///
/// Delegates to [`system_cache_flush_full`] because the firmware issues the
/// identical `0x13 << 56` command for both full ops on this target.
#[inline]
#[must_use]
pub fn system_cache_clean_invalidate_full() -> usize {
   system_cache_flush_full()
}

/// Firmware `DiveSystem_EnterCriticalSection`.
///
/// Reports `MIE` into the [`FenceGuard`] depth. The read already cleared `MIE`
/// on device, and fence waiters keep spinning while the guard reads locked.
#[inline]
pub fn system_enter_critical_section(guard: &mut FenceGuard) {
   guard.enter(mie_was_set());
}

/// Firmware `DiveSystem_ExitCriticalSection`.
///
/// Drops one nesting level and restores `MIE` only on the outermost exit.
#[inline]
pub fn system_exit_critical_section(guard: &mut FenceGuard) {
   if guard.exit() {
      restore_mie();
   }
}

/// Firmware `DiveSystem_SendControlClusterInterrupt`.
///
/// One volatile write to the caller-owned control-cluster doorbell. The
/// register address stays with the caller because cluster routing is a board
/// property, not a firmware constant.
#[inline]
pub fn system_send_control_cluster_interrupt(doorbell: &MmioU64, value: u64) {
   doorbell.write(value);
}

/// Raises the `ebreak`-equivalent trap.
#[inline]
fn trap() {
   #[cfg(target_arch = "riscv64")]
   {
      // SAFETY, `ebreak` raises the firmware breakpoint trap with no operands.
      unsafe {
         core::arch::asm!("ebreak", options(nomem, nostack));
      }
   }
   #[cfg(not(target_arch = "riscv64"))]
   {
      dive::cease(2, "debug.cc");
   }
}

/// Firmware `DiveDebug_CustomBreakpoint`.
///
/// Traps only when breakpoints are `enabled`, `break_id` names one of the 32
/// mask lanes, and that lane bit is set in `mask`. Every other input is a
/// no-op, matching the firmware gate.
#[inline]
pub fn debug_custom_breakpoint(break_id: u32, enabled: bool, mask: u32) {
   if !enabled {
      return;
   }
   if !(1..=32).contains(&break_id) {
      return;
   }
   if (mask & (1_u32 << (break_id - 1))) == 0 {
      return;
   }
   trap();
}

/// One software trace entry: the firmware three-word slot plus the op lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceEntry {
   /// The three firmware entry words: stamp, aux, and value.
   pub words: [u64; 3],
   /// Operation id kept in the entry `_WORD` lane.
   pub op:    u16,
}

/// Bounded software trace stack behind `DiveTracing_*`, with an enable gate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceRing {
   /// Buffered entries, oldest first.
   entries: [TraceEntry; TRACE_CAP],
   /// Count of live entries.
   len:     usize,
   /// While set, start calls append; while clear, they report dropped.
   enabled: bool,
}

impl TraceRing {
   /// Starts empty with the gate clear.
   #[inline]
   #[must_use]
   pub const fn new() -> Self {
      Self {
         entries: [TraceEntry {
            words: [0; 3],
            op:    0,
         }; TRACE_CAP],
         len:     0,
         enabled: false,
      }
   }

   /// Opens or closes the append gate.
   #[inline]
   #[expect(
      clippy::missing_const_for_fn,
      reason = "gate flip stays a runtime transition so const contexts cannot reorder tracing \
                capture"
   )]
   pub fn set_enabled(&mut self, enabled: bool) {
      self.enabled = enabled;
   }

   /// Whether the append gate is open.
   #[inline]
   #[must_use]
   pub const fn is_enabled(&self) -> bool {
      self.enabled
   }

   /// Count of live entries.
   #[inline]
   #[must_use]
   pub const fn len(&self) -> usize {
      self.len
   }

   /// Whether no entries are buffered.
   #[inline]
   #[must_use]
   pub const fn is_empty(&self) -> bool {
      self.len == 0
   }

   /// Firmware `DiveTracing_StartTrace`.
   ///
   /// Appends a stamp-plus-value entry with a clear op lane. Reports whether
   /// the entry buffered.
   #[inline]
   pub fn start_trace(&mut self, id: TraceId, stamp: u64) -> bool {
      self.push(stamp, u64::from(id.value()), 0)
   }

   /// Firmware `DiveTracing_StartOperationTrace`.
   ///
   /// Appends a stamp-plus-op entry with a clear value word. Reports whether
   /// the entry buffered.
   #[inline]
   pub fn start_operation_trace(&mut self, op: TraceId, stamp: u64) -> bool {
      self.push(stamp, 0, op.value())
   }

   /// Firmware `DiveTracing_EndTrace`.
   #[inline]
   #[expect(
      clippy::missing_const_for_fn,
      reason = "pop shortens the live trace window so const contexts cannot reorder capture"
   )]
   pub fn end_trace(&mut self) -> bool {
      if self.is_empty() {
         false
      } else {
         self.len -= 1;
         true
      }
   }

   /// Appends one entry through the enable gate.
   ///
   /// A full ring pops one entry and drops the newcomer, matching the
   /// firmware overflow path.
   fn push(&mut self, stamp: u64, value: u64, op: u16) -> bool {
      if !self.enabled {
         return false;
      }
      if self.len >= TRACE_CAP {
         self.end_trace();
         return false;
      }
      match self.entries.get_mut(self.len) {
         Some(slot) => {
            *slot = TraceEntry {
               words: [stamp, 0, value],
               op,
            };
            self.len += 1;
            true
         },
         None => false,
      }
   }
}

/// Firmware `DiveItcTracing_Enable`.
///
/// Programs the control, filter, OR-mask, and set-all registers with the
/// firmware immediates.
#[inline]
pub fn itc_tracing_enable(control: &MmioU64, filter: &MmioU64, mask: &MmioU64, mask_set: &MmioU64) {
   control.write((control.read() & 0x0FFF_FE78) | 0x5000_0087);
   filter.write((filter.read() & 0xFFFF_80FC) | 0x5203);
   mask.write(mask.read() | 0x4_8003);
   mask_set.write(u64::MAX);
}

/// Firmware `DiveItcTracing_Disable`.
///
/// Clears the run bit, spins until the idle bit reads set, then clears the
/// enable bit.
#[inline]
pub fn itc_tracing_disable(control: &MmioU64) {
   control.write(control.read() & !2_u64);
   while (control.read() & 8) == 0 {
      spin_loop();
   }
   control.write(control.read() & !1_u64);
}

/// Firmware `DiveItcTracing_StartOperationTrace`.
///
/// Spins until the ready register reads nonzero, then posts the op id.
#[inline]
pub fn itc_tracing_start_operation_trace(ready: &MmioU64, slot: &MmioU64, op: TraceId) {
   while ready.read() == 0 {
      spin_loop();
   }
   slot.write(u64::from(op.value()));
}

/// Firmware `DiveItcTracing_StartTrace`.
///
/// The firmware hands nonzero ids to its hardware start routine, so the host
/// model reports whether capture would start for `id`.
#[inline]
#[must_use]
#[expect(
   clippy::missing_const_for_fn,
   reason = "trace start query stays a runtime decision so const contexts cannot gate capture"
)]
pub fn itc_tracing_start_trace(id: TraceId) -> bool {
   id.value() != 0
}

/// Firmware `DiveItcTracing_EndTrace`.
///
/// Spins until the done register reads nonzero, then acknowledges it with
/// zero. The firmware polls and clears one register, so callers may pass the
/// same register twice.
#[inline]
pub fn itc_tracing_end_trace(done: &MmioU64, ack: &MmioU64) {
   while done.read() == 0 {
      spin_loop();
   }
   ack.write(0);
}
