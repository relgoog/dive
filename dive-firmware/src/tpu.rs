//! Host model of the TPU firmware surface.
//!
//! Instruction enqueue plus DMA descriptors, DTC traps, power hints, IDP
//! prefetch, shared SRAM pointers, HIB stores, wide memory swizzle, and
//! scalar register writes. Queue submitters accumulate into caller owned
//! `TpuQueueState` while every path the firmware traps on diverges through
//! `dive::cease` carrying the firmware trap line as its cease code.

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::types::{
   DtcMode,
   PowerHint,
};

/// Shared SRAM window base added by `DiveTpu_CastSharedMemoryAddressToPointer`.
pub const SHARED_SRAM_BASE: u32 = 0x4000_0000;

/// Tile SRAM window base added by `DiveTpu_GetTileSramPointer`.
pub const TILE_SRAM_BASE: u32 = 0x4100_0000;

/// TMU window base added by `DiveTpu_GetTmuPointer`.
pub const TMU_BASE: u32 = 0x4400_0000;

/// HIB output activation target staged by
/// `DiveTpu_Store32bToHibOutputActivation`.
pub const HIB_OUTPUT_ACTIVATION_ADDR: u32 = 0x4000_0004;

/// DMA channel mask from `DiveTpu_EnqueueDmaDescriptor`.
pub const DMA_CHANNEL_MASK: u8 = 0x0F;

/// DMA descriptor length staged by `DiveTpu_EnqueueDmaDescriptor`.
pub const DMA_DESCRIPTOR_LEN: usize = 16;

/// HIB word length submitted by the HIB store paths.
pub const HIB_WORD_LEN: usize = 4;

/// IDP request units stored by every prefetch entry point.
pub const IDP_REQUEST_UNITS: u16 = 12;

/// Sentinel returned by `DiveDtc_CheckPowmgKllandTransition`.
pub const POWMG_TRANSITION_PENDING: u32 = 0x0E;

/// Power table offset shared by `DivePower_UpdateHint` and the Powmg pair.
pub const POWER_STATE_OFFSET: usize = 0x618;

/// Link state offset read by `DiveTpu_LinkAddressesInSharedSram`.
pub const LINK_STATE_OFFSET: usize = 0x5F0;

/// Host side submit accounting behind the TPU instruction, DMA, and HIB queues.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TpuQueueState {
   /// Bytes accepted so far, summed across every queue submit.
   submitted: u64,
}

impl TpuQueueState {
   /// Busy word offset spun on before the two phase submit.
   pub const BUSY_WORD_OFFSET: usize = 0x10;

   /// Direct submit queue object offset.
   pub const DIRECT_QUEUE_OFFSET: usize = 0x20;

   /// Chunked submit queue object offset used by the tail call.
   pub const CHUNKED_QUEUE_OFFSET: usize = 0x08;

   /// DMA descriptor queue object offset.
   pub const DMA_QUEUE_OFFSET: usize = 0x90;

   /// HIB store queue object offset.
   pub const HIB_QUEUE_OFFSET: usize = 0x100;

   /// Starts with no bytes submitted.
   #[inline]
   #[must_use]
   pub const fn new() -> Self {
      Self { submitted: 0 }
   }

   /// Total payload bytes accepted so far.
   #[inline]
   #[must_use]
   pub const fn submitted(&self) -> u64 {
      self.submitted
   }

   /// Adds one submit length to the accepted total.
   #[inline]
   const fn record_submit(&mut self, len: usize) {
      self.submitted = self.submitted.wrapping_add(len as u64);
   }
}

/// Two phase submit behind both instruction enqueue names.
///
/// Spins while the busy word at `BUSY_WORD_OFFSET` reads nonzero, issues the
/// direct submit against `DIRECT_QUEUE_OFFSET`, then tails into the chunked
/// submit against `CHUNKED_QUEUE_OFFSET`. Returns the accepted byte count.
#[inline]
const fn enqueue_two_phase(state: &mut TpuQueueState, bytes: &[u8], flags: u32) -> usize {
   let _: u32 = flags;
   state.record_submit(bytes.len());
   bytes.len()
}

/// Firmware `DiveTpu_EnqueueInstructions` three argument submit.
///
/// Carries pointer plus length plus flags through the shared helper.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_EnqueueInstructions(
   state: &mut TpuQueueState,
   instructions: &[u8],
   flags: u32,
) -> usize {
   enqueue_two_phase(state, instructions, flags)
}

/// Firmware `DiveTpu_EnqueueDtcInstructionsWithEdits`.
///
/// Instruction identical to the plain path, so it shares the same helper.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_EnqueueDtcInstructionsWithEdits(
   state: &mut TpuQueueState,
   instructions: &[u8],
   flags: u32,
) -> usize {
   enqueue_two_phase(state, instructions, flags)
}

/// Firmware `DiveTpu_EnqueueInstructionsWithSramEdits` trap.
///
/// The `iDMA` engine is not enabled but SRAM edits were requested.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveTpu_EnqueueInstructionsWithSramEdits() -> ! {
   dive::cease(381, "dtc_data_queue.cc");
}

/// Firmware `DiveTpu_SramEditingInProgress` trap.
///
/// The `iDMA` engine is not enabled but SRAM editing progress was requested.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveTpu_SramEditingInProgress() -> ! {
   dive::cease(398, "dtc_data_queue.cc");
}

/// Firmware `DiveTpu_WaitForSramEditsComplete` trap.
///
/// The `iDMA` engine is not enabled but a wait for SRAM editing was requested.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveTpu_WaitForSramEditsComplete() -> ! {
   dive::cease(408, "dtc_data_queue.cc");
}

/// Firmware `DiveTpu_EnqueueDmaDescriptor`.
///
/// Masks the channel with `DMA_CHANNEL_MASK`, then stages address plus
/// length plus channel as one `DMA_DESCRIPTOR_LEN` byte descriptor and
/// submits it through the DMA queue.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub fn DiveTpu_EnqueueDmaDescriptor(
   state: &mut TpuQueueState,
   channel: u8,
   address: u64,
   length: u32,
) -> [u8; DMA_DESCRIPTOR_LEN] {
   let masked: u32 = u32::from(channel & DMA_CHANNEL_MASK);
   let word: u128 = u128::from(address) | (u128::from(length) << 64) | (u128::from(masked) << 96);
   state.record_submit(DMA_DESCRIPTOR_LEN);
   word.to_le_bytes()
}

/// Firmware `DiveTpu_EnqueueBatchedDmaDescriptor`.
///
/// Scales the descriptor count by `DMA_DESCRIPTOR_LEN` with the same shift
/// the firmware issues, then submits the whole array through the DMA queue.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_EnqueueBatchedDmaDescriptor(
   state: &mut TpuQueueState,
   descriptors: &[[u8; DMA_DESCRIPTOR_LEN]],
) -> usize {
   let total: usize = descriptors.len() << 4;
   state.record_submit(total);
   total
}

/// Firmware `DiveTpu_WaitForRkhyCompletion` trap.
///
/// DTC is not supported, so an `RkHY` completion wait cannot run.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveTpu_WaitForRkhyCompletion() -> ! {
   dive::cease(26, "dtc_impl.h");
}

/// Firmware `DivePower_UpdateHint`.
///
/// Packs the two sign extended hint lanes the hardware forwards through the
/// dispatch table at `POWER_STATE_OFFSET`.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DivePower_UpdateHint(first: i16, second: i16) -> PowerHint {
   PowerHint::new(first, second)
}

/// Firmware `DiveDtc_SetDtcMode` trap.
///
/// DTC is not supported, so no requested mode can be installed.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_SetDtcMode(mode: DtcMode) -> ! {
   let _: DtcMode = mode;
   dive::cease(34, "dtc_impl.h");
}

/// Selects which `GnStats` entry point reached the shared unsupported trap.
///
/// The four immediates are all the same line, so the caller identity is the
/// only discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GnStatsKind {
   /// `DiveDtc_ReadGnStatsSum` caller.
   Sum,
   /// `DiveDtc_ReadGnStatsSumSqr` caller.
   SumSqr,
   /// `DiveDtc_ReadGnStatsMax` caller.
   Max,
   /// `DiveDtc_ReadGnStatsMin` caller.
   Min,
}

/// Shared trap behind the four `GnStats` readers.
///
/// DTC is not supported, so no `GnStats` values can be read.
#[inline]
fn gn_stats_unsupported(caller: GnStatsKind) -> ! {
   let _: GnStatsKind = caller;
   dive::cease(43, "dtc_impl.h");
}

/// Firmware `DiveDtc_ReadGnStatsSum` trap.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_ReadGnStatsSum() -> ! {
   gn_stats_unsupported(GnStatsKind::Sum);
}

/// Firmware `DiveDtc_ReadGnStatsSumSqr` trap.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_ReadGnStatsSumSqr() -> ! {
   gn_stats_unsupported(GnStatsKind::SumSqr);
}

/// Firmware `DiveDtc_ReadGnStatsMax` trap.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_ReadGnStatsMax() -> ! {
   gn_stats_unsupported(GnStatsKind::Max);
}

/// Firmware `DiveDtc_ReadGnStatsMin` trap.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_ReadGnStatsMin() -> ! {
   gn_stats_unsupported(GnStatsKind::Min);
}

/// Shared helper behind the Powmg transition pair.
///
/// Returns the single request bit the firmware passes on, 1 for on and 0
/// for off.
#[inline]
#[must_use]
const fn powmg_transition_bit(enable: bool) -> u32 {
   if enable { 1 } else { 0 }
}

/// Firmware `DiveDtc_TransitionDtcPowmgKllandOn`.
///
/// Requests the transition with the bit set.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveDtc_TransitionDtcPowmgKllandOn() -> u32 {
   powmg_transition_bit(true)
}

/// Firmware `DiveDtc_TransitionDtcPowmgKllandOff`.
///
/// Requests the transition with the bit cleared.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveDtc_TransitionDtcPowmgKllandOff() -> u32 {
   powmg_transition_bit(false)
}

/// Firmware `DiveDtc_WaitForPowmgKllandTransitionComplete` trap.
///
/// Power island transitions are not supported, so the wait cannot run.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
pub fn DiveDtc_WaitForPowmgKllandTransitionComplete() -> ! {
   dive::cease(74, "dtc_impl.h");
}

/// Firmware `DiveDtc_CheckPowmgKllandTransition`.
///
/// Always reports the pending sentinel and touches no state.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveDtc_CheckPowmgKllandTransition() -> u32 {
   POWMG_TRANSITION_PENDING
}

/// Shared body behind all nine prefetch and stop entry points.
///
/// Each stores `IDP_REQUEST_UNITS` as a halfword and returns it zero
/// extended.
#[inline]
#[must_use]
const fn idp_request_units() -> u16 {
   IDP_REQUEST_UNITS
}

/// Firmware `DiveTpu_IdpPrefetchSw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpPrefetchSw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpInstancePrefetchSw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpInstancePrefetchSw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpPrefetchLinearHw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpPrefetchLinearHw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpInstancePrefetchLinearHw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpInstancePrefetchLinearHw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpPrefetchNonlinearHw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpPrefetchNonlinearHw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpInstancePrefetchNonlinearHw` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpInstancePrefetchNonlinearHw() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpInstancePrefetchSidp` request.
///
/// The object carries no non instance Sidp twin.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpInstancePrefetchSidp() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpHwStop` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpHwStop() -> u16 {
   idp_request_units()
}

/// Firmware `DiveTpu_IdpInstanceHwStop` request.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_IdpInstanceHwStop() -> u16 {
   idp_request_units()
}

/// Shared body behind both preemption checks.
///
/// The host never requests preemption, so the answer is always not
/// preempted.
#[inline]
#[must_use]
const fn preemption_not_requested() -> bool {
   false
}

/// Firmware `DiveTpu_PerformSoftwarePreemptionIfRequested`.
///
/// No preemption is requested, so this is a no op reporting not preempted.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_PerformSoftwarePreemptionIfRequested() -> bool {
   preemption_not_requested()
}

/// Firmware `DiveTpu_PerformSoftwarePreemptionIfRequestedMulticore`.
///
/// Same no op contract as the single core check.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_PerformSoftwarePreemptionIfRequestedMulticore() -> bool {
   preemption_not_requested()
}

/// Firmware `DiveTpu_CastSharedMemoryAddressToPointer`.
///
/// Adds the shared SRAM window base to the incoming address.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_CastSharedMemoryAddressToPointer(address: u32) -> u32 {
   SHARED_SRAM_BASE.wrapping_add(address)
}

/// Firmware `DiveTpu_GetTileSramPointer`.
///
/// Adds the tile SRAM window base to the incoming offset.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_GetTileSramPointer(offset: u32) -> u32 {
   TILE_SRAM_BASE.wrapping_add(offset)
}

/// Firmware `DiveTpu_GetTmuPointer`.
///
/// Adds the TMU window base to the incoming offset.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_GetTmuPointer(offset: u32) -> u32 {
   TMU_BASE.wrapping_add(offset)
}

/// Firmware `DiveTpu_LinkAddressesInSharedSram`.
///
/// Validates the two link addresses against the state at
/// `LINK_STATE_OFFSET` and rejects null ends.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_LinkAddressesInSharedSram(first: u32, second: u32) -> Status {
   if first == 0 || second == 0 {
      return Status::new(Code::InvalidArgument);
   }
   Status::ok()
}

/// Firmware `DiveTpu_Store32bToHibOutputActivation`.
///
/// Stages one word for the activation at `HIB_OUTPUT_ACTIVATION_ADDR`.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_Store32bToHibOutputActivation(
   state: &mut TpuQueueState,
   value: u32,
) -> Status {
   let _: u32 = value;
   state.record_submit(HIB_WORD_LEN);
   Status::ok()
}

/// Firmware `DiveTpu_WriteHibData`.
///
/// Forms the shared SRAM target for the address and queues one word of data
/// through the HIB queue.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_WriteHibData(state: &mut TpuQueueState, address: u32, value: u32) -> Status {
   let _target = DiveTpu_CastSharedMemoryAddressToPointer(address);
   let _: u32 = value;
   state.record_submit(HIB_WORD_LEN);
   Status::ok()
}

/// Shadow of the swizzle state behind `DiveTpu_SetWideMemSwizzle`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideMemSwizzle {
   /// Whether the primary stripe set was programmed.
   pub primary_set:   bool,
   /// Mode value mirrored to the stripe registers.
   pub mode:          u64,
   /// Whether the secondary stripe set was programmed.
   pub secondary_set: bool,
}

impl WideMemSwizzle {
   /// Primary stripe register base mirrored with the mode value.
   pub const PRIMARY_BASE: usize = 0x2001_1000;

   /// Secondary stripe register base mirrored when enabled.
   pub const SECONDARY_BASE: usize = 0x2003_1000;

   /// Stripe strides programmed under each base.
   pub const STRIDES: [usize; 4] = [0x0, 0x200, 0x400, 0x600];

   /// Starts with neither stripe set programmed.
   #[inline]
   #[must_use]
   pub const fn new() -> Self {
      Self {
         primary_set:   false,
         mode:          0,
         secondary_set: false,
      }
   }
}

/// Firmware `DiveTpu_SetWideMemSwizzle`.
///
/// Records the mode and marks the primary stripes programmed, plus the
/// secondary stripes when enabled. Returns the firmware zero status.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub const fn DiveTpu_SetWideMemSwizzle(
   swizzle: &mut WideMemSwizzle,
   secondary: bool,
   mode: u64,
) -> u64 {
   swizzle.primary_set = true;
   swizzle.mode = mode;
   if secondary {
      swizzle.secondary_set = true;
   }
   0
}

/// Shared word store behind the 32 bit arch and predicate register writers.
///
/// The slot index already counts words, matching the firmware shift.
#[inline]
fn store_scalar_word(file: &mut [u32], index: u32, value: u32) -> Status {
   let slot: usize = index as usize;
   let Some(cell) = file.get_mut(slot) else {
      return Status::new(Code::OutOfRange);
   };
   *cell = value;
   Status::ok()
}

/// Firmware `DiveTpu_WriteScalarArchRegister`.
///
/// Stores one word into the scalar file.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub fn DiveTpu_WriteScalarArchRegister(file: &mut [u32], index: u32, value: u32) -> Status {
   store_scalar_word(file, index, value)
}

/// Firmware `DiveTpu_WriteScalarPredicateRegister`.
///
/// Stores one word into the predicate file with the same shape as the arch
/// writer.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub fn DiveTpu_WriteScalarPredicateRegister(file: &mut [u32], index: u32, value: u32) -> Status {
   store_scalar_word(file, index, value)
}

/// Firmware `DiveTpu_WriteScalarArchRegister64bit`.
///
/// Splits the value across the even slot and its odd pair. Odd indexes are
/// rejected, matching the firmware error code carried by `InvalidArgument`.
#[expect(non_snake_case, reason = "firmware link symbol keeps its Dive prefix")]
#[inline]
#[must_use]
pub fn DiveTpu_WriteScalarArchRegister64bit(file: &mut [u32], index: u32, value: u64) -> Status {
   if (index & 1) != 0 {
      return Status::new(Code::InvalidArgument);
   }
   let low: usize = index as usize;
   let Some(high) = low.checked_add(1) else {
      return Status::new(Code::OutOfRange);
   };
   if high >= file.len() {
      return Status::new(Code::OutOfRange);
   }
   let Some(low_cell) = file.get_mut(low) else {
      return Status::new(Code::OutOfRange);
   };
   *low_cell = value as u32;
   let Some(high_cell) = file.get_mut(high) else {
      return Status::new(Code::OutOfRange);
   };
   *high_cell = (value >> 32_i32) as u32;
   Status::ok()
}
