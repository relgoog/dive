use dive_firmware::{
   debug::{
      FENCE_FLAG_OFFSET,
      TraceRing,
      debug_wait_for_fence_completion_and_clear_memory,
      tpu_wait_for_fence_completion_multicore,
   },
   tpu::{
      DMA_CHANNEL_MASK,
      DMA_DESCRIPTOR_LEN,
      DiveDtc_CheckPowmgKllandTransition,
      DiveTpu_CastSharedMemoryAddressToPointer,
      DiveTpu_EnqueueBatchedDmaDescriptor,
      DiveTpu_EnqueueDmaDescriptor,
      DiveTpu_EnqueueDtcInstructionsWithEdits,
      DiveTpu_EnqueueInstructions,
      DiveTpu_GetTileSramPointer,
      DiveTpu_GetTmuPointer,
      DiveTpu_IdpHwStop,
      DiveTpu_IdpInstanceHwStop,
      DiveTpu_IdpInstancePrefetchLinearHw,
      DiveTpu_IdpInstancePrefetchNonlinearHw,
      DiveTpu_IdpInstancePrefetchSidp,
      DiveTpu_IdpInstancePrefetchSw,
      DiveTpu_IdpPrefetchLinearHw,
      DiveTpu_IdpPrefetchNonlinearHw,
      DiveTpu_IdpPrefetchSw,
      DiveTpu_SetWideMemSwizzle,
      IDP_REQUEST_UNITS,
      POWMG_TRANSITION_PENDING,
      SHARED_SRAM_BASE,
      TILE_SRAM_BASE,
      TMU_BASE,
      TpuQueueState,
      WideMemSwizzle,
   },
   types::{
      FenceGuard,
      TraceId,
   },
};

#[test]
fn enqueue_plain_and_dtc_share_behavior() {
   let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
   let mut plain = TpuQueueState::new();
   let mut edited = TpuQueueState::new();
   let plain_count = DiveTpu_EnqueueInstructions(&mut plain, &bytes, 0);
   let edited_count = DiveTpu_EnqueueDtcInstructionsWithEdits(&mut edited, &bytes, 0);
   assert_eq!(plain_count, bytes.len());
   assert_eq!(edited_count, plain_count);
   assert_eq!(plain.submitted(), edited.submitted());
   assert_eq!(plain.submitted(), bytes.len() as u64);
}

#[test]
fn dma_descriptor_masks_channel_to_low_nibble() {
   assert_eq!(DMA_CHANNEL_MASK, 0x0F);
   let mut state = TpuQueueState::new();
   let wide = DiveTpu_EnqueueDmaDescriptor(&mut state, 0xFF, 0x1000, 64);
   let narrow = DiveTpu_EnqueueDmaDescriptor(&mut state, 0x0F, 0x1000, 64);
   assert_eq!(wide, narrow);
   let word = u128::from_le_bytes(wide);
   assert_eq!(((word >> 96_u32) & 0xFF) as u8, 0x0F);
   assert_eq!(state.submitted(), (2 * DMA_DESCRIPTOR_LEN) as u64);
}

#[test]
fn batched_descriptor_count_scales_by_sixteen() {
   let mut state = TpuQueueState::new();
   let descriptors = [[0x11_u8; DMA_DESCRIPTOR_LEN]; 3];
   let total = DiveTpu_EnqueueBatchedDmaDescriptor(&mut state, &descriptors);
   assert_eq!(total, 3 * 16);
   assert_eq!(total, descriptors.len() << 4_u32);
   assert_eq!(state.submitted(), total as u64);
}

#[test]
fn check_powmg_transition_always_reports_pending() {
   assert_eq!(POWMG_TRANSITION_PENDING, 14);
   assert_eq!(DiveDtc_CheckPowmgKllandTransition(), 14);
   assert_eq!(
      DiveDtc_CheckPowmgKllandTransition(),
      POWMG_TRANSITION_PENDING
   );
}

#[test]
fn all_nine_prefetch_entries_return_twelve_units() {
   assert_eq!(IDP_REQUEST_UNITS, 12);
   assert_eq!(DiveTpu_IdpPrefetchSw(), 12);
   assert_eq!(DiveTpu_IdpInstancePrefetchSw(), 12);
   assert_eq!(DiveTpu_IdpPrefetchLinearHw(), 12);
   assert_eq!(DiveTpu_IdpInstancePrefetchLinearHw(), 12);
   assert_eq!(DiveTpu_IdpPrefetchNonlinearHw(), 12);
   assert_eq!(DiveTpu_IdpInstancePrefetchNonlinearHw(), 12);
   assert_eq!(DiveTpu_IdpInstancePrefetchSidp(), 12);
   assert_eq!(DiveTpu_IdpHwStop(), 12);
   assert_eq!(DiveTpu_IdpInstanceHwStop(), 12);
}

#[test]
fn pointer_helpers_add_window_bases() {
   assert_eq!(SHARED_SRAM_BASE, 0x4000_0000);
   assert_eq!(TILE_SRAM_BASE, 0x4100_0000);
   assert_eq!(TMU_BASE, 0x4400_0000);
   assert_eq!(DiveTpu_CastSharedMemoryAddressToPointer(0x100), 0x4000_0100);
   assert_eq!(DiveTpu_GetTileSramPointer(0x100), 0x4100_0100);
   assert_eq!(DiveTpu_GetTmuPointer(0x100), 0x4400_0100);
}

#[test]
fn swizzle_mirrors_mode_with_secondary_gate() {
   let mut primary_only = WideMemSwizzle::new();
   assert_eq!(DiveTpu_SetWideMemSwizzle(&mut primary_only, false, 0x2A), 0);
   assert!(primary_only.primary_set);
   assert_eq!(primary_only.mode, 0x2A);
   assert!(!primary_only.secondary_set);

   let mut both = WideMemSwizzle::new();
   assert_eq!(DiveTpu_SetWideMemSwizzle(&mut both, true, 0x2A), 0);
   assert!(both.primary_set);
   assert_eq!(both.mode, 0x2A);
   assert!(both.secondary_set);
}

#[test]
fn fence_poll_helper_clears_completion_flag() {
   let mut fence_state = vec![0_u8; FENCE_FLAG_OFFSET + 1];
   let base = fence_state.as_mut_ptr();
   // SAFETY: `base` spans `FENCE_FLAG_OFFSET + 1` live bytes, so the flag
   // offset is in bounds.
   let flag_addr = unsafe { base.byte_add(FENCE_FLAG_OFFSET) };
   // SAFETY: `flag_addr` was just computed in bounds of the live fence state.
   let () = unsafe { *flag_addr = 1 };
   let mut guard = FenceGuard::new();
   // SAFETY: `base` points at live fence state with the flag byte set, so the
   // poll returns.
   let () = unsafe { debug_wait_for_fence_completion_and_clear_memory(base, &mut guard) };
   // SAFETY: `base` still spans the live fence state, so rereading the flag is
   // in bounds.
   let flag_after = unsafe { base.byte_add(FENCE_FLAG_OFFSET) };
   // SAFETY: `flag_after` was just computed in bounds of the live fence state.
   let flag_value = unsafe { *flag_after };
   assert_eq!(flag_value, 0);
   assert_eq!(guard.depth(), 0);

   let mut idle_guard = FenceGuard::new();
   // SAFETY: `base` points at live fence state and the nonzero token skips the
   // poll.
   let passthrough = unsafe { tpu_wait_for_fence_completion_multicore(base, &mut idle_guard, 7) };
   assert_eq!(passthrough, 7);
}

#[test]
fn trace_ring_appends_behind_enable_gate() {
   let mut ring = TraceRing::new();
   assert!(!ring.is_enabled());
   assert!(!ring.start_trace(TraceId::new(1), 100));
   assert!(ring.is_empty());

   ring.set_enabled(true);
   assert!(ring.is_enabled());
   assert!(ring.start_trace(TraceId::new(1), 100));
   assert!(ring.start_operation_trace(TraceId::new(2), 200));
   assert_eq!(ring.len(), 2);

   assert!(ring.end_trace());
   assert_eq!(ring.len(), 1);
   assert!(ring.end_trace());
   assert!(ring.is_empty());
   assert!(!ring.end_trace());

   ring.set_enabled(false);
   assert!(!ring.start_trace(TraceId::new(3), 300));
   assert!(ring.is_empty());
}
