use core::ptr;

use dive_firmware::{
   debug::{
      system_enter_critical_section,
      system_exit_critical_section,
   },
   runtime::{
      DiveRuntime_AlignedAllocate,
      DiveRuntime_Allocate,
      DiveRuntime_GetSize,
      DiveRuntime_Reallocate,
   },
   types::{
      CacheLineOps,
      FenceGuard,
   },
};

#[test]
fn bump_hands_out_disjoint_regions() {
   let first = DiveRuntime_Allocate(32);
   let second = DiveRuntime_Allocate(32);
   assert!(!first.is_null());
   assert!(!second.is_null());
   let first_addr = first as usize;
   let second_addr = second as usize;
   let first_range = first_addr..first_addr.wrapping_add(32);
   let second_range = second_addr..second_addr.wrapping_add(32);
   assert!(first_range.end <= second_range.start || second_range.end <= first_range.start);
   // SAFETY: `first` is a live 32-byte bump block, so filling it stays in
   // bounds.
   let () = unsafe { ptr::write_bytes(first, 0xA5, 32) };
   // SAFETY: `second` is a live 32-byte bump block, so filling it stays in
   // bounds.
   let () = unsafe { ptr::write_bytes(second, 0x5A, 32) };
   for offset in 0..32_usize {
      // SAFETY: `offset` is below 32, so it stays inside the `first` block.
      let first_elem = unsafe { first.add(offset) };
      // SAFETY: `first_elem` was just computed in bounds of the live block.
      let first_value = unsafe { *first_elem };
      assert_eq!(first_value, 0xA5);
      // SAFETY: `offset` is below 32, so it stays inside the `second` block.
      let second_elem = unsafe { second.add(offset) };
      // SAFETY: `second_elem` was just computed in bounds of the live block.
      let second_value = unsafe { *second_elem };
      assert_eq!(second_value, 0x5A);
   }
}

#[test]
fn reallocate_preserves_prefix_bytes() {
   let block = DiveRuntime_Allocate(16);
   assert!(!block.is_null());
   for offset in 0..16_usize {
      // SAFETY: `offset` is below 16, so it stays inside the block.
      let elem = unsafe { block.add(offset) };
      // SAFETY: `elem` was just computed in bounds of the live block.
      unsafe {
         *elem = offset as u8;
      }
   }
   let grown = DiveRuntime_Reallocate(block, 32);
   assert!(!grown.is_null());
   assert_eq!(DiveRuntime_GetSize(grown), 32);
   for offset in 0..16_usize {
      // SAFETY: `offset` is below 16, so it stays inside the preserved prefix.
      let elem = unsafe { grown.add(offset) };
      // SAFETY: `elem` was just computed in bounds of the live grown block.
      let value = unsafe { *elem };
      assert_eq!(value, offset as u8);
   }
}

#[test]
fn aligned_allocate_honors_alignment() {
   for alignment in [16_usize, 32, 64] {
      let block = DiveRuntime_AlignedAllocate(24, alignment);
      assert!(!block.is_null());
      assert_eq!((block as usize) % alignment, 0);
   }
}

#[test]
fn get_size_reports_requested_bytes() {
   let block = DiveRuntime_Allocate(48);
   assert!(!block.is_null());
   assert_eq!(DiveRuntime_GetSize(block), 48);
}

#[test]
fn cache_line_command_matches_range_encoding() {
   assert_eq!(CacheLineOps::CMD_OR, 0x03_u64 << 56_u32);
   for addr in [0x1010_4000_usize, 0x1010_4040, 0x2000_0000] {
      let line = CacheLineOps::align_down(addr);
      let command = CacheLineOps::line_command(line);
      assert_eq!(command, (line as u64) | (0x03_u64 << 56_u32));
      assert_eq!(command >> 56_u32, 0x03);
   }
}

#[test]
fn cache_full_flush_matches_full_encoding() {
   assert_eq!(CacheLineOps::FULL_FLUSH_CMD, 0x13_u64 << 56_u32);
   assert_ne!(CacheLineOps::FULL_FLUSH_CMD, CacheLineOps::CMD_OR);
   assert_eq!(CacheLineOps::FULL_FLUSH_CMD >> 56_u32, 0x13);
}

#[test]
fn critical_section_nesting_restores_on_outermost_exit() {
   let mut guard = FenceGuard::new();
   system_enter_critical_section(&mut guard);
   system_enter_critical_section(&mut guard);
   assert_eq!(guard.depth(), 2);
   assert!(guard.is_locked());
   system_exit_critical_section(&mut guard);
   assert_eq!(guard.depth(), 1);
   assert!(guard.is_locked());
   system_exit_critical_section(&mut guard);
   assert_eq!(guard.depth(), 0);
   assert!(!guard.is_locked());
}
