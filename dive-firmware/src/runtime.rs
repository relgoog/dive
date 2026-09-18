//! Dive runtime bump allocator plus log and trap shims.
//!
//! Thin wrappers over one bump arena mirror the tiny firmware stubs
//! while host helpers expose formatting through core fmt.

// Firmware ABI keeps DiveRuntime camel names so the case lint stays off here.
#![allow(non_snake_case, reason = "firmware ABI keeps DiveRuntime camel names")]

use alloc::string::String;
use core::{
   cell::UnsafeCell,
   fmt,
   hint::black_box,
   ptr,
};

use dive_abi::{
   dive,
   status::{
      Code,
      Status,
   },
};

use crate::types::AllocatorPtr;

/// Bump arena size in bytes.
const ARENA_BYTES: usize = 0x0001_0000;
/// Header kept below each block holding the requested size.
const HEADER_BYTES: usize = 8;
/// Smallest alignment honored by the bump path.
const MIN_ALIGN: usize = 8;
/// DMA path round and alignment in bytes.
const DMA_ALIGN: usize = 64;
/// Alignment injected by the reallocate wrapper.
const REALLOC_ALIGN: usize = 16;
/// File tag reported with runtime log output.
const FILE: &str = "runtime.cc";
/// File tag reported with allocator trap output.
const ALLOC_FILE: &str = "sequential_allocator.cc";
/// Trap code for a null allocator install matching firmware line 95.
const NULL_ALLOCATOR_CODE: i32 = 95;
/// Trap code for arena exhaustion.
const OUT_OF_MEMORY_CODE: i32 = 90;
/// Trap code for a non power of two alignment.
const BAD_ALIGN_CODE: i32 = 91;
/// Trap code for a null or out of range block pointer.
const BAD_POINTER_CODE: i32 = 92;

/// Bump cursor plus installed allocator cell for the runtime.
struct BumpRuntime {
   /// Backing bytes handed out as disjoint regions.
   arena:   UnsafeCell<[u8; ARENA_BYTES]>,
   /// Next free offset into the arena.
   next:    UnsafeCell<usize>,
   /// Installed allocator address with zero meaning default.
   current: UnsafeCell<usize>,
}

// SAFETY: the bump arena is single hart firmware state with no concurrent
// access.
unsafe impl Sync for BumpRuntime {}

/// Lone bump runtime owning every block handed out here.
static RUNTIME: BumpRuntime = BumpRuntime {
   arena:   UnsafeCell::new([0_u8; ARENA_BYTES]),
   next:    UnsafeCell::new(0),
   current: UnsafeCell::new(0),
};

/// Rounds a value up to a power of two boundary.
#[inline]
#[must_use]
const fn align_up(value: usize, align: usize) -> usize {
   (value.wrapping_add(align.wrapping_sub(1))) & !(align.wrapping_sub(1))
}

/// Normalizes a requested alignment to the bump minimum.
///
/// # Panics
///
/// Ceases when the request is not a power of two.
#[inline]
#[must_use]
fn effective_align(requested: usize) -> usize {
   if requested == 0 {
      return MIN_ALIGN;
   }
   if !requested.is_power_of_two() {
      dive::cease(BAD_ALIGN_CODE, ALLOC_FILE);
   }
   requested.max(MIN_ALIGN)
}

/// Address of the built in bump arena used before any install.
#[inline]
#[must_use]
fn default_allocator() -> AllocatorPtr {
   AllocatorPtr::new(RUNTIME.arena.get() as usize)
}

/// Hands out one region with a size header kept below it.
///
/// # Panics
///
/// Ceases when the arena cannot fit the request.
#[inline]
#[must_use]
#[expect(
   clippy::multiple_unsafe_ops_per_block,
   reason = "bump pointer reads, header store, and cursor store form one allocation step"
)]
fn bump_allocate(requested_size: usize, requested_align: usize) -> *mut u8 {
   let align = effective_align(requested_align);
   // SAFETY: single hart bump state with no concurrent access and every
   // address checked against the arena end before any store.
   unsafe {
      let base = RUNTIME.arena.get().cast::<u8>();
      let base_addr = base as usize;
      let cursor = ptr::read(RUNTIME.next.get());
      let raw_addr = base_addr.wrapping_add(cursor);
      let payload_addr = align_up(raw_addr.wrapping_add(HEADER_BYTES), align);
      let header_addr = payload_addr.wrapping_sub(HEADER_BYTES);
      let end_addr = payload_addr.wrapping_add(requested_size);
      if end_addr > base_addr.wrapping_add(ARENA_BYTES) {
         dive::cease(OUT_OF_MEMORY_CODE, ALLOC_FILE);
      }
      ptr::write_unaligned(header_addr as *mut u64, requested_size as u64);
      ptr::write(RUNTIME.next.get(), end_addr.wrapping_sub(base_addr));
      payload_addr as *mut u8
   }
}

/// Hands out one bump block with minimum alignment.
///
/// # Panics
///
/// Ceases when the arena cannot fit the request.
#[inline]
#[must_use]
pub fn DiveRuntime_Allocate(size: usize) -> *mut u8 {
   bump_allocate(size, MIN_ALIGN)
}

/// Releases a block without reclaim since the bump cursor only grows.
#[inline]
#[expect(
   clippy::missing_const_for_fn,
   reason = "free must stay callable in const-adjacent allocator plumbing without promising const \
             semantics"
)]
pub fn DiveRuntime_Free(block: *mut u8) {
   black_box(block);
}

/// Grows a block by copy with default alignment and keeps prefix bytes.
///
/// # Panics
///
/// Ceases when the old pointer is bad or the arena cannot fit the copy.
#[inline]
#[must_use]
pub fn DiveRuntime_Reallocate(block: *mut u8, new_size: usize) -> *mut u8 {
   if block.is_null() {
      return DiveRuntime_Allocate(new_size);
   }
   let old_size = DiveRuntime_GetSize(block);
   let fresh = bump_allocate(new_size, REALLOC_ALIGN);
   let keep = old_size.min(new_size);
   // SAFETY: both ends come from this arena with headers checked above
   // and the kept range sits inside both blocks.
   unsafe {
      ptr::copy_nonoverlapping(block.cast_const(), fresh, keep);
   };
   DiveRuntime_Free(block);
   fresh
}

/// Hands out one bump block aligned to a power of two boundary.
///
/// # Panics
///
/// Ceases on a bad alignment or when the arena cannot fit the request.
#[inline]
#[must_use]
pub fn DiveRuntime_AlignedAllocate(size: usize, alignment: usize) -> *mut u8 {
   bump_allocate(size, alignment)
}

/// Hands out one 64 byte rounded block for the DMA path.
///
/// # Panics
///
/// Ceases when the arena cannot fit the rounded request.
#[inline]
#[must_use]
pub fn DiveRuntime_DmaAllocate(size: usize) -> *mut u8 {
   let rounded = align_up(size, DMA_ALIGN);
   bump_allocate(rounded, DMA_ALIGN)
}

/// Loads the installed allocator or the built in arena when unset.
#[inline]
pub fn DiveRuntime_GetAllocator() -> AllocatorPtr {
   // SAFETY: one word load from single hart firmware state.
   #[expect(
      clippy::undocumented_unsafe_blocks,
      reason = "SAFETY comment above names the single hart ownership"
   )]
   unsafe {
      let stored = ptr::read(RUNTIME.current.get());
      if stored == 0 {
         return default_allocator();
      }
      AllocatorPtr::new(stored)
   }
}

/// Installs one allocator address and echoes it back.
///
/// # Panics
///
/// Ceases when asked to install the null address.
#[inline]
#[must_use]
pub fn DiveRuntime_SetAllocator(allocator: AllocatorPtr) -> AllocatorPtr {
   if allocator.is_null() {
      DiveRuntime_LogString("Invalid allocator set to nullptr");
      dive::cease(NULL_ALLOCATOR_CODE, FILE);
   }
   // SAFETY: one word store to single hart firmware state.
   unsafe {
      ptr::write(RUNTIME.current.get(), allocator.addr());
   };
   allocator
}

/// Reads the size header kept below one bump block.
///
/// # Panics
///
/// Ceases on a null misaligned or out of range pointer.
#[inline]
#[must_use]
pub fn DiveRuntime_GetSize(block: *mut u8) -> usize {
   if block.is_null() {
      dive::cease(BAD_POINTER_CODE, ALLOC_FILE);
   }
   // SAFETY: range and alignment checked before the header load below.
   unsafe {
      let base_addr = RUNTIME.arena.get() as usize;
      let addr = block as usize;
      let lowest = base_addr.wrapping_add(HEADER_BYTES);
      let limit = base_addr.wrapping_add(ARENA_BYTES);
      if addr < lowest || addr >= limit || !addr.is_multiple_of(MIN_ALIGN) {
         dive::cease(BAD_POINTER_CODE, ALLOC_FILE);
      }
      let header = addr.wrapping_sub(HEADER_BYTES) as *const u64;
      ptr::read_unaligned(header) as usize
   }
}

/// Admits one child dispatch and reports it as unsupported.
///
/// The firmware adapter forwards a fixed context slot into the executor
/// while this host shim has no child table so it logs the request and
/// answers with unimplemented status.
#[inline]
#[must_use]
pub fn DiveRuntime_ExecuteChildModel(child: u32, first_word: usize, second_word: usize) -> Status {
   black_box(first_word.wrapping_add(second_word));
   DiveRuntime_Log(format_args!("ExecuteChildModel {child}"));
   Status::new(Code::Unimplemented)
}

/// Traps with one code and never returns.
///
/// # Panics
///
/// Always ceases through the shared trap path.
#[inline]
pub fn DiveRuntime_Cease(code: i32) -> ! {
   dive::cease(code, FILE);
}

/// Formats core fmt arguments through the runtime log sink.
#[inline]
pub fn DiveRuntime_Log(args: fmt::Arguments<'_>) {
   let mut text = String::new();
   let _result: fmt::Result = fmt::Write::write_fmt(&mut text, args);
   dive::log(&text, FILE);
}

/// Logs one string through the `%s` runtime path.
#[inline]
pub fn DiveRuntime_LogString(message: &str) {
   DiveRuntime_Log(format_args!("{message}"));
}

/// Logs each lane as one indexed line through the runtime path.
#[inline]
pub fn DiveRuntime_LogInt32Array(values: &[i32]) {
   for (index, value) in values.iter().enumerate() {
      DiveRuntime_Log(format_args!("Array[{index}]: {value}\n"));
   }
}
