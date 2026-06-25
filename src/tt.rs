// =============================================================================
// tt.rs — Transposition Table
// =============================================================================
//
// Hash table for caching search results. Uses interior mutability (UnsafeCell)
// for lock-free access — safe in single-threaded mode, benign races in
// multi-threaded mode (worst case: garbled entry fails key check).
//
// 16-byte entries, power-of-2 sizing, bitmask indexing.

use crate::types::Move;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};

// =============================================================================
// Node bound type
// =============================================================================

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum NodeBound {
    Exact = 0, // PV node: score is exact
    Lower = 1, // Cut node: score is a lower bound (failed high / beta cutoff)
    Upper = 2, // All node: score is an upper bound (failed low)
}

impl NodeBound {
    #[inline]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => NodeBound::Exact,
            1 => NodeBound::Lower,
            2 => NodeBound::Upper,
            _ => NodeBound::Upper, // safe fallback
        }
    }
}

// =============================================================================
// TT Entry — 16 bytes
// =============================================================================

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u32,   // upper 32 bits of Zobrist hash
    pub mv: u16,    // best move packed to 16 bits
    pub score: i16, // centipawn score
    pub depth: i8,  // search depth
    pub bound: u8,  // NodeBound as u8
    pub age: u8,    // search generation
    _pad: [u8; 5],  // pad to 16 bytes
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            mv: 0,
            score: 0,
            depth: -128, // sentinel: uninitialized
            bound: 0,
            age: 0,
            _pad: [0; 5],
        }
    }
}

// =============================================================================
// Atomic wrapper for interior mutability
// =============================================================================

#[repr(transparent)]
struct AtomicEntry(UnsafeCell<TTEntry>);

unsafe impl Send for AtomicEntry {}
unsafe impl Sync for AtomicEntry {}

impl AtomicEntry {
    fn new() -> Self {
        AtomicEntry(UnsafeCell::new(TTEntry::default()))
    }

    #[inline]
    fn load(&self) -> TTEntry {
        unsafe { *self.0.get() }
    }

    #[inline]
    fn store(&self, entry: TTEntry) {
        unsafe {
            *self.0.get() = entry;
        }
    }
}

// =============================================================================
// Transposition Table
// =============================================================================

pub struct TranspositionTable {
    entries: Box<[AtomicEntry]>,
    size: usize,   // always power of 2
    age: AtomicU8, // search generation
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Allocate a transposition table of `mb` megabytes.
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let count = (bytes / std::mem::size_of::<TTEntry>()).next_power_of_two();
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(AtomicEntry::new());
        }
        TranspositionTable {
            entries: entries.into_boxed_slice(),
            size: count,
            age: AtomicU8::new(0),
        }
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & (self.size - 1)
    }

    /// Probe the TT for the given Zobrist hash.
    /// Returns Some(entry) if found, None if miss or collision.
    #[inline]
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let entry = self.entries[self.index(hash)].load();
        let key32 = (hash >> 32) as u32;
        if entry.key == key32 && entry.depth != -128 {
            Some(entry)
        } else {
            None
        }
    }

    /// Store a search result in the TT.
    /// Replacement: replace if same position, deeper search, or stale entry.
    #[inline]
    pub fn store(&self, hash: u64, score: i16, mv: Move, depth: i8, bound: NodeBound) {
        let idx = self.index(hash);
        let key32 = (hash >> 32) as u32;
        let old = self.entries[idx].load();
        let age = self.age.load(Ordering::Relaxed);

        // Replacement scheme
        let should_replace = old.depth == -128        // uninitialized
            || old.key == key32                        // same position
            || depth >= old.depth                      // deeper search
            || old.age != age; // stale entry

        if should_replace {
            self.entries[idx].store(TTEntry {
                key: key32,
                mv: mv.pack_16(),
                score,
                depth,
                bound: bound as u8,
                age,
                _pad: [0; 5],
            });
        }
    }

    /// Increment the age counter. Call once per `go` command.
    pub fn advance_age(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    /// TT fill percentage (0–1000). Used for UCI `info hashfull`.
    pub fn hashfull(&self) -> usize {
        let sample = 1000.min(self.size);
        let age = self.age.load(Ordering::Relaxed);
        let used = self.entries[..sample]
            .iter()
            .filter(|e| {
                let entry = e.load();
                entry.depth != -128 && entry.age == age
            })
            .count();
        used * 1000 / sample
    }

    /// Clear all entries. Called on `ucinewgame`.
    pub fn clear(&self) {
        for entry in self.entries.iter() {
            entry.store(TTEntry::default());
        }
    }
}

// =============================================================================
// Mate score adjustment for TT storage
// =============================================================================

use crate::eval::{MATE_VALUE, is_mate_score};

/// Adjust a mate score for TT storage: convert from root-relative to node-relative.
#[inline]
pub fn adjust_mate_for_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_VALUE - 128 {
        score + ply as i32
    } else if score <= -(MATE_VALUE - 128) {
        score - ply as i32
    } else {
        score
    }
}

/// Adjust a mate score retrieved from TT: convert from node-relative to root-relative.
#[inline]
pub fn adjust_mate_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_VALUE - 128 {
        score - ply as i32
    } else if score <= -(MATE_VALUE - 128) {
        score + ply as i32
    } else {
        score
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Move, Square};

    #[test]
    fn test_tt_entry_size() {
        assert_eq!(
            std::mem::size_of::<TTEntry>(),
            16,
            "TTEntry must be 16 bytes"
        );
    }

    #[test]
    fn test_tt_store_probe() {
        let tt = TranspositionTable::new(1); // 1 MB
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let mv = Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH);

        tt.store(hash, 42, mv, 5, NodeBound::Exact);

        let result = tt.probe(hash);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.score, 42);
        assert_eq!(entry.depth, 5);
        assert_eq!(NodeBound::from_u8(entry.bound), NodeBound::Exact);
        assert_eq!(Move::unpack_16(entry.mv), mv);
    }

    #[test]
    fn test_tt_miss() {
        let tt = TranspositionTable::new(1);
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        assert!(tt.probe(hash).is_none());
    }

    #[test]
    fn test_tt_replace_deeper() {
        let tt = TranspositionTable::new(1);
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let mv = Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH);

        tt.store(hash, 10, mv, 3, NodeBound::Lower);
        tt.store(hash, 20, mv, 5, NodeBound::Exact);

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.score, 20);
        assert_eq!(entry.depth, 5);
    }

    #[test]
    fn test_mate_score_adjustment() {
        let mate3 = MATE_VALUE - 3;
        let stored = adjust_mate_for_tt(mate3, 5);
        let retrieved = adjust_mate_from_tt(stored, 5);
        assert_eq!(retrieved, mate3);

        let mated3 = -MATE_VALUE + 3;
        let stored2 = adjust_mate_for_tt(mated3, 5);
        let retrieved2 = adjust_mate_from_tt(stored2, 5);
        assert_eq!(retrieved2, mated3);
    }

    #[test]
    fn test_hashfull() {
        let tt = TranspositionTable::new(1);
        assert_eq!(tt.hashfull(), 0);
    }
}
