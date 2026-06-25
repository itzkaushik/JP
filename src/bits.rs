// =============================================================================
// bits.rs — Bit manipulation primitives for the chess engine
// =============================================================================
//
// Square mapping: bit 0 = a1, bit 63 = h8.
// Rank 1 = bits 0–7, rank 2 = bits 8–15, …, rank 8 = bits 56–63.

// ---------------------------------------------------------------------------
// 1. Core type alias
// ---------------------------------------------------------------------------
pub type Bitboard = u64;

// ---------------------------------------------------------------------------
// Board constants (used from Phase 2 onward)
// ---------------------------------------------------------------------------
pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = 0x8080_8080_8080_8080;
pub const RANK_1: Bitboard = 0x0000_0000_0000_00FF;
pub const RANK_2: Bitboard = 0x0000_0000_0000_FF00;
pub const RANK_7: Bitboard = 0x00FF_0000_0000_0000;
pub const RANK_8: Bitboard = 0xFF00_0000_0000_0000;
pub const LIGHT_SQUARES: Bitboard = 0x55AA_55AA_55AA_55AA;
pub const DARK_SQUARES: Bitboard = 0xAA55_AA55_AA55_AA55;

// ---------------------------------------------------------------------------
// 2. Popcount — count set bits
// ---------------------------------------------------------------------------
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

// ---------------------------------------------------------------------------
// 3. LSB isolation — isolate the lowest set bit
// ---------------------------------------------------------------------------
pub fn lsb(bb: Bitboard) -> Bitboard {
    bb & bb.wrapping_neg()
}

// ---------------------------------------------------------------------------
// 4. LSB index — square index of lowest set bit
// ---------------------------------------------------------------------------
pub fn lsb_index(bb: Bitboard) -> u32 {
    bb.trailing_zeros()
}

// ---------------------------------------------------------------------------
// 5. Pop LSB — remove lowest set bit, return its index
// ---------------------------------------------------------------------------
pub fn pop_lsb(bb: &mut Bitboard) -> u32 {
    let idx = lsb_index(*bb);
    *bb &= *bb - 1;
    idx
}

// ---------------------------------------------------------------------------
// 6. MSB index — square index of highest set bit
// ---------------------------------------------------------------------------
pub fn msb_index(bb: Bitboard) -> u32 {
    63 - bb.leading_zeros()
}

// ---------------------------------------------------------------------------
// 7. Bit set / clear / toggle
// ---------------------------------------------------------------------------
pub fn set_bit(bb: Bitboard, sq: u32) -> Bitboard {
    bb | (1u64 << sq)
}

pub fn clear_bit(bb: Bitboard, sq: u32) -> Bitboard {
    bb & !(1u64 << sq)
}

pub fn toggle_bit(bb: Bitboard, sq: u32) -> Bitboard {
    bb ^ (1u64 << sq)
}

// ---------------------------------------------------------------------------
// 8. Square / coordinate utilities
// ---------------------------------------------------------------------------
pub fn sq(rank: u32, file: u32) -> u32 {
    rank * 8 + file
}

pub fn rank_of(square: u32) -> u32 {
    square / 8
}

pub fn file_of(square: u32) -> u32 {
    square % 8
}

// ---------------------------------------------------------------------------
// 9. Bitboard pretty-printer (for debugging)
// ---------------------------------------------------------------------------
/// Prints an 8×8 grid to stdout.  Rank 8 at the top, rank 1 at the bottom.
/// `1` marks a set bit, `.` marks an empty square.
pub fn print_bb(bb: Bitboard) {
    for rank in (0..8).rev() {
        let mut line = String::new();
        for file in 0..8u32 {
            if file > 0 {
                line.push(' ');
            }
            let square = sq(rank, file);
            if bb & (1u64 << square) != 0 {
                line.push('1');
            } else {
                line.push('.');
            }
        }
        println!("{line}");
    }
}

// ---------------------------------------------------------------------------
// 10. Zobrist hash table
// ---------------------------------------------------------------------------

/// Deterministic splitmix64 PRNG — better distribution than xorshift64.
/// Used exclusively for generating Zobrist keys.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct ZobristTable {
    /// Random keys indexed as pieces[color * 6 + piece_type][square].
    /// color: 0=White, 1=Black.  piece_type: 0=Pawn..5=King.
    pub pieces: [[u64; 64]; 12],
    /// XOR this to flip the side-to-move hash.
    pub side: u64,
    /// Random keys for all 16 castling-rights combinations (4-bit mask).
    pub castling: [u64; 16],
    /// Random keys for en-passant files 0–7.
    pub en_passant: [u64; 8],
}

impl ZobristTable {
    /// Creates a new table filled with deterministic pseudo-random values.
    /// Same seed → same table every time.
    pub fn new() -> Self {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;

        let mut pieces = [[0u64; 64]; 12];
        for piece in 0..12 {
            for square in 0..64 {
                pieces[piece][square] = splitmix64(&mut state);
            }
        }

        let side = splitmix64(&mut state);

        let mut castling = [0u64; 16];
        for entry in &mut castling {
            *entry = splitmix64(&mut state);
        }

        let mut en_passant = [0u64; 8];
        for entry in &mut en_passant {
            *entry = splitmix64(&mut state);
        }

        Self {
            pieces,
            side,
            castling,
            en_passant,
        }
    }

    /// Index into the pieces table for a given (color, piece_type) pair.
    /// color: 0=White, 1=Black.  piece_type: 0=Pawn..5=King.
    #[inline(always)]
    pub fn piece_key(&self, color: usize, piece: usize, sq: usize) -> u64 {
        self.pieces[color * 6 + piece][sq]
    }
}

// Thread-safe global singleton — computed once on first access.
use std::sync::OnceLock;

static ZOBRIST: OnceLock<ZobristTable> = OnceLock::new();

/// Get the global Zobrist key table (lazily initialized, thread-safe).
pub fn zobrist() -> &'static ZobristTable {
    ZOBRIST.get_or_init(ZobristTable::new)
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // popcount
    // -----------------------------------------------------------------------
    #[test]
    fn test_popcount_zero() {
        assert_eq!(popcount(0), 0);
    }

    #[test]
    fn test_popcount_max() {
        assert_eq!(popcount(u64::MAX), 64);
    }

    #[test]
    fn test_popcount_rank2() {
        assert_eq!(popcount(0xFF00), 8);
    }

    // -----------------------------------------------------------------------
    // lsb / lsb_index
    // -----------------------------------------------------------------------
    #[test]
    fn test_lsb_isolation() {
        assert_eq!(lsb(0b1100), 0b0100);
    }

    #[test]
    fn test_lsb_index_basic() {
        assert_eq!(lsb_index(0b1100), 2);
    }

    #[test]
    fn test_lsb_consistency() {
        // lsb_index(lsb(x)) == lsb_index(x) for several values
        for x in [1u64, 0b1100, 0xFF00, 0x8000_0000_0000_0000] {
            assert_eq!(lsb_index(lsb(x)), lsb_index(x));
        }
    }

    // -----------------------------------------------------------------------
    // pop_lsb
    // -----------------------------------------------------------------------
    #[test]
    fn test_pop_lsb_drain() {
        let mut bb: Bitboard = 0b1010;
        let first = pop_lsb(&mut bb);
        assert_eq!(first, 1);
        assert_eq!(bb, 0b1000);

        let second = pop_lsb(&mut bb);
        assert_eq!(second, 3);
        assert_eq!(bb, 0);
    }

    #[test]
    fn test_pop_lsb_loop() {
        let mut bb: Bitboard = 0xFF; // bits 0–7
        let mut indices = Vec::new();
        while bb != 0 {
            indices.push(pop_lsb(&mut bb));
        }
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    // -----------------------------------------------------------------------
    // msb_index
    // -----------------------------------------------------------------------
    #[test]
    fn test_msb_index_basic() {
        assert_eq!(msb_index(0b1010), 3);
    }

    #[test]
    fn test_msb_index_power_of_two() {
        assert_eq!(msb_index(1u64 << 42), 42);
    }

    // -----------------------------------------------------------------------
    // set_bit / clear_bit / toggle_bit
    // -----------------------------------------------------------------------
    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(0, 5), 1u64 << 5);
    }

    #[test]
    fn test_clear_bit() {
        let bb = set_bit(0, 5);
        assert_eq!(clear_bit(bb, 5), 0);
    }

    #[test]
    fn test_set_clear_roundtrip() {
        for sq in 0..64u32 {
            assert_eq!(clear_bit(set_bit(0, sq), sq), 0);
        }
    }

    #[test]
    fn test_toggle_bit() {
        let bb = toggle_bit(0, 10);
        assert_eq!(bb, 1u64 << 10);
        let bb = toggle_bit(bb, 10);
        assert_eq!(bb, 0);
    }

    // -----------------------------------------------------------------------
    // sq / rank_of / file_of
    // -----------------------------------------------------------------------
    #[test]
    fn test_sq_e1() {
        assert_eq!(sq(0, 4), 4); // e1
    }

    #[test]
    fn test_sq_e8() {
        assert_eq!(sq(7, 4), 60); // e8
    }

    #[test]
    fn test_sq_rank_file_roundtrip() {
        for s in 0..64u32 {
            assert_eq!(sq(rank_of(s), file_of(s)), s);
        }
    }

    // -----------------------------------------------------------------------
    // Constants — board perimeter
    // -----------------------------------------------------------------------
    #[test]
    fn test_perimeter() {
        let perimeter = FILE_A | FILE_H | RANK_1 | RANK_8;
        // 8×8 = 64 total squares.  Inner 6×6 = 36 squares.
        // Perimeter = 64 − 36 = 28.
        assert_eq!(popcount(perimeter), 28);
    }

    #[test]
    fn test_light_dark_complement() {
        assert_eq!(LIGHT_SQUARES ^ DARK_SQUARES, u64::MAX);
    }

    // -----------------------------------------------------------------------
    // Zobrist table
    // -----------------------------------------------------------------------
    #[test]
    fn test_zobrist_nonzero() {
        let table = ZobristTable::new();
        assert_ne!(table.pieces[0][0], 0);
        assert_ne!(table.side, 0);
    }

    #[test]
    fn test_zobrist_adjacent_differ() {
        let table = ZobristTable::new();
        assert_ne!(table.pieces[0][0], table.pieces[0][1]);
        assert_ne!(table.pieces[0][0], table.pieces[1][0]);
    }

    #[test]
    fn test_zobrist_deterministic() {
        let t1 = ZobristTable::new();
        let t2 = ZobristTable::new();
        assert_eq!(t1.pieces[5][32], t2.pieces[5][32]);
        assert_eq!(t1.side, t2.side);
    }
}
