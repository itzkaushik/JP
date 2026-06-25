// =============================================================================
// attacks.rs — Attack tables for all piece types
// =============================================================================
//
// Non-sliding pieces (knight, king, pawn): compile-time computed via const fn.
// Sliding pieces (rook, bishop, queen): magic bitboard lookup, initialized at
// runtime via OnceLock (deterministic, ~50ms init time, ~840KB tables).

use crate::types::{BitBoard, Color, Square};
use std::sync::OnceLock;

// =============================================================================
// File masks as raw u64 for const fn usage
// =============================================================================

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = 0x0202_0202_0202_0202;
const FILE_G: u64 = 0x4040_4040_4040_4040;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const FILE_AB: u64 = FILE_A | FILE_B;
const FILE_GH: u64 = FILE_G | FILE_H;

// =============================================================================
// Knight attacks (compile-time)
// =============================================================================

const fn compute_knight_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        table[sq] = ((bb << 17) & !FILE_A)
            | ((bb << 15) & !FILE_H)
            | ((bb << 10) & !FILE_AB)
            | ((bb << 6) & !FILE_GH)
            | ((bb >> 17) & !FILE_H)
            | ((bb >> 15) & !FILE_A)
            | ((bb >> 10) & !FILE_GH)
            | ((bb >> 6) & !FILE_AB);
        sq += 1;
    }
    table
}

static KNIGHT_ATTACKS_RAW: [u64; 64] = compute_knight_attacks();

/// Get the attack bitboard for a knight on the given square.
#[inline(always)]
pub fn knight_attacks(sq: Square) -> BitBoard {
    BitBoard(KNIGHT_ATTACKS_RAW[sq.0 as usize])
}

// =============================================================================
// King attacks (compile-time)
// =============================================================================

const fn compute_king_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        table[sq] = ((bb << 1) & !FILE_A)
            | ((bb >> 1) & !FILE_H)
            | (bb << 8)
            | (bb >> 8)
            | ((bb << 9) & !FILE_A)
            | ((bb << 7) & !FILE_H)
            | ((bb >> 9) & !FILE_H)
            | ((bb >> 7) & !FILE_A);
        sq += 1;
    }
    table
}

static KING_ATTACKS_RAW: [u64; 64] = compute_king_attacks();

/// Get the attack bitboard for a king on the given square.
#[inline(always)]
pub fn king_attacks(sq: Square) -> BitBoard {
    BitBoard(KING_ATTACKS_RAW[sq.0 as usize])
}

// =============================================================================
// Pawn attacks (compile-time)
// =============================================================================

const fn compute_pawn_attacks() -> [[u64; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        // White pawn attacks: up-left and up-right
        table[0][sq] = ((bb << 7) & !FILE_H) | ((bb << 9) & !FILE_A);
        // Black pawn attacks: down-left and down-right
        table[1][sq] = ((bb >> 7) & !FILE_A) | ((bb >> 9) & !FILE_H);
        sq += 1;
    }
    table
}

static PAWN_ATTACKS_RAW: [[u64; 64]; 2] = compute_pawn_attacks();

/// Get the attack bitboard for a pawn of the given color on the given square.
/// This returns the squares the pawn *attacks* (diagonal captures), not pushes.
#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> BitBoard {
    BitBoard(PAWN_ATTACKS_RAW[color as usize][sq.0 as usize])
}

// =============================================================================
// Magic Bitboards — Sliding piece attack tables
// =============================================================================

/// One entry per square for magic bitboard lookup.
#[derive(Clone, Copy)]
struct MagicEntry {
    mask: u64,     // relevant occupancy mask (edges excluded)
    magic: u64,    // magic multiplier
    shift: u32,    // right-shift amount = 64 - popcount(mask)
    offset: usize, // index into the flat attack table
}

impl Default for MagicEntry {
    fn default() -> Self {
        MagicEntry {
            mask: 0,
            magic: 0,
            shift: 64,
            offset: 0,
        }
    }
}

/// All magic bitboard data. Initialized once, shared globally.
struct MagicTables {
    rook_magics: [MagicEntry; 64],
    bishop_magics: [MagicEntry; 64],
    rook_table: Vec<u64>,
    bishop_table: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Relevant occupancy mask: which squares can block a sliding piece's path?
// Edges of the board are excluded because a blocker on the edge does not
// change the attack pattern (the ray ends there anyway).
// ---------------------------------------------------------------------------

fn relevant_occupancy_mask(sq: usize, is_rook: bool) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut mask = 0u64;

    if is_rook {
        // North (rank increases, exclude rank 7 = board edge)
        let mut r = rank + 1;
        while r < 7 {
            mask |= 1u64 << (r * 8 + file);
            r += 1;
        }
        // South (exclude rank 0)
        r = rank - 1;
        while r > 0 {
            mask |= 1u64 << (r * 8 + file);
            r -= 1;
        }
        // East (exclude file 7)
        let mut f = file + 1;
        while f < 7 {
            mask |= 1u64 << (rank * 8 + f);
            f += 1;
        }
        // West (exclude file 0)
        f = file - 1;
        while f > 0 {
            mask |= 1u64 << (rank * 8 + f);
            f -= 1;
        }
    } else {
        // NE
        let (mut r, mut f) = (rank + 1, file + 1);
        while r < 7 && f < 7 {
            mask |= 1u64 << (r * 8 + f);
            r += 1;
            f += 1;
        }
        // NW
        let (mut r, mut f) = (rank + 1, file - 1);
        while r < 7 && f > 0 {
            mask |= 1u64 << (r * 8 + f);
            r += 1;
            f -= 1;
        }
        // SE
        let (mut r, mut f) = (rank - 1, file + 1);
        while r > 0 && f < 7 {
            mask |= 1u64 << (r * 8 + f);
            r -= 1;
            f += 1;
        }
        // SW
        let (mut r, mut f) = (rank - 1, file - 1);
        while r > 0 && f > 0 {
            mask |= 1u64 << (r * 8 + f);
            r -= 1;
            f -= 1;
        }
    }

    mask
}

// ---------------------------------------------------------------------------
// Slow sliding attacks: reference implementation used to build tables.
// Scans rays direction-by-direction, stopping at the first blocker (inclusive).
// ---------------------------------------------------------------------------

fn slow_sliding_attacks(sq: usize, occupied: u64, is_rook: bool) -> u64 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut attacks = 0u64;

    let directions: &[(i32, i32)] = if is_rook {
        &[(1, 0), (-1, 0), (0, 1), (0, -1)] // N, S, E, W
    } else {
        &[(1, 1), (1, -1), (-1, 1), (-1, -1)] // NE, NW, SE, SW
    };

    for &(dr, df) in directions {
        let mut r = rank + dr;
        let mut f = file + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let sq_idx = (r * 8 + f) as u64;
            attacks |= 1u64 << sq_idx;
            if occupied & (1u64 << sq_idx) != 0 {
                break; // blocker found — include it (can be captured) but stop
            }
            r += dr;
            f += df;
        }
    }

    attacks
}

// ---------------------------------------------------------------------------
// PRNG for magic number search — xorshift64 producing sparse candidates
// ---------------------------------------------------------------------------

fn xorshift_magic(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn sparse_random(state: &mut u64) -> u64 {
    xorshift_magic(state) & xorshift_magic(state) & xorshift_magic(state)
}

// ---------------------------------------------------------------------------
// Find a magic number for one square via random search
// ---------------------------------------------------------------------------

fn find_magic(
    sq: usize,
    is_rook: bool,
    mask: u64,
    bits: u32,
    occupancies: &[u64],
    attacks: &[u64],
) -> u64 {
    let n = 1usize << bits;
    let shift = 64 - bits;

    // Deterministic seed per (square, piece_type)
    let mut rng_state = (sq as u64)
        .wrapping_mul(if is_rook { 728_361 } else { 391_457 })
        .wrapping_add(0xDEAD_BEEF_CAFE_1234);

    let mut used = vec![u64::MAX; n];

    loop {
        let candidate = sparse_random(&mut rng_state);

        // Quick rejection: the multiply must spread bits into the top
        if (mask.wrapping_mul(candidate) >> 56).count_ones() < 6 {
            continue;
        }

        // Test for collisions
        let mut fail = false;
        for entry in used.iter_mut().take(n) {
            *entry = u64::MAX; // sentinel
        }

        for i in 0..occupancies.len() {
            let idx = (occupancies[i].wrapping_mul(candidate) >> shift) as usize;
            if used[idx] == u64::MAX {
                used[idx] = attacks[i];
            } else if used[idx] != attacks[i] {
                fail = true;
                break;
            }
        }

        if !fail {
            return candidate;
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize all magic tables for one piece type (rook or bishop)
// ---------------------------------------------------------------------------

fn init_magics_for_piece(is_rook: bool) -> ([MagicEntry; 64], Vec<u64>) {
    let mut entries = [MagicEntry::default(); 64];
    let mut table: Vec<u64> = Vec::new();

    for sq in 0..64usize {
        let mask = relevant_occupancy_mask(sq, is_rook);
        let bits = mask.count_ones();
        let shift = 64 - bits;
        let n = 1usize << bits;

        // Enumerate all subsets of the mask (Carry-Rippler trick)
        let mut occupancies = Vec::with_capacity(n);
        let mut attacks = Vec::with_capacity(n);

        let mut subset = 0u64;
        loop {
            occupancies.push(subset);
            attacks.push(slow_sliding_attacks(sq, subset, is_rook));
            subset = subset.wrapping_sub(mask) & mask;
            if subset == 0 {
                break;
            }
        }

        // Find magic number
        let magic = find_magic(sq, is_rook, mask, bits, &occupancies, &attacks);

        // Record offset and allocate sub-table
        let offset = table.len();
        table.resize(offset + n, 0u64);

        // Fill sub-table
        for i in 0..occupancies.len() {
            let idx = (occupancies[i].wrapping_mul(magic) >> shift) as usize;
            table[offset + idx] = attacks[i];
        }

        entries[sq] = MagicEntry {
            mask,
            magic,
            shift,
            offset,
        };
    }

    (entries, table)
}

impl MagicTables {
    fn init() -> Self {
        let (rook_magics, rook_table) = init_magics_for_piece(true);
        let (bishop_magics, bishop_table) = init_magics_for_piece(false);
        MagicTables {
            rook_magics,
            bishop_magics,
            rook_table,
            bishop_table,
        }
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static MAGIC: OnceLock<MagicTables> = OnceLock::new();

fn magic_tables() -> &'static MagicTables {
    MAGIC.get_or_init(MagicTables::init)
}

/// Force initialization of magic tables. Call early in main() to avoid
/// paying the init cost during the first search.
pub fn init() {
    let _ = magic_tables();
}

// ---------------------------------------------------------------------------
// Public sliding attack functions
// ---------------------------------------------------------------------------

/// Rook attacks from `sq` given the current `occupied` bitboard.
#[inline(always)]
pub fn rook_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    let t = magic_tables();
    let e = &t.rook_magics[sq.0 as usize];
    let idx = ((occupied.0 & e.mask).wrapping_mul(e.magic)) >> e.shift;
    BitBoard(t.rook_table[e.offset + idx as usize])
}

/// Bishop attacks from `sq` given the current `occupied` bitboard.
#[inline(always)]
pub fn bishop_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    let t = magic_tables();
    let e = &t.bishop_magics[sq.0 as usize];
    let idx = ((occupied.0 & e.mask).wrapping_mul(e.magic)) >> e.shift;
    BitBoard(t.bishop_table[e.offset + idx as usize])
}

/// Queen attacks = rook attacks | bishop attacks.
#[inline(always)]
pub fn queen_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    rook_attacks(sq, occupied) | bishop_attacks(sq, occupied)
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // --- Knight attack tests ---

    #[test]
    fn test_knight_attacks_center() {
        let attacks = knight_attacks(Square(28));
        assert_eq!(attacks.popcount(), 8);
    }

    #[test]
    fn test_knight_attacks_corner() {
        let attacks = knight_attacks(Square::A1);
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.has(Square(10))); // c2
        assert!(attacks.has(Square(17))); // b3
    }

    #[test]
    fn test_knight_attacks_edge() {
        let attacks = knight_attacks(Square(24));
        assert_eq!(attacks.popcount(), 4);
    }

    #[test]
    fn test_knight_attacks_b1() {
        let attacks = knight_attacks(Square::B1);
        assert_eq!(attacks.popcount(), 3);
        assert!(attacks.has(Square(16))); // a3
        assert!(attacks.has(Square(18))); // c3
        assert!(attacks.has(Square(11))); // d2
    }

    // --- King attack tests ---

    #[test]
    fn test_king_attacks_center() {
        let attacks = king_attacks(Square(28));
        assert_eq!(attacks.popcount(), 8);
    }

    #[test]
    fn test_king_attacks_corner() {
        let attacks = king_attacks(Square::A1);
        assert_eq!(attacks.popcount(), 3);
        assert!(attacks.has(Square(1)));
        assert!(attacks.has(Square(8)));
        assert!(attacks.has(Square(9)));
    }

    #[test]
    fn test_king_attacks_edge() {
        let attacks = king_attacks(Square(24));
        assert_eq!(attacks.popcount(), 5);
    }

    // --- Pawn attack tests ---

    #[test]
    fn test_white_pawn_attacks_center() {
        let attacks = pawn_attacks(Color::White, Square(28));
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.has(Square(35)));
        assert!(attacks.has(Square(37)));
    }

    #[test]
    fn test_black_pawn_attacks_center() {
        let attacks = pawn_attacks(Color::Black, Square(36));
        assert_eq!(attacks.popcount(), 2);
        assert!(attacks.has(Square(27)));
        assert!(attacks.has(Square(29)));
    }

    #[test]
    fn test_white_pawn_attacks_a_file() {
        let attacks = pawn_attacks(Color::White, Square(8));
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.has(Square(17)));
    }

    #[test]
    fn test_white_pawn_attacks_h_file() {
        let attacks = pawn_attacks(Color::White, Square(15));
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.has(Square(22)));
    }

    #[test]
    fn test_black_pawn_attacks_a_file() {
        let attacks = pawn_attacks(Color::Black, Square(48));
        assert_eq!(attacks.popcount(), 1);
        assert!(attacks.has(Square(41)));
    }

    // --- Sliding piece attack tests ---

    #[test]
    fn test_rook_attacks_empty_board() {
        // Rook on e4 (28) on empty board attacks 14 squares (7 rank + 7 file)
        let attacks = rook_attacks(Square(28), BitBoard::EMPTY);
        assert_eq!(attacks.popcount(), 14);
    }

    #[test]
    fn test_rook_attacks_with_blocker() {
        // Rook on a1, blocker on a4 → attacks a2, a3, a4 (stops), b1..h1
        let occ = BitBoard(1u64 << 24); // a4
        let attacks = rook_attacks(Square::A1, occ);
        assert!(attacks.has(Square(8))); // a2
        assert!(attacks.has(Square(16))); // a3
        assert!(attacks.has(Square(24))); // a4 (blocker, included)
        assert!(!attacks.has(Square(32))); // a5 (blocked)
        assert!(attacks.has(Square(1))); // b1
        assert!(attacks.has(Square(7))); // h1
    }

    #[test]
    fn test_bishop_attacks_empty_board() {
        // Bishop on e4 (28) on empty board: 13 diagonal squares
        let attacks = bishop_attacks(Square(28), BitBoard::EMPTY);
        assert_eq!(attacks.popcount(), 13);
    }

    #[test]
    fn test_bishop_attacks_with_blocker() {
        // Bishop on e4, blocker on g6 → attacks f5, g6 (stops)
        let occ = BitBoard(1u64 << 46); // g6
        let attacks = bishop_attacks(Square(28), occ);
        assert!(attacks.has(Square(37))); // f5
        assert!(attacks.has(Square(46))); // g6 (blocker, included)
        assert!(!attacks.has(Square(55))); // h7 (blocked)
    }

    #[test]
    fn test_queen_attacks_empty_board() {
        // Queen on e4 = rook(14) + bishop(13) = 27
        let attacks = queen_attacks(Square(28), BitBoard::EMPTY);
        assert_eq!(attacks.popcount(), 27);
    }

    #[test]
    fn test_rook_attacks_corner_a1() {
        let attacks = rook_attacks(Square::A1, BitBoard::EMPTY);
        assert_eq!(attacks.popcount(), 14);
    }

    #[test]
    fn test_bishop_attacks_corner_a1() {
        // Bishop on a1 empty board: 7 squares on the a1-h8 diagonal
        let attacks = bishop_attacks(Square::A1, BitBoard::EMPTY);
        assert_eq!(attacks.popcount(), 7);
    }

    #[test]
    fn test_sliding_attacks_match_slow_reference() {
        // Verify magic lookup matches slow reference for all squares on empty board
        for sq_idx in 0..64u8 {
            let sq = Square(sq_idx);
            let slow_r = slow_sliding_attacks(sq_idx as usize, 0, true);
            let fast_r = rook_attacks(sq, BitBoard::EMPTY);
            assert_eq!(
                fast_r.0, slow_r,
                "Rook mismatch at sq {} (empty board)",
                sq_idx
            );

            let slow_b = slow_sliding_attacks(sq_idx as usize, 0, false);
            let fast_b = bishop_attacks(sq, BitBoard::EMPTY);
            assert_eq!(
                fast_b.0, slow_b,
                "Bishop mismatch at sq {} (empty board)",
                sq_idx
            );
        }
    }

    #[test]
    fn test_sliding_attacks_match_slow_with_blockers() {
        // Test a few positions with various blocker configurations
        let blockers = [
            0x0000_0010_0000_0000u64, // e5
            0x0000_0000_FF00_0000u64, // rank 4 full
            0x0101_0101_0101_0101u64, // a-file full
            0x8040_2010_0804_0201u64, // main diagonal
        ];
        for &occ in &blockers {
            for sq_idx in 0..64u8 {
                let sq = Square(sq_idx);
                let slow_r = slow_sliding_attacks(sq_idx as usize, occ, true);
                let fast_r = rook_attacks(sq, BitBoard(occ));
                assert_eq!(
                    fast_r.0, slow_r,
                    "Rook mismatch at sq {} with occ {:#018x}",
                    sq_idx, occ
                );

                let slow_b = slow_sliding_attacks(sq_idx as usize, occ, false);
                let fast_b = bishop_attacks(sq, BitBoard(occ));
                assert_eq!(
                    fast_b.0, slow_b,
                    "Bishop mismatch at sq {} with occ {:#018x}",
                    sq_idx, occ
                );
            }
        }
    }
}
