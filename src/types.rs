// =============================================================================
// types.rs — Foundational newtypes for the chess engine
// =============================================================================
//
// All core types use the newtype pattern for compile-time type safety.
// Square mapping: LERF (Little-Endian Rank-File) — bit 0 = a1, bit 63 = h8.

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

// =============================================================================
// BitBoard
// =============================================================================

/// A 64-bit bitboard where each bit corresponds to a square on the chess board.
/// Bit 0 = a1, bit 7 = h1, bit 56 = a8, bit 63 = h8 (LERF mapping).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub const EMPTY: BitBoard = BitBoard(0);
    pub const ALL: BitBoard = BitBoard(u64::MAX);

    // File masks
    pub const FILE_A: BitBoard = BitBoard(0x0101_0101_0101_0101);
    pub const FILE_B: BitBoard = BitBoard(0x0202_0202_0202_0202);
    pub const FILE_C: BitBoard = BitBoard(0x0404_0404_0404_0404);
    pub const FILE_D: BitBoard = BitBoard(0x0808_0808_0808_0808);
    pub const FILE_E: BitBoard = BitBoard(0x1010_1010_1010_1010);
    pub const FILE_F: BitBoard = BitBoard(0x2020_2020_2020_2020);
    pub const FILE_G: BitBoard = BitBoard(0x4040_4040_4040_4040);
    pub const FILE_H: BitBoard = BitBoard(0x8080_8080_8080_8080);

    // Rank masks
    pub const RANK_1: BitBoard = BitBoard(0x0000_0000_0000_00FF);
    pub const RANK_2: BitBoard = BitBoard(0x0000_0000_0000_FF00);
    pub const RANK_3: BitBoard = BitBoard(0x0000_0000_00FF_0000);
    pub const RANK_4: BitBoard = BitBoard(0x0000_0000_FF00_0000);
    pub const RANK_5: BitBoard = BitBoard(0x0000_00FF_0000_0000);
    pub const RANK_6: BitBoard = BitBoard(0x0000_FF00_0000_0000);
    pub const RANK_7: BitBoard = BitBoard(0x00FF_0000_0000_0000);
    pub const RANK_8: BitBoard = BitBoard(0xFF00_0000_0000_0000);

    /// Returns true if no bits are set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true if any bit is set.
    #[inline(always)]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Population count — number of set bits.
    #[inline(always)]
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Index of the least significant set bit. Undefined if empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Remove and return the least significant set bit.
    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }

    /// Check if a specific square's bit is set.
    #[inline(always)]
    pub const fn has(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0 as u64) != 0
    }

    /// Set a specific square's bit.
    #[inline(always)]
    pub const fn set(self, sq: Square) -> BitBoard {
        BitBoard(self.0 | (1u64 << sq.0 as u64))
    }

    /// Clear a specific square's bit.
    #[inline(always)]
    pub const fn clear(self, sq: Square) -> BitBoard {
        BitBoard(self.0 & !(1u64 << sq.0 as u64))
    }

    /// Toggle a specific square's bit.
    #[inline(always)]
    pub const fn toggle(self, sq: Square) -> BitBoard {
        BitBoard(self.0 ^ (1u64 << sq.0 as u64))
    }

    /// Returns true if exactly one bit is set.
    #[inline(always)]
    pub const fn is_single(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) == 0
    }

    /// Returns true if more than one bit is set.
    #[inline(always)]
    pub const fn has_multiple(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) != 0
    }
}

// --- Operator overloads ---

impl BitAnd for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        BitBoard(self.0 & rhs.0)
    }
}

impl BitAndAssign for BitBoard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        BitBoard(self.0 | rhs.0)
    }
}

impl BitOrAssign for BitBoard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        BitBoard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for BitBoard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        BitBoard(!self.0)
    }
}

impl Shl<u32> for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn shl(self, n: u32) -> Self {
        BitBoard(self.0 << n)
    }
}

impl Shr<u32> for BitBoard {
    type Output = Self;
    #[inline(always)]
    fn shr(self, n: u32) -> Self {
        BitBoard(self.0 >> n)
    }
}

// --- Iterator: traverse set bits ---

impl Iterator for BitBoard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pop_lsb())
    }
}

// --- Debug: print as 8×8 grid ---

impl fmt::Debug for BitBoard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                write!(f, "{} ", if self.0 >> sq & 1 == 1 { "1" } else { "." })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl fmt::Display for BitBoard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// =============================================================================
// Square
// =============================================================================

/// A square index in 0..64 using LERF mapping.
/// 0 = a1, 7 = h1, 56 = a8, 63 = h8.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(pub u8);

impl Square {
    /// Create a square from file (0=a..7=h) and rank (0=1..7=8).
    #[inline(always)]
    pub const fn new(file: u8, rank: u8) -> Self {
        Square(rank * 8 + file)
    }

    /// File index (0=a, 7=h).
    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Rank index (0=rank 1, 7=rank 8).
    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Convert to a BitBoard with just this square set.
    #[inline(always)]
    pub const fn bit(self) -> BitBoard {
        BitBoard(1u64 << self.0 as u64)
    }

    /// Parse algebraic notation like "e4" into a Square.
    pub fn from_algebraic(s: &str) -> Option<Square> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }

    /// Convert to algebraic notation like "e4".
    pub fn to_algebraic(self) -> String {
        let file_char = (b'a' + self.file()) as char;
        let rank_char = (b'1' + self.rank()) as char;
        format!("{}{}", file_char, rank_char)
    }
}

// Named square constants
#[allow(dead_code)]
impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);

    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);

    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);

    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);

    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 < 64 {
            write!(f, "{}", self.to_algebraic())
        } else {
            write!(f, "Square(INVALID:{})", self.0)
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// =============================================================================
// Color
// =============================================================================

/// Side to move.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Return the opposite color.
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Array index (0 or 1).
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

// =============================================================================
// Piece
// =============================================================================

/// Chess piece type (without color).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    /// All six piece types, for iteration.
    pub const ALL: [Piece; 6] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];

    /// Array index (0..6).
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Character for this piece in the given color (uppercase=White, lowercase=Black).
    pub fn to_char(self, color: Color) -> char {
        let ch = match self {
            Piece::Pawn => 'P',
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
        };
        match color {
            Color::White => ch,
            Color::Black => ch.to_ascii_lowercase(),
        }
    }

    /// Parse a FEN piece character (e.g. 'N' → (White, Knight), 'p' → (Black, Pawn)).
    pub fn from_char(ch: char) -> Option<(Color, Piece)> {
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match ch.to_ascii_uppercase() {
            'P' => Piece::Pawn,
            'N' => Piece::Knight,
            'B' => Piece::Bishop,
            'R' => Piece::Rook,
            'Q' => Piece::Queen,
            'K' => Piece::King,
            _ => return None,
        };
        Some((color, piece))
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ch = match self {
            Piece::Pawn => "Pawn",
            Piece::Knight => "Knight",
            Piece::Bishop => "Bishop",
            Piece::Rook => "Rook",
            Piece::Queen => "Queen",
            Piece::King => "King",
        };
        write!(f, "{}", ch)
    }
}

// =============================================================================
// CastlingRights
// =============================================================================

/// Castling rights packed into 4 bits.
/// Bit 0 = White kingside, bit 1 = White queenside,
/// bit 2 = Black kingside, bit 3 = Black queenside.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct CastlingRights(pub u8);

#[allow(dead_code)]
impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);

    pub const WK: u8 = 0b0001;
    pub const WQ: u8 = 0b0010;
    pub const BK: u8 = 0b0100;
    pub const BQ: u8 = 0b1000;
    pub const ALL: u8 = 0b1111;

    pub const WHITE_BOTH: u8 = Self::WK | Self::WQ;
    pub const BLACK_BOTH: u8 = Self::BK | Self::BQ;

    /// Check if a specific castling right is available.
    #[inline(always)]
    pub const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Remove a castling right.
    #[inline(always)]
    pub fn remove(&mut self, flag: u8) {
        self.0 &= !flag;
    }

    /// Add a castling right.
    #[inline(always)]
    pub fn add(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Parse from FEN castling string (e.g. "KQkq", "Kq", "-").
    pub fn from_fen(s: &str) -> CastlingRights {
        if s == "-" {
            return CastlingRights::NONE;
        }
        let mut rights = CastlingRights::NONE;
        for ch in s.chars() {
            match ch {
                'K' => rights.add(Self::WK),
                'Q' => rights.add(Self::WQ),
                'k' => rights.add(Self::BK),
                'q' => rights.add(Self::BQ),
                _ => {} // ignore invalid chars
            }
        }
        rights
    }

    /// Serialize to FEN castling string.
    pub fn to_fen(self) -> String {
        if self.0 == 0 {
            return "-".to_string();
        }
        let mut s = String::with_capacity(4);
        if self.has(Self::WK) {
            s.push('K');
        }
        if self.has(Self::WQ) {
            s.push('Q');
        }
        if self.has(Self::BK) {
            s.push('k');
        }
        if self.has(Self::BQ) {
            s.push('q');
        }
        s
    }
}

impl fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CastlingRights({})", self.to_fen())
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_fen())
    }
}

// =============================================================================
// Move
// =============================================================================

/// A chess move packed into 32 bits.
///
/// Layout:
/// - bits  0..5:  from square (0–63)
/// - bits  6..11: to square (0–63)
/// - bits 12..15: flags
/// - bits 16..31: reserved (move ordering score — Phase 4)
///
/// Flag encoding:
/// 0b0000 = quiet move
/// 0b0001 = double pawn push
/// 0b0010 = king-side castle
/// 0b0011 = queen-side castle
/// 0b0100 = capture
/// 0b0101 = en passant capture
/// 0b1000 = knight promotion
/// 0b1001 = bishop promotion
/// 0b1010 = rook promotion
/// 0b1011 = queen promotion
/// 0b1100 = knight promo-capture
/// 0b1101 = bishop promo-capture
/// 0b1110 = rook promo-capture
/// 0b1111 = queen promo-capture
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Move(pub u32);

#[allow(dead_code)]
impl Move {
    // Flag constants
    pub const FLAG_QUIET: u32 = 0b0000;
    pub const FLAG_DOUBLE_PUSH: u32 = 0b0001;
    pub const FLAG_KS_CASTLE: u32 = 0b0010;
    pub const FLAG_QS_CASTLE: u32 = 0b0011;
    pub const FLAG_CAPTURE: u32 = 0b0100;
    pub const FLAG_EP_CAPTURE: u32 = 0b0101;
    pub const FLAG_PROMO_N: u32 = 0b1000;
    pub const FLAG_PROMO_B: u32 = 0b1001;
    pub const FLAG_PROMO_R: u32 = 0b1010;
    pub const FLAG_PROMO_Q: u32 = 0b1011;
    pub const FLAG_PROMO_CAP_N: u32 = 0b1100;
    pub const FLAG_PROMO_CAP_B: u32 = 0b1101;
    pub const FLAG_PROMO_CAP_R: u32 = 0b1110;
    pub const FLAG_PROMO_CAP_Q: u32 = 0b1111;

    pub const NULL: Move = Move(0);

    /// Create a move from components.
    #[inline(always)]
    pub const fn new(from: Square, to: Square, flags: u32) -> Move {
        Move((from.0 as u32) | ((to.0 as u32) << 6) | (flags << 12))
    }

    /// Source square.
    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    /// Destination square.
    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    /// Move flags (4 bits).
    #[inline(always)]
    pub const fn flags(self) -> u32 {
        (self.0 >> 12) & 0xF
    }

    /// Is this a capture (regular, en passant, or promotion capture)?
    /// Uses single-bit check: flag bit 2 (= move bit 14) is set for all captures.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.0 & (1 << 14) != 0
    }

    /// Is this a promotion (with or without capture)?
    /// Uses single-bit check: flag bit 3 (= move bit 15) is set for all promotions.
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.0 & (1 << 15) != 0
    }

    /// Is this a castling move?
    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        let f = self.flags();
        f == Self::FLAG_KS_CASTLE || f == Self::FLAG_QS_CASTLE
    }

    /// Is this an en passant capture?
    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flags() == Self::FLAG_EP_CAPTURE
    }

    /// Is this a double pawn push?
    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flags() == Self::FLAG_DOUBLE_PUSH
    }

    /// For promotion moves, return the promotion piece type.
    pub const fn promotion_piece(self) -> Option<Piece> {
        match self.flags() {
            Self::FLAG_PROMO_N | Self::FLAG_PROMO_CAP_N => Some(Piece::Knight),
            Self::FLAG_PROMO_B | Self::FLAG_PROMO_CAP_B => Some(Piece::Bishop),
            Self::FLAG_PROMO_R | Self::FLAG_PROMO_CAP_R => Some(Piece::Rook),
            Self::FLAG_PROMO_Q | Self::FLAG_PROMO_CAP_Q => Some(Piece::Queen),
            _ => None,
        }
    }

    /// UCI-style string representation (e.g. "e2e4", "e7e8q").
    pub fn to_uci(self) -> String {
        let from = self.from_sq().to_algebraic();
        let to = self.to_sq().to_algebraic();
        let promo = match self.promotion_piece() {
            Some(Piece::Knight) => "n",
            Some(Piece::Bishop) => "b",
            Some(Piece::Rook) => "r",
            Some(Piece::Queen) => "q",
            _ => "",
        };
        format!("{}{}{}", from, to, promo)
    }

    /// Pack into 16 bits for TT storage (from/to/flags fits in 16 bits).
    #[inline(always)]
    pub const fn pack_16(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// Unpack from 16-bit TT storage.
    #[inline(always)]
    pub const fn unpack_16(raw: u16) -> Self {
        Move(raw as u32)
    }
}

// =============================================================================
// MoveList — stack-allocated, zero-heap move storage
// =============================================================================

/// Fixed-capacity move list. No heap allocation.
/// Maximum legal moves in any chess position is 218; 256 gives headroom.
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

#[allow(dead_code)]
impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList {
            moves: [Move::NULL; 256],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256, "MoveList overflow");
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = Move;
    fn index(&self, i: usize) -> &Move {
        &self.moves[i]
    }
}

impl std::ops::IndexMut<usize> for MoveList {
    fn index_mut(&mut self, i: usize) -> &mut Move {
        &mut self.moves[i]
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Move({})", self.to_uci())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // --- BitBoard tests ---

    #[test]
    fn test_bitboard_operators() {
        let a = BitBoard(0xFF00);
        let b = BitBoard(0x00FF);
        assert_eq!((a | b), BitBoard(0xFFFF));
        assert_eq!((a & b), BitBoard::EMPTY);
        assert_eq!((a ^ b), BitBoard(0xFFFF));
        assert_eq!((!BitBoard::EMPTY), BitBoard::ALL);
    }

    #[test]
    fn test_bitboard_shifts() {
        let bb = BitBoard(1);
        assert_eq!((bb << 8), BitBoard(256)); // a1 -> a2
        assert_eq!((BitBoard(256) >> 8), bb); // a2 -> a1
    }

    #[test]
    fn test_bitboard_popcount() {
        assert_eq!(BitBoard::EMPTY.popcount(), 0);
        assert_eq!(BitBoard::ALL.popcount(), 64);
        assert_eq!(BitBoard::RANK_1.popcount(), 8);
    }

    #[test]
    fn test_bitboard_has_set_clear() {
        let bb = BitBoard::EMPTY.set(Square::E4);
        assert!(bb.has(Square::E4));
        assert!(!bb.has(Square::D4));
        let bb2 = bb.clear(Square::E4);
        assert!(!bb2.has(Square::E4));
    }

    #[test]
    fn test_bitboard_iterator() {
        let bb = BitBoard(0b1010);
        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares, vec![Square(1), Square(3)]);
    }

    #[test]
    fn test_bitboard_is_single() {
        assert!(BitBoard(1).is_single());
        assert!(BitBoard(1 << 32).is_single());
        assert!(!BitBoard(0).is_single());
        assert!(!BitBoard(3).is_single());
    }

    // --- Square tests ---

    #[test]
    fn test_square_new() {
        assert_eq!(Square::new(0, 0), Square::A1);
        assert_eq!(Square::new(4, 0), Square::E1);
        assert_eq!(Square::new(7, 7), Square::H8);
    }

    #[test]
    fn test_square_file_rank() {
        assert_eq!(Square::E1.file(), 4);
        assert_eq!(Square::E1.rank(), 0);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::H8.rank(), 7);
    }

    #[test]
    fn test_square_roundtrip() {
        for sq_idx in 0..64u8 {
            let sq = Square(sq_idx);
            assert_eq!(Square::new(sq.file(), sq.rank()), sq);
        }
    }

    #[test]
    fn test_square_bit() {
        assert_eq!(Square::A1.bit(), BitBoard(1));
        assert_eq!(Square::H8.bit(), BitBoard(1u64 << 63));
    }

    #[test]
    fn test_square_algebraic() {
        assert_eq!(Square::A1.to_algebraic(), "a1");
        assert_eq!(Square::E4.to_algebraic(), "e4");
        assert_eq!(Square::H8.to_algebraic(), "h8");
    }

    #[test]
    fn test_square_from_algebraic() {
        assert_eq!(Square::from_algebraic("a1"), Some(Square::A1));
        assert_eq!(Square::from_algebraic("e4"), Some(Square(28)));
        assert_eq!(Square::from_algebraic("h8"), Some(Square::H8));
        assert_eq!(Square::from_algebraic("i1"), None);
        assert_eq!(Square::from_algebraic("a9"), None);
    }

    // --- Color tests ---

    #[test]
    fn test_color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }

    #[test]
    fn test_color_index() {
        assert_eq!(Color::White.index(), 0);
        assert_eq!(Color::Black.index(), 1);
    }

    // --- Piece tests ---

    #[test]
    fn test_piece_from_char() {
        assert_eq!(Piece::from_char('K'), Some((Color::White, Piece::King)));
        assert_eq!(Piece::from_char('p'), Some((Color::Black, Piece::Pawn)));
        assert_eq!(Piece::from_char('N'), Some((Color::White, Piece::Knight)));
        assert_eq!(Piece::from_char('x'), None);
    }

    #[test]
    fn test_piece_to_char() {
        assert_eq!(Piece::King.to_char(Color::White), 'K');
        assert_eq!(Piece::King.to_char(Color::Black), 'k');
        assert_eq!(Piece::Pawn.to_char(Color::White), 'P');
    }

    // --- CastlingRights tests ---

    #[test]
    fn test_castling_from_fen() {
        assert_eq!(CastlingRights::from_fen("KQkq").0, CastlingRights::ALL);
        assert_eq!(CastlingRights::from_fen("-").0, 0);
        assert_eq!(
            CastlingRights::from_fen("Kq").0,
            CastlingRights::WK | CastlingRights::BQ
        );
    }

    #[test]
    fn test_castling_to_fen() {
        assert_eq!(CastlingRights(CastlingRights::ALL).to_fen(), "KQkq");
        assert_eq!(CastlingRights::NONE.to_fen(), "-");
        assert_eq!(
            CastlingRights(CastlingRights::WK | CastlingRights::BQ).to_fen(),
            "Kq"
        );
    }

    #[test]
    fn test_castling_roundtrip() {
        for bits in 0..16u8 {
            let cr = CastlingRights(bits);
            let fen = cr.to_fen();
            let cr2 = CastlingRights::from_fen(&fen);
            assert_eq!(cr, cr2);
        }
    }

    // --- Move tests ---

    #[test]
    fn test_move_encode_decode() {
        let mv = Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH);
        assert_eq!(mv.from_sq(), Square::E2);
        assert_eq!(mv.to_sq(), Square::E4);
        assert_eq!(mv.flags(), Move::FLAG_DOUBLE_PUSH);
        assert!(mv.is_double_push());
        assert!(!mv.is_capture());
    }

    #[test]
    fn test_move_promotion() {
        let mv = Move::new(Square::E7, Square::E8, Move::FLAG_PROMO_Q);
        assert!(mv.is_promotion());
        assert_eq!(mv.promotion_piece(), Some(Piece::Queen));
    }

    #[test]
    fn test_move_uci() {
        let mv = Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH);
        assert_eq!(mv.to_uci(), "e2e4");

        let mv2 = Move::new(Square::E7, Square::E8, Move::FLAG_PROMO_Q);
        assert_eq!(mv2.to_uci(), "e7e8q");
    }

    #[test]
    fn test_move_castle_flags() {
        let ks = Move::new(Square::E1, Square::G1, Move::FLAG_KS_CASTLE);
        assert!(ks.is_castle());
        assert!(!ks.is_capture());

        let qs = Move::new(Square::E1, Square::C1, Move::FLAG_QS_CASTLE);
        assert!(qs.is_castle());
    }

    // --- MoveList tests ---

    #[test]
    fn test_movelist_push_and_len() {
        let mut list = MoveList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        list.push(Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH));
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list[0].from_sq(), Square::E2);
    }

    #[test]
    fn test_movelist_as_slice() {
        let mut list = MoveList::new();
        list.push(Move::new(Square::E2, Square::E4, Move::FLAG_DOUBLE_PUSH));
        list.push(Move::new(Square::D2, Square::D4, Move::FLAG_DOUBLE_PUSH));
        let slice = list.as_slice();
        assert_eq!(slice.len(), 2);
    }
}
