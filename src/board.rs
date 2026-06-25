// =============================================================================
// board.rs — The Board struct: complete chess position representation
// =============================================================================
//
// 12 bitboards (one per color × piece type) + supplementary mailbox for O(1)
// piece-at-square lookup. Includes FEN parsing/serialization, Zobrist hashing,
// and human-readable display.
//
// Copy-make approach: `make_move` returns a new Board (no undo stack needed).

use crate::attacks;
use crate::bits::zobrist;
use crate::types::{BitBoard, CastlingRights, Color, Move, Piece, Square};
use std::fmt;

// =============================================================================
// FEN error type
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    MissingField(&'static str),
    InvalidPiecePlacement(String),
    InvalidColor(String),
    InvalidSquare(String),
    InvalidNumber(String),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FenError::MissingField(field) => write!(f, "Missing FEN field: {}", field),
            FenError::InvalidPiecePlacement(msg) => write!(f, "Invalid piece placement: {}", msg),
            FenError::InvalidColor(msg) => write!(f, "Invalid color: {}", msg),
            FenError::InvalidSquare(msg) => write!(f, "Invalid square: {}", msg),
            FenError::InvalidNumber(msg) => write!(f, "Invalid number: {}", msg),
        }
    }
}

// =============================================================================
// Board
// =============================================================================

/// Complete chess position. This is the single most important struct in the engine.
///
/// Layout:
/// - `pieces[color][piece_type]` — 12 bitboards (96 bytes, fits ~2 cache lines)
/// - `by_color[color]` — aggregated occupancy per side (derived, kept in sync)
/// - `occupied` — all pieces combined (derived)
/// - `mailbox[sq]` — O(1) piece-at-square lookup (64 bytes)
/// - Game state: side to move, castling, en passant, clocks
/// - `hash` — Zobrist hash, updated incrementally on every mutation
#[derive(Clone, Copy)]
pub struct Board {
    // --- Piece data (96 bytes) ---
    pub pieces: [[BitBoard; 6]; 2], // [color][piece_type]

    // --- Derived (keep updated incrementally) ---
    pub occupied: BitBoard,      // all pieces
    pub by_color: [BitBoard; 2], // [White occupancy, Black occupancy]

    // --- Supplementary mailbox: O(1) piece-at-square lookup ---
    pub mailbox: [Option<(Color, Piece)>; 64],

    // --- Game state ---
    pub side_to_move: Color,
    pub castling_rights: CastlingRights,
    pub en_passant_sq: Option<Square>,
    pub halfmove_clock: u8, // 50-move rule counter (0–100)
    pub fullmove_number: u16,

    // --- Zobrist hash ---
    pub hash: u64,
}

// =============================================================================
// Core construction and piece manipulation
// =============================================================================

impl Board {
    /// An empty board with no pieces, White to move, no castling, no en passant.
    pub fn empty() -> Self {
        Board {
            pieces: [[BitBoard::EMPTY; 6]; 2],
            occupied: BitBoard::EMPTY,
            by_color: [BitBoard::EMPTY; 2],
            mailbox: [None; 64],
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant_sq: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
        }
    }

    /// The standard starting position.
    pub fn start_pos() -> Self {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Starting position FEN is always valid")
    }

    /// Place a piece on the board. Updates bitboards, mailbox, and Zobrist hash.
    #[inline]
    pub fn set_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = sq.bit();
        self.pieces[color.index()][piece.index()] |= bb;
        self.by_color[color.index()] |= bb;
        self.occupied |= bb;
        self.mailbox[sq.0 as usize] = Some((color, piece));
        self.hash ^= zobrist().piece_key(color.index(), piece.index(), sq.0 as usize);
    }

    /// Remove a piece from the board. Updates bitboards, mailbox, and Zobrist hash.
    #[inline]
    pub fn remove_piece(&mut self, color: Color, piece: Piece, sq: Square) {
        let bb = sq.bit();
        self.pieces[color.index()][piece.index()] ^= bb;
        self.by_color[color.index()] ^= bb;
        self.occupied ^= bb;
        self.mailbox[sq.0 as usize] = None;
        self.hash ^= zobrist().piece_key(color.index(), piece.index(), sq.0 as usize);
    }

    /// Move a piece from one square to another. Does NOT handle captures, castling,
    /// en passant, or promotions — use `make_move` for that.
    #[inline]
    pub fn move_piece(&mut self, color: Color, piece: Piece, from: Square, to: Square) {
        self.remove_piece(color, piece, from);
        self.set_piece(color, piece, to);
    }

    /// Look up what piece (if any) is on a given square. O(1) via mailbox.
    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, Piece)> {
        self.mailbox[sq.0 as usize]
    }

    /// Get the bitboard for a specific (color, piece_type) pair.
    #[inline(always)]
    pub fn bb(&self, color: Color, piece: Piece) -> BitBoard {
        self.pieces[color.index()][piece.index()]
    }

    /// Recompute `occupied` and `by_color` from the 12 piece bitboards.
    /// Used after FEN parsing to ensure consistency.
    pub fn recompute_occupancy(&mut self) {
        self.by_color[0] = BitBoard::EMPTY;
        self.by_color[1] = BitBoard::EMPTY;
        for piece in Piece::ALL {
            self.by_color[0] |= self.pieces[0][piece.index()];
            self.by_color[1] |= self.pieces[1][piece.index()];
        }
        self.occupied = self.by_color[0] | self.by_color[1];
    }

    /// Compute the Zobrist hash from scratch (full recomputation).
    /// Used after FEN parsing and for debugging hash consistency.
    pub fn recompute_hash(&mut self) {
        let z = zobrist();
        let mut hash = 0u64;

        // Piece-square keys
        for color in [Color::White, Color::Black] {
            for piece in Piece::ALL {
                let mut bb = self.pieces[color.index()][piece.index()];
                while bb.is_not_empty() {
                    let sq = bb.pop_lsb();
                    hash ^= z.piece_key(color.index(), piece.index(), sq.0 as usize);
                }
            }
        }

        // Side to move
        if self.side_to_move == Color::Black {
            hash ^= z.side;
        }

        // Castling rights
        hash ^= z.castling[self.castling_rights.0 as usize];

        // En passant file
        if let Some(ep_sq) = self.en_passant_sq {
            hash ^= z.en_passant[ep_sq.file() as usize];
        }

        self.hash = hash;
    }
}

// =============================================================================
// Attack detection
// =============================================================================

impl Board {
    /// Returns true if `sq` is attacked by any piece of `by_color`.
    /// Uses the "reverse lookup" trick: cast rays FROM the target square
    /// and check if they hit an enemy piece of the matching type.
    #[inline]
    pub fn is_square_attacked(&self, sq: Square, by_color: Color) -> bool {
        let them = by_color.index();
        let occ = self.occupied;

        // Pawns: look for enemy pawns that attack this square.
        // Use our color's attack pattern (from sq) to find enemy pawns.
        let pawn_attackers = self.pieces[them][Piece::Pawn.index()];
        if (attacks::pawn_attacks(by_color.flip(), sq) & pawn_attackers).is_not_empty() {
            return true;
        }

        // Knights
        if (attacks::knight_attacks(sq) & self.pieces[them][Piece::Knight.index()]).is_not_empty() {
            return true;
        }

        // King
        if (attacks::king_attacks(sq) & self.pieces[them][Piece::King.index()]).is_not_empty() {
            return true;
        }

        // Bishops + Queens (diagonal)
        let diag =
            self.pieces[them][Piece::Bishop.index()] | self.pieces[them][Piece::Queen.index()];
        if (attacks::bishop_attacks(sq, occ) & diag).is_not_empty() {
            return true;
        }

        // Rooks + Queens (orthogonal)
        let orth = self.pieces[them][Piece::Rook.index()] | self.pieces[them][Piece::Queen.index()];
        if (attacks::rook_attacks(sq, occ) & orth).is_not_empty() {
            return true;
        }

        false
    }

    /// Returns true if `color`'s king is in check.
    #[inline]
    pub fn in_check(&self, color: Color) -> bool {
        let king_bb = self.pieces[color.index()][Piece::King.index()];
        if king_bb.is_empty() {
            return false;
        }
        let king_sq = king_bb.lsb();
        self.is_square_attacked(king_sq, color.flip())
    }
}

// =============================================================================
// Castling rights mask — per-square lookup
// =============================================================================

/// For each square, which castling rights to KEEP (AND-mask).
/// If a king or rook moves from, or a rook is captured on, a relevant square,
/// the corresponding castling right is revoked.
///
/// Squares not involved in castling have mask 0b1111 (keep all rights).
const CASTLING_RIGHTS_MASK: [u8; 64] = {
    let mut mask = [0b1111u8; 64];
    mask[0] = 0b1111 & !CastlingRights::WQ; // a1 — white queenside rook
    mask[4] = 0b1111 & !(CastlingRights::WK | CastlingRights::WQ); // e1 — white king
    mask[7] = 0b1111 & !CastlingRights::WK; // h1 — white kingside rook
    mask[56] = 0b1111 & !CastlingRights::BQ; // a8 — black queenside rook
    mask[60] = 0b1111 & !(CastlingRights::BK | CastlingRights::BQ); // e8 — black king
    mask[63] = 0b1111 & !CastlingRights::BK; // h8 — black kingside rook
    mask
};

// =============================================================================
// Make / unmake (in-place, for search)
// =============================================================================

/// Saved state for `do_move` / `undo_move`.
#[derive(Clone, Copy)]
pub struct Undo {
    pub mv: Move,
    pub moved_piece: Piece,
    pub captured: Option<(Color, Piece, Square)>,
    pub castling_rights: CastlingRights,
    pub en_passant_sq: Option<Square>,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    pub hash: u64,
}

/// Saved state for null-move pruning.
#[derive(Clone, Copy)]
pub struct NullUndo {
    pub en_passant_sq: Option<Square>,
    pub halfmove_clock: u8,
    pub hash: u64,
}

// =============================================================================
// Copy-make (also used by do_move)
// =============================================================================

impl Board {
    /// Apply a move in-place. Returns undo info for `undo_move`.
    pub fn do_move(&mut self, mv: Move) -> Undo {
        let us = self.side_to_move;
        let them = us.flip();
        let from = mv.from_sq();
        let to = mv.to_sq();
        let flags = mv.flags();
        let z = zobrist();

        let (_, moving_piece) = self.piece_at(from).expect("do_move: no piece at source");

        let mut undo = Undo {
            mv,
            moved_piece: moving_piece,
            captured: None,
            castling_rights: self.castling_rights,
            en_passant_sq: self.en_passant_sq,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            hash: self.hash,
        };

        self.hash ^= z.castling[self.castling_rights.0 as usize];
        if let Some(ep_sq) = self.en_passant_sq {
            self.hash ^= z.en_passant[ep_sq.file() as usize];
        }
        self.en_passant_sq = None;

        if flags == Move::FLAG_EP_CAPTURE {
            let captured_sq = Square::new(to.file(), from.rank());
            self.remove_piece(them, Piece::Pawn, captured_sq);
            undo.captured = Some((them, Piece::Pawn, captured_sq));
        } else if mv.is_capture() {
            if let Some((cap_color, cap_piece)) = self.piece_at(to) {
                debug_assert_eq!(cap_color, them);
                self.remove_piece(cap_color, cap_piece, to);
                undo.captured = Some((cap_color, cap_piece, to));
            }
        }

        self.remove_piece(us, moving_piece, from);

        if mv.is_promotion() {
            let promo_piece = mv
                .promotion_piece()
                .expect("do_move: promotion without piece");
            self.set_piece(us, promo_piece, to);
        } else {
            self.set_piece(us, moving_piece, to);
        }

        if flags == Move::FLAG_KS_CASTLE {
            match us {
                Color::White => self.move_piece(us, Piece::Rook, Square::H1, Square::F1),
                Color::Black => self.move_piece(us, Piece::Rook, Square::H8, Square::F8),
            }
        } else if flags == Move::FLAG_QS_CASTLE {
            match us {
                Color::White => self.move_piece(us, Piece::Rook, Square::A1, Square::D1),
                Color::Black => self.move_piece(us, Piece::Rook, Square::A8, Square::D8),
            }
        }

        if flags == Move::FLAG_DOUBLE_PUSH {
            let ep_sq = match us {
                Color::White => Square::new(from.file(), from.rank() + 1),
                Color::Black => Square::new(from.file(), from.rank() - 1),
            };
            self.en_passant_sq = Some(ep_sq);
        }

        self.castling_rights.0 &= CASTLING_RIGHTS_MASK[from.0 as usize];
        self.castling_rights.0 &= CASTLING_RIGHTS_MASK[to.0 as usize];

        self.hash ^= z.castling[self.castling_rights.0 as usize];
        if let Some(ep_sq) = self.en_passant_sq {
            self.hash ^= z.en_passant[ep_sq.file() as usize];
        }

        if moving_piece == Piece::Pawn || mv.is_capture() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if us == Color::Black {
            self.fullmove_number += 1;
        }

        self.side_to_move = them;
        self.hash ^= z.side;

        undo
    }

    /// Undo a move applied with `do_move`.
    pub fn undo_move(&mut self, undo: Undo) {
        let us = self.side_to_move.flip();
        let mv = undo.mv;
        let from = mv.from_sq();
        let to = mv.to_sq();
        let flags = mv.flags();

        self.side_to_move = us;
        self.hash = undo.hash;
        self.castling_rights = undo.castling_rights;
        self.en_passant_sq = undo.en_passant_sq;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;

        if flags == Move::FLAG_KS_CASTLE {
            match us {
                Color::White => self.move_piece(us, Piece::Rook, Square::F1, Square::H1),
                Color::Black => self.move_piece(us, Piece::Rook, Square::F8, Square::H8),
            }
        } else if flags == Move::FLAG_QS_CASTLE {
            match us {
                Color::White => self.move_piece(us, Piece::Rook, Square::D1, Square::A1),
                Color::Black => self.move_piece(us, Piece::Rook, Square::D8, Square::A8),
            }
        }

        if mv.is_promotion() {
            let promo = mv.promotion_piece().unwrap();
            self.remove_piece(us, promo, to);
        } else if let Some((color, piece)) = self.piece_at(to) {
            self.remove_piece(color, piece, to);
        }

        self.set_piece(us, undo.moved_piece, from);

        if let Some((color, piece, sq)) = undo.captured {
            self.set_piece(color, piece, sq);
        }
    }

    /// Apply a move using copy-make (for non-search callers).
    pub fn make_move(&self, mv: Move) -> Board {
        let mut b = *self;
        b.do_move(mv);
        b
    }

    /// Null move in-place for NMP.
    pub fn do_null_move(&mut self) -> NullUndo {
        let z = zobrist();
        let undo = NullUndo {
            en_passant_sq: self.en_passant_sq,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
        };
        if let Some(ep_sq) = self.en_passant_sq {
            self.hash ^= z.en_passant[ep_sq.file() as usize];
            self.en_passant_sq = None;
        }
        self.side_to_move = self.side_to_move.flip();
        self.hash ^= z.side;
        self.halfmove_clock += 1;
        undo
    }

    pub fn undo_null_move(&mut self, undo: NullUndo) {
        let z = zobrist();
        self.side_to_move = self.side_to_move.flip();
        self.hash = undo.hash;
        self.en_passant_sq = undo.en_passant_sq;
        self.halfmove_clock = undo.halfmove_clock;
        if let Some(ep_sq) = self.en_passant_sq {
            self.hash ^= z.en_passant[ep_sq.file() as usize];
        }
    }

    /// Copy-make null move (legacy).
    pub fn make_null_move(&self) -> Board {
        let mut b = *self;
        b.do_null_move();
        b
    }
}

// =============================================================================
// FEN parsing and serialization
// =============================================================================

/// Standard starting position FEN.
pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Board {
    /// Parse a FEN string into a Board.
    ///
    /// FEN format: `<pieces> <color> <castling> <en_passant> <halfmove> <fullmove>`
    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let mut parts = fen.split_whitespace();

        let pieces_str = parts
            .next()
            .ok_or(FenError::MissingField("piece placement"))?;
        let color_str = parts.next().ok_or(FenError::MissingField("side to move"))?;
        let castle_str = parts
            .next()
            .ok_or(FenError::MissingField("castling rights"))?;
        let ep_str = parts.next().ok_or(FenError::MissingField("en passant"))?;
        let hmove_str = parts.next().unwrap_or("0");
        let fmove_str = parts.next().unwrap_or("1");

        let mut board = Board::empty();

        // --- Parse piece placement ---
        // FEN ranks go from rank 8 (top) to rank 1 (bottom), separated by '/'
        for (rank_idx, rank_str) in pieces_str.split('/').enumerate() {
            if rank_idx >= 8 {
                return Err(FenError::InvalidPiecePlacement(
                    "too many ranks".to_string(),
                ));
            }
            let rank = 7 - rank_idx as u8; // FEN starts at rank 8
            let mut file = 0u8;

            for ch in rank_str.chars() {
                if file > 8 {
                    return Err(FenError::InvalidPiecePlacement(format!(
                        "too many squares on rank {}",
                        rank + 1
                    )));
                }
                match ch {
                    '1'..='8' => {
                        file += ch as u8 - b'0';
                    }
                    _ => {
                        let (color, piece) = Piece::from_char(ch).ok_or_else(|| {
                            FenError::InvalidPiecePlacement(format!("invalid piece char: '{}'", ch))
                        })?;
                        if file >= 8 {
                            return Err(FenError::InvalidPiecePlacement(format!(
                                "file overflow on rank {}",
                                rank + 1
                            )));
                        }
                        let sq = Square::new(file, rank);
                        board.set_piece(color, piece, sq);
                        file += 1;
                    }
                }
            }
        }

        // --- Side to move ---
        board.side_to_move = match color_str {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError::InvalidColor(color_str.to_string())),
        };

        // --- Castling rights ---
        board.castling_rights = CastlingRights::from_fen(castle_str);

        // --- En passant square ---
        board.en_passant_sq = if ep_str == "-" {
            None
        } else {
            Some(
                Square::from_algebraic(ep_str)
                    .ok_or_else(|| FenError::InvalidSquare(ep_str.to_string()))?,
            )
        };

        // --- Clocks ---
        board.halfmove_clock = hmove_str
            .parse::<u8>()
            .map_err(|_| FenError::InvalidNumber(hmove_str.to_string()))?;
        board.fullmove_number = fmove_str
            .parse::<u16>()
            .map_err(|_| FenError::InvalidNumber(fmove_str.to_string()))?;

        // --- Recompute derived state ---
        board.recompute_occupancy();
        // Hash was already incrementally computed by set_piece calls,
        // but we need to add castling, en passant, and side-to-move keys.
        // Easiest to just recompute from scratch.
        board.recompute_hash();

        Ok(board)
    }

    /// Serialize the board to a FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(80);

        // --- Piece placement ---
        for rank in (0..8).rev() {
            let mut empty_count = 0u8;
            for file in 0..8u8 {
                let sq = Square::new(file, rank);
                match self.piece_at(sq) {
                    Some((color, piece)) => {
                        if empty_count > 0 {
                            fen.push((b'0' + empty_count) as char);
                            empty_count = 0;
                        }
                        fen.push(piece.to_char(color));
                    }
                    None => {
                        empty_count += 1;
                    }
                }
            }
            if empty_count > 0 {
                fen.push((b'0' + empty_count) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // --- Side to move ---
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // --- Castling rights ---
        fen.push(' ');
        fen.push_str(&self.castling_rights.to_fen());

        // --- En passant ---
        fen.push(' ');
        match self.en_passant_sq {
            Some(sq) => fen.push_str(&sq.to_algebraic()),
            None => fen.push('-'),
        }

        // --- Clocks ---
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }
}

// =============================================================================
// Display — human-readable board
// =============================================================================

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} │", rank + 1)?;
            for file in 0..8u8 {
                let sq = Square::new(file, rank);
                let ch = match self.piece_at(sq) {
                    Some((color, piece)) => piece.to_char(color),
                    None => '.',
                };
                write!(f, " {}", ch)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "    ╰────────────────")?;
        writeln!(f, "      a b c d e f g h")?;
        writeln!(f)?;
        writeln!(f, "  Side to move: {}", self.side_to_move)?;
        writeln!(f, "  Castling:     {}", self.castling_rights)?;
        write!(f, "  En passant:   ")?;
        match self.en_passant_sq {
            Some(sq) => writeln!(f, "{}", sq.to_algebraic())?,
            None => writeln!(f, "-")?,
        }
        writeln!(f, "  Halfmove:     {}", self.halfmove_clock)?;
        writeln!(f, "  Fullmove:     {}", self.fullmove_number)?;
        writeln!(f, "  Zobrist:      {:#018x}", self.hash)?;
        Ok(())
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    // -----------------------------------------------------------------------
    // Empty board
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_board() {
        let board = Board::empty();
        assert!(board.occupied.is_empty());
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling_rights, CastlingRights::NONE);
        assert_eq!(board.en_passant_sq, None);
        for sq in 0..64 {
            assert_eq!(board.piece_at(Square(sq)), None);
        }
    }

    // -----------------------------------------------------------------------
    // Set / remove piece
    // -----------------------------------------------------------------------
    #[test]
    fn test_set_remove_piece() {
        let mut board = Board::empty();

        board.set_piece(Color::White, Piece::Knight, Square::E1);
        assert_eq!(
            board.piece_at(Square::E1),
            Some((Color::White, Piece::Knight))
        );
        assert!(board.occupied.has(Square::E1));
        assert!(board.by_color[0].has(Square::E1));
        assert_eq!(board.bb(Color::White, Piece::Knight), Square::E1.bit());

        let hash_before = board.hash;
        board.remove_piece(Color::White, Piece::Knight, Square::E1);
        assert_eq!(board.piece_at(Square::E1), None);
        assert!(board.occupied.is_empty());
        // After remove, hash should return to 0 (XOR is self-inverse)
        assert_eq!(
            board.hash, 0,
            "Zobrist hash should return to 0 after set+remove"
        );
        assert_ne!(
            hash_before, 0,
            "Hash should be non-zero after placing a piece"
        );
    }

    // -----------------------------------------------------------------------
    // FEN parsing — starting position
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_fen_startpos() {
        let board = Board::from_fen(STARTPOS).unwrap();

        // Piece counts
        assert_eq!(board.occupied.popcount(), 32);
        assert_eq!(board.by_color[0].popcount(), 16); // White
        assert_eq!(board.by_color[1].popcount(), 16); // Black

        // Specific pieces
        assert_eq!(board.bb(Color::White, Piece::Pawn).popcount(), 8);
        assert_eq!(board.bb(Color::Black, Piece::Pawn).popcount(), 8);
        assert_eq!(board.bb(Color::White, Piece::Rook).popcount(), 2);
        assert_eq!(board.bb(Color::White, Piece::Knight).popcount(), 2);
        assert_eq!(board.bb(Color::White, Piece::Bishop).popcount(), 2);
        assert_eq!(board.bb(Color::White, Piece::Queen).popcount(), 1);
        assert_eq!(board.bb(Color::White, Piece::King).popcount(), 1);

        // Specific squares
        assert_eq!(
            board.piece_at(Square::E1),
            Some((Color::White, Piece::King))
        );
        assert_eq!(
            board.piece_at(Square::E8),
            Some((Color::Black, Piece::King))
        );
        assert_eq!(
            board.piece_at(Square::D1),
            Some((Color::White, Piece::Queen))
        );
        assert_eq!(
            board.piece_at(Square::A1),
            Some((Color::White, Piece::Rook))
        );
        assert_eq!(
            board.piece_at(Square::H8),
            Some((Color::Black, Piece::Rook))
        );

        // Game state
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling_rights.0, CastlingRights::ALL);
        assert_eq!(board.en_passant_sq, None);
        assert_eq!(board.halfmove_clock, 0);
        assert_eq!(board.fullmove_number, 1);
    }

    // -----------------------------------------------------------------------
    // FEN parsing — kiwipete position
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_fen_kiwipete() {
        let board = Board::from_fen(KIWIPETE).unwrap();

        assert_eq!(board.occupied.popcount(), 32);
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling_rights.0, CastlingRights::ALL);
        assert_eq!(board.en_passant_sq, None);

        // White knight on e5
        assert_eq!(
            board.piece_at(Square(36)),
            Some((Color::White, Piece::Knight))
        );
        // Black queen on e7
        assert_eq!(
            board.piece_at(Square(52)),
            Some((Color::Black, Piece::Queen))
        );
    }

    // -----------------------------------------------------------------------
    // FEN roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn test_fen_roundtrip_startpos() {
        let board = Board::from_fen(STARTPOS).unwrap();
        assert_eq!(board.to_fen(), STARTPOS);
    }

    #[test]
    fn test_fen_roundtrip_kiwipete() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(board.to_fen(), KIWIPETE);
    }

    #[test]
    fn test_fen_roundtrip_various() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "8/8/8/8/8/8/8/4K3 w - - 0 1",               // lone king
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",      // rooks + kings
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // perft position 3
        ];
        for fen in &fens {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(&board.to_fen(), fen, "FEN roundtrip failed for: {}", fen);
        }
    }

    // -----------------------------------------------------------------------
    // FEN error handling
    // -----------------------------------------------------------------------
    #[test]
    fn test_from_fen_missing_fields() {
        assert!(Board::from_fen("").is_err());
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").is_err());
    }

    #[test]
    fn test_from_fen_invalid_color() {
        let result = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // start_pos convenience
    // -----------------------------------------------------------------------
    #[test]
    fn test_start_pos() {
        let a = Board::start_pos();
        let b = Board::from_fen(STARTPOS).unwrap();
        assert_eq!(a.to_fen(), b.to_fen());
    }

    // -----------------------------------------------------------------------
    // Zobrist hash consistency
    // -----------------------------------------------------------------------
    #[test]
    fn test_hash_deterministic() {
        let a = Board::from_fen(STARTPOS).unwrap();
        let b = Board::from_fen(STARTPOS).unwrap();
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn test_hash_differs_by_position() {
        let a = Board::from_fen(STARTPOS).unwrap();
        let b = Board::from_fen(KIWIPETE).unwrap();
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn test_hash_nonzero() {
        let board = Board::start_pos();
        assert_ne!(board.hash, 0);
    }

    #[test]
    fn test_recompute_hash_consistency() {
        let mut board = Board::from_fen(STARTPOS).unwrap();
        let original_hash = board.hash;
        board.recompute_hash();
        assert_eq!(
            board.hash, original_hash,
            "Full recompute should match incremental hash"
        );
    }

    // -----------------------------------------------------------------------
    // Copy-make basic tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_make_move_quiet() {
        let board = Board::start_pos();
        // Nf3: g1(6) -> f3(21)
        let mv = Move::new(Square::G1, Square(21), Move::FLAG_QUIET);
        let new_board = board.make_move(mv);

        assert_eq!(new_board.piece_at(Square::G1), None);
        assert_eq!(
            new_board.piece_at(Square(21)),
            Some((Color::White, Piece::Knight))
        );
        assert_eq!(new_board.side_to_move, Color::Black);
        assert_eq!(new_board.halfmove_clock, 1);
        assert_eq!(new_board.fullmove_number, 1); // still move 1
    }

    #[test]
    fn test_make_move_double_push() {
        let board = Board::start_pos();
        // e2-e4 (double push)
        let mv = Move::new(Square::E2, Square(28), Move::FLAG_DOUBLE_PUSH);
        let new_board = board.make_move(mv);

        assert_eq!(new_board.piece_at(Square::E2), None);
        assert_eq!(
            new_board.piece_at(Square(28)),
            Some((Color::White, Piece::Pawn))
        );
        assert_eq!(new_board.en_passant_sq, Some(Square(20))); // e3
        assert_eq!(new_board.halfmove_clock, 0); // pawn move resets
    }

    #[test]
    fn test_make_move_preserves_original() {
        let board = Board::start_pos();
        let mv = Move::new(Square::E2, Square(28), Move::FLAG_DOUBLE_PUSH);
        let _new_board = board.make_move(mv);

        // Original board should be unchanged (copy-make)
        assert_eq!(
            board.piece_at(Square::E2),
            Some((Color::White, Piece::Pawn))
        );
        assert_eq!(board.piece_at(Square(28)), None);
        assert_eq!(board.side_to_move, Color::White);
    }

    #[test]
    fn test_make_move_hash_changes() {
        let board = Board::start_pos();
        let mv = Move::new(Square::E2, Square(28), Move::FLAG_DOUBLE_PUSH);
        let new_board = board.make_move(mv);

        assert_ne!(
            board.hash, new_board.hash,
            "Hash should change after a move"
        );

        // Verify incremental hash matches full recompute
        let mut verify = new_board.clone();
        verify.recompute_hash();
        assert_eq!(
            new_board.hash, verify.hash,
            "Incremental Zobrist should match full recompute"
        );
    }

    #[test]
    fn test_make_move_kingside_castle() {
        // Position where White can castle kingside
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let mv = Move::new(Square::E1, Square::G1, Move::FLAG_KS_CASTLE);
        let new_board = board.make_move(mv);

        assert_eq!(new_board.piece_at(Square::E1), None);
        assert_eq!(
            new_board.piece_at(Square::G1),
            Some((Color::White, Piece::King))
        );
        assert_eq!(new_board.piece_at(Square::H1), None);
        assert_eq!(
            new_board.piece_at(Square::F1),
            Some((Color::White, Piece::Rook))
        );
        // White castling rights removed
        assert!(!new_board.castling_rights.has(CastlingRights::WK));
        assert!(!new_board.castling_rights.has(CastlingRights::WQ));
        // Black castling rights preserved
        assert!(new_board.castling_rights.has(CastlingRights::BK));
        assert!(new_board.castling_rights.has(CastlingRights::BQ));
    }

    #[test]
    fn test_make_move_queenside_castle() {
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let mv = Move::new(Square::E1, Square::C1, Move::FLAG_QS_CASTLE);
        let new_board = board.make_move(mv);

        assert_eq!(new_board.piece_at(Square::E1), None);
        assert_eq!(
            new_board.piece_at(Square::C1),
            Some((Color::White, Piece::King))
        );
        assert_eq!(new_board.piece_at(Square::A1), None);
        assert_eq!(
            new_board.piece_at(Square::D1),
            Some((Color::White, Piece::Rook))
        );
    }

    #[test]
    fn test_make_move_promotion() {
        let board = Board::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = Move::new(Square(52), Square(60), Move::FLAG_PROMO_Q);
        let new_board = board.make_move(mv);

        assert_eq!(new_board.piece_at(Square(52)), None);
        assert_eq!(
            new_board.piece_at(Square(60)),
            Some((Color::White, Piece::Queen))
        );
    }
}
