// =============================================================================
// movegen.rs — Full pseudo-legal and legal move generation + perft
// =============================================================================
//
// Generates all moves for all piece types. Special cases:
// - Pawn: single/double push, captures, en passant, promotions (all 4 types)
// - Castling: all 5 legality checks
// - Legal filtering: copy-make + in_check test
//
// Perft: recursive leaf counter for correctness validation.

use crate::attacks;
use crate::board::Board;
use crate::types::{BitBoard, CastlingRights, Color, Move, MoveList, Piece, Square};

// =============================================================================
// Castling constants
// =============================================================================

// Squares that must be empty for castling
const WK_CASTLE_EMPTY: BitBoard = BitBoard(0x0000_0000_0000_0060); // f1, g1
const WQ_CASTLE_EMPTY: BitBoard = BitBoard(0x0000_0000_0000_000E); // b1, c1, d1
const BK_CASTLE_EMPTY: BitBoard = BitBoard(0x6000_0000_0000_0000); // f8, g8
const BQ_CASTLE_EMPTY: BitBoard = BitBoard(0x0E00_0000_0000_0000); // b8, c8, d8

// =============================================================================
// Move generation — impl on Board
// =============================================================================

impl Board {
    /// Generate all pseudo-legal moves for the side to move.
    /// These are geometrically valid moves but may leave the king in check.
    pub fn generate_pseudo_legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        let us = self.side_to_move;

        self.gen_pawn_moves(&mut list, us);
        self.gen_knight_moves(&mut list, us);
        self.gen_bishop_moves(&mut list, us);
        self.gen_rook_moves(&mut list, us);
        self.gen_queen_moves(&mut list, us);
        self.gen_king_moves(&mut list, us);
        self.gen_castling_moves(&mut list, us);

        list
    }

    /// Generate only legal moves (king not in check after the move).
    pub fn generate_legal_moves(&self) -> MoveList {
        let pseudo = self.generate_pseudo_legal_moves();
        let mut legal = MoveList::new();
        let us = self.side_to_move;
        for mv in pseudo.as_slice() {
            let after = self.make_move(*mv);
            if !after.in_check(us) {
                legal.push(*mv);
            }
        }
        legal
    }

    /// Generate pseudo-legal captures + queen promotions (for quiescence search).
    /// Includes: captures, en passant, promotion captures, non-capture queen promotions.
    /// Excludes: quiet moves, castling, non-capture underpromotions.
    pub fn generate_captures(&self) -> MoveList {
        let mut list = MoveList::new();
        let us = self.side_to_move;
        let them = us.flip();
        let their_occ = self.by_color[them.index()];
        let our = self.by_color[us.index()];
        let occ = self.occupied;
        let empty = !occ;

        // --- Pawn captures + queen promotions ---
        let pawns = self.pieces[us.index()][Piece::Pawn.index()];
        match us {
            Color::White => {
                let promo_mask = BitBoard::RANK_8;
                // Left captures
                let left_cap = (pawns << 7) & !BitBoard::FILE_H & their_occ;
                let mut non_promo_left = left_cap & !promo_mask;
                while non_promo_left.is_not_empty() {
                    let to = non_promo_left.pop_lsb();
                    list.push(Move::new(Square(to.0 - 7), to, Move::FLAG_CAPTURE));
                }
                let mut promo_left = left_cap & promo_mask;
                while promo_left.is_not_empty() {
                    let to = promo_left.pop_lsb();
                    let from = Square(to.0 - 7);
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
                }
                // Right captures
                let right_cap = (pawns << 9) & !BitBoard::FILE_A & their_occ;
                let mut non_promo_right = right_cap & !promo_mask;
                while non_promo_right.is_not_empty() {
                    let to = non_promo_right.pop_lsb();
                    list.push(Move::new(Square(to.0 - 9), to, Move::FLAG_CAPTURE));
                }
                let mut promo_right = right_cap & promo_mask;
                while promo_right.is_not_empty() {
                    let to = promo_right.pop_lsb();
                    let from = Square(to.0 - 9);
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
                }
                // Non-capture queen promotions only
                let single = (pawns << 8) & empty;
                let mut promo_push = single & promo_mask;
                while promo_push.is_not_empty() {
                    let to = promo_push.pop_lsb();
                    list.push(Move::new(Square(to.0 - 8), to, Move::FLAG_PROMO_Q));
                }
            }
            Color::Black => {
                let promo_mask = BitBoard::RANK_1;
                let left_cap = (pawns >> 9) & !BitBoard::FILE_H & their_occ;
                let mut non_promo_left = left_cap & !promo_mask;
                while non_promo_left.is_not_empty() {
                    let to = non_promo_left.pop_lsb();
                    list.push(Move::new(Square(to.0 + 9), to, Move::FLAG_CAPTURE));
                }
                let mut promo_left = left_cap & promo_mask;
                while promo_left.is_not_empty() {
                    let to = promo_left.pop_lsb();
                    let from = Square(to.0 + 9);
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
                }
                let right_cap = (pawns >> 7) & !BitBoard::FILE_A & their_occ;
                let mut non_promo_right = right_cap & !promo_mask;
                while non_promo_right.is_not_empty() {
                    let to = non_promo_right.pop_lsb();
                    list.push(Move::new(Square(to.0 + 7), to, Move::FLAG_CAPTURE));
                }
                let mut promo_right = right_cap & promo_mask;
                while promo_right.is_not_empty() {
                    let to = promo_right.pop_lsb();
                    let from = Square(to.0 + 7);
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
                    list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
                }
                let single = (pawns >> 8) & empty;
                let mut promo_push = single & promo_mask;
                while promo_push.is_not_empty() {
                    let to = promo_push.pop_lsb();
                    list.push(Move::new(Square(to.0 + 8), to, Move::FLAG_PROMO_Q));
                }
            }
        }

        // En passant
        if let Some(ep_sq) = self.en_passant_sq {
            let ep_attackers = attacks::pawn_attacks(them, ep_sq) & pawns;
            let mut attackers = ep_attackers;
            while attackers.is_not_empty() {
                let from = attackers.pop_lsb();
                list.push(Move::new(from, ep_sq, Move::FLAG_EP_CAPTURE));
            }
        }

        // --- Non-pawn piece captures ---
        // Knights
        let mut knights = self.pieces[us.index()][Piece::Knight.index()];
        while knights.is_not_empty() {
            let from = knights.pop_lsb();
            let mut caps = attacks::knight_attacks(from) & their_occ;
            while caps.is_not_empty() {
                list.push(Move::new(from, caps.pop_lsb(), Move::FLAG_CAPTURE));
            }
        }

        // Bishops
        let mut bishops = self.pieces[us.index()][Piece::Bishop.index()];
        while bishops.is_not_empty() {
            let from = bishops.pop_lsb();
            let mut caps = attacks::bishop_attacks(from, occ) & their_occ;
            while caps.is_not_empty() {
                list.push(Move::new(from, caps.pop_lsb(), Move::FLAG_CAPTURE));
            }
        }

        // Rooks
        let mut rooks = self.pieces[us.index()][Piece::Rook.index()];
        while rooks.is_not_empty() {
            let from = rooks.pop_lsb();
            let mut caps = attacks::rook_attacks(from, occ) & their_occ;
            while caps.is_not_empty() {
                list.push(Move::new(from, caps.pop_lsb(), Move::FLAG_CAPTURE));
            }
        }

        // Queens
        let mut queens = self.pieces[us.index()][Piece::Queen.index()];
        while queens.is_not_empty() {
            let from = queens.pop_lsb();
            let mut caps = attacks::queen_attacks(from, occ) & their_occ;
            while caps.is_not_empty() {
                list.push(Move::new(from, caps.pop_lsb(), Move::FLAG_CAPTURE));
            }
        }

        // King
        let king_bb = self.pieces[us.index()][Piece::King.index()];
        if king_bb.is_not_empty() {
            let from = king_bb.lsb();
            let mut caps = attacks::king_attacks(from) & their_occ;
            while caps.is_not_empty() {
                list.push(Move::new(from, caps.pop_lsb(), Move::FLAG_CAPTURE));
            }
        }

        list
    }

    // =========================================================================
    // Pawn moves — the most complex piece
    // =========================================================================

    fn gen_pawn_moves(&self, list: &mut MoveList, us: Color) {
        let them = us.flip();
        let pawns = self.pieces[us.index()][Piece::Pawn.index()];
        let their_occ = self.by_color[them.index()];
        let empty = !self.occupied;

        match us {
            Color::White => self.gen_white_pawn_moves(list, pawns, their_occ, empty),
            Color::Black => self.gen_black_pawn_moves(list, pawns, their_occ, empty),
        }

        // En passant
        if let Some(ep_sq) = self.en_passant_sq {
            let ep_attackers = attacks::pawn_attacks(them, ep_sq) & pawns;
            let mut attackers = ep_attackers;
            while attackers.is_not_empty() {
                let from = attackers.pop_lsb();
                list.push(Move::new(from, ep_sq, Move::FLAG_EP_CAPTURE));
            }
        }
    }

    fn gen_white_pawn_moves(
        &self,
        list: &mut MoveList,
        pawns: BitBoard,
        their_occ: BitBoard,
        empty: BitBoard,
    ) {
        let promo_mask = BitBoard::RANK_8;

        // Single pushes
        let single = (pawns << 8) & empty;
        let mut non_promo = single & !promo_mask;
        while non_promo.is_not_empty() {
            let to = non_promo.pop_lsb();
            list.push(Move::new(Square(to.0 - 8), to, Move::FLAG_QUIET));
        }
        let mut promo = single & promo_mask;
        while promo.is_not_empty() {
            let to = promo.pop_lsb();
            let from = Square(to.0 - 8);
            list.push(Move::new(from, to, Move::FLAG_PROMO_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_N));
        }

        // Double pushes (only from rank 2, both squares must be empty)
        let double = ((pawns & BitBoard::RANK_2) << 8 & empty) << 8 & empty;
        let mut doubles = double;
        while doubles.is_not_empty() {
            let to = doubles.pop_lsb();
            list.push(Move::new(Square(to.0 - 16), to, Move::FLAG_DOUBLE_PUSH));
        }

        // Left captures (file decreases = shift 7, mask !FILE_H to prevent wrap)
        let left_cap = (pawns << 7) & !BitBoard::FILE_H & their_occ;
        let mut non_promo_left = left_cap & !promo_mask;
        while non_promo_left.is_not_empty() {
            let to = non_promo_left.pop_lsb();
            list.push(Move::new(Square(to.0 - 7), to, Move::FLAG_CAPTURE));
        }
        let mut promo_left = left_cap & promo_mask;
        while promo_left.is_not_empty() {
            let to = promo_left.pop_lsb();
            let from = Square(to.0 - 7);
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
        }

        // Right captures (file increases = shift 9, mask !FILE_A to prevent wrap)
        let right_cap = (pawns << 9) & !BitBoard::FILE_A & their_occ;
        let mut non_promo_right = right_cap & !promo_mask;
        while non_promo_right.is_not_empty() {
            let to = non_promo_right.pop_lsb();
            list.push(Move::new(Square(to.0 - 9), to, Move::FLAG_CAPTURE));
        }
        let mut promo_right = right_cap & promo_mask;
        while promo_right.is_not_empty() {
            let to = promo_right.pop_lsb();
            let from = Square(to.0 - 9);
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
        }
    }

    fn gen_black_pawn_moves(
        &self,
        list: &mut MoveList,
        pawns: BitBoard,
        their_occ: BitBoard,
        empty: BitBoard,
    ) {
        let promo_mask = BitBoard::RANK_1;

        // Single pushes
        let single = (pawns >> 8) & empty;
        let mut non_promo = single & !promo_mask;
        while non_promo.is_not_empty() {
            let to = non_promo.pop_lsb();
            list.push(Move::new(Square(to.0 + 8), to, Move::FLAG_QUIET));
        }
        let mut promo = single & promo_mask;
        while promo.is_not_empty() {
            let to = promo.pop_lsb();
            let from = Square(to.0 + 8);
            list.push(Move::new(from, to, Move::FLAG_PROMO_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_N));
        }

        // Double pushes (only from rank 7)
        let double = ((pawns & BitBoard::RANK_7) >> 8 & empty) >> 8 & empty;
        let mut doubles = double;
        while doubles.is_not_empty() {
            let to = doubles.pop_lsb();
            list.push(Move::new(Square(to.0 + 16), to, Move::FLAG_DOUBLE_PUSH));
        }

        // Left captures (from Black's perspective: shift right 9, mask !FILE_H)
        let left_cap = (pawns >> 9) & !BitBoard::FILE_H & their_occ;
        let mut non_promo_left = left_cap & !promo_mask;
        while non_promo_left.is_not_empty() {
            let to = non_promo_left.pop_lsb();
            list.push(Move::new(Square(to.0 + 9), to, Move::FLAG_CAPTURE));
        }
        let mut promo_left = left_cap & promo_mask;
        while promo_left.is_not_empty() {
            let to = promo_left.pop_lsb();
            let from = Square(to.0 + 9);
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
        }

        // Right captures (shift right 7, mask !FILE_A)
        let right_cap = (pawns >> 7) & !BitBoard::FILE_A & their_occ;
        let mut non_promo_right = right_cap & !promo_mask;
        while non_promo_right.is_not_empty() {
            let to = non_promo_right.pop_lsb();
            list.push(Move::new(Square(to.0 + 7), to, Move::FLAG_CAPTURE));
        }
        let mut promo_right = right_cap & promo_mask;
        while promo_right.is_not_empty() {
            let to = promo_right.pop_lsb();
            let from = Square(to.0 + 7);
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_Q));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_R));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_B));
            list.push(Move::new(from, to, Move::FLAG_PROMO_CAP_N));
        }
    }

    // =========================================================================
    // Knight moves
    // =========================================================================

    fn gen_knight_moves(&self, list: &mut MoveList, us: Color) {
        let our = self.by_color[us.index()];
        let their_occ = self.by_color[us.flip().index()];
        let mut knights = self.pieces[us.index()][Piece::Knight.index()];

        while knights.is_not_empty() {
            let from = knights.pop_lsb();
            let targets = attacks::knight_attacks(from) & !our;

            let mut captures = targets & their_occ;
            while captures.is_not_empty() {
                list.push(Move::new(from, captures.pop_lsb(), Move::FLAG_CAPTURE));
            }
            let mut quiets = targets & !their_occ;
            while quiets.is_not_empty() {
                list.push(Move::new(from, quiets.pop_lsb(), Move::FLAG_QUIET));
            }
        }
    }

    // =========================================================================
    // Bishop moves
    // =========================================================================

    fn gen_bishop_moves(&self, list: &mut MoveList, us: Color) {
        let our = self.by_color[us.index()];
        let their_occ = self.by_color[us.flip().index()];
        let occ = self.occupied;
        let mut bishops = self.pieces[us.index()][Piece::Bishop.index()];

        while bishops.is_not_empty() {
            let from = bishops.pop_lsb();
            let targets = attacks::bishop_attacks(from, occ) & !our;

            let mut captures = targets & their_occ;
            while captures.is_not_empty() {
                list.push(Move::new(from, captures.pop_lsb(), Move::FLAG_CAPTURE));
            }
            let mut quiets = targets & !their_occ;
            while quiets.is_not_empty() {
                list.push(Move::new(from, quiets.pop_lsb(), Move::FLAG_QUIET));
            }
        }
    }

    // =========================================================================
    // Rook moves
    // =========================================================================

    fn gen_rook_moves(&self, list: &mut MoveList, us: Color) {
        let our = self.by_color[us.index()];
        let their_occ = self.by_color[us.flip().index()];
        let occ = self.occupied;
        let mut rooks = self.pieces[us.index()][Piece::Rook.index()];

        while rooks.is_not_empty() {
            let from = rooks.pop_lsb();
            let targets = attacks::rook_attacks(from, occ) & !our;

            let mut captures = targets & their_occ;
            while captures.is_not_empty() {
                list.push(Move::new(from, captures.pop_lsb(), Move::FLAG_CAPTURE));
            }
            let mut quiets = targets & !their_occ;
            while quiets.is_not_empty() {
                list.push(Move::new(from, quiets.pop_lsb(), Move::FLAG_QUIET));
            }
        }
    }

    // =========================================================================
    // Queen moves
    // =========================================================================

    fn gen_queen_moves(&self, list: &mut MoveList, us: Color) {
        let our = self.by_color[us.index()];
        let their_occ = self.by_color[us.flip().index()];
        let occ = self.occupied;
        let mut queens = self.pieces[us.index()][Piece::Queen.index()];

        while queens.is_not_empty() {
            let from = queens.pop_lsb();
            let targets = attacks::queen_attacks(from, occ) & !our;

            let mut captures = targets & their_occ;
            while captures.is_not_empty() {
                list.push(Move::new(from, captures.pop_lsb(), Move::FLAG_CAPTURE));
            }
            let mut quiets = targets & !their_occ;
            while quiets.is_not_empty() {
                list.push(Move::new(from, quiets.pop_lsb(), Move::FLAG_QUIET));
            }
        }
    }

    // =========================================================================
    // King moves (non-castling)
    // =========================================================================

    fn gen_king_moves(&self, list: &mut MoveList, us: Color) {
        let our = self.by_color[us.index()];
        let their_occ = self.by_color[us.flip().index()];
        let king_bb = self.pieces[us.index()][Piece::King.index()];
        if king_bb.is_empty() {
            return;
        }
        let from = king_bb.lsb();
        let targets = attacks::king_attacks(from) & !our;

        let mut captures = targets & their_occ;
        while captures.is_not_empty() {
            list.push(Move::new(from, captures.pop_lsb(), Move::FLAG_CAPTURE));
        }
        let mut quiets = targets & !their_occ;
        while quiets.is_not_empty() {
            list.push(Move::new(from, quiets.pop_lsb(), Move::FLAG_QUIET));
        }
    }

    // =========================================================================
    // Castling — 5 legality conditions per side
    // =========================================================================

    fn gen_castling_moves(&self, list: &mut MoveList, us: Color) {
        let them = us.flip();
        let occ = self.occupied;

        match us {
            Color::White => {
                // Kingside: e1→g1, rook h1→f1
                if self.castling_rights.has(CastlingRights::WK)
                    && (occ & WK_CASTLE_EMPTY).is_empty()
                    && !self.is_square_attacked(Square::E1, them)
                    && !self.is_square_attacked(Square::F1, them)
                    && !self.is_square_attacked(Square::G1, them)
                {
                    list.push(Move::new(Square::E1, Square::G1, Move::FLAG_KS_CASTLE));
                }
                // Queenside: e1→c1, rook a1→d1
                // Note: b1 must be empty but does NOT need to be un-attacked
                if self.castling_rights.has(CastlingRights::WQ)
                    && (occ & WQ_CASTLE_EMPTY).is_empty()
                    && !self.is_square_attacked(Square::E1, them)
                    && !self.is_square_attacked(Square::D1, them)
                    && !self.is_square_attacked(Square::C1, them)
                {
                    list.push(Move::new(Square::E1, Square::C1, Move::FLAG_QS_CASTLE));
                }
            }
            Color::Black => {
                // Kingside: e8→g8
                if self.castling_rights.has(CastlingRights::BK)
                    && (occ & BK_CASTLE_EMPTY).is_empty()
                    && !self.is_square_attacked(Square::E8, them)
                    && !self.is_square_attacked(Square::F8, them)
                    && !self.is_square_attacked(Square::G8, them)
                {
                    list.push(Move::new(Square::E8, Square::G8, Move::FLAG_KS_CASTLE));
                }
                // Queenside: e8→c8
                if self.castling_rights.has(CastlingRights::BQ)
                    && (occ & BQ_CASTLE_EMPTY).is_empty()
                    && !self.is_square_attacked(Square::E8, them)
                    && !self.is_square_attacked(Square::D8, them)
                    && !self.is_square_attacked(Square::C8, them)
                {
                    list.push(Move::new(Square::E8, Square::C8, Move::FLAG_QS_CASTLE));
                }
            }
        }
    }
}

// =============================================================================
// Perft — recursive leaf node counter for correctness validation
// =============================================================================

/// Count leaf nodes at `depth` from the given position.
/// Uses pseudo-legal generation + in_check filtering (copy-make).
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = board.generate_pseudo_legal_moves();
    let us = board.side_to_move;
    let mut count = 0u64;

    for mv in moves.as_slice() {
        let child = board.make_move(*mv);
        if !child.in_check(us) {
            count += perft(&child, depth - 1);
        }
    }

    count
}

/// Divide perft: print per-root-move node counts.
/// Used to bisect where counts diverge from a reference engine.
#[allow(dead_code)]
pub fn perft_divide(board: &Board, depth: u32) {
    let moves = board.generate_legal_moves();
    let mut total = 0u64;

    for mv in moves.as_slice() {
        let child = board.make_move(*mv);
        let nodes = if depth > 1 {
            perft(&child, depth - 1)
        } else {
            1
        };
        println!("{}: {}", mv.to_uci(), nodes);
        total += nodes;
    }

    println!("\nMoves: {}", moves.len());
    println!("Total: {}", total);
}

// =============================================================================
// Tests — perft is the non-negotiable exit condition
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Starting position perft
    // =========================================================================

    #[test]
    fn test_startpos_legal_moves() {
        let board = Board::start_pos();
        let moves = board.generate_legal_moves();
        assert_eq!(
            moves.len(),
            20,
            "Starting position should have exactly 20 legal moves"
        );
    }

    #[test]
    fn test_perft_startpos_1() {
        let board = Board::start_pos();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn test_perft_startpos_2() {
        let board = Board::start_pos();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn test_perft_startpos_3() {
        let board = Board::start_pos();
        assert_eq!(perft(&board, 3), 8_902);
    }

    #[test]
    fn test_perft_startpos_4() {
        let board = Board::start_pos();
        assert_eq!(perft(&board, 4), 197_281);
    }

    #[test]
    fn test_perft_startpos_5() {
        let board = Board::start_pos();
        assert_eq!(perft(&board, 5), 4_865_609);
    }

    // =========================================================================
    // Kiwipete — castling, en passant, pins, discovered checks
    // =========================================================================

    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn test_perft_kiwipete_1() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn test_perft_kiwipete_2() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 2), 2_039);
    }

    #[test]
    fn test_perft_kiwipete_3() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 3), 97_862);
    }

    #[test]
    fn test_perft_kiwipete_4() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        assert_eq!(perft(&board, 4), 4_085_603);
    }

    // =========================================================================
    // Position 3 — en passant & promotion edge cases
    // =========================================================================

    const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    #[test]
    fn test_perft_pos3_1() {
        let board = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&board, 1), 14);
    }

    #[test]
    fn test_perft_pos3_2() {
        let board = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&board, 2), 191);
    }

    #[test]
    fn test_perft_pos3_3() {
        let board = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&board, 3), 2_812);
    }

    #[test]
    fn test_perft_pos3_4() {
        let board = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&board, 4), 43_238);
    }

    #[test]
    fn test_perft_pos3_5() {
        let board = Board::from_fen(POS3).unwrap();
        assert_eq!(perft(&board, 5), 674_624);
    }

    // =========================================================================
    // Position 4 — pins & discovered check
    // =========================================================================

    const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

    #[test]
    fn test_perft_pos4_1() {
        let board = Board::from_fen(POS4).unwrap();
        assert_eq!(perft(&board, 1), 6);
    }

    #[test]
    fn test_perft_pos4_2() {
        let board = Board::from_fen(POS4).unwrap();
        assert_eq!(perft(&board, 2), 264);
    }

    #[test]
    fn test_perft_pos4_3() {
        let board = Board::from_fen(POS4).unwrap();
        assert_eq!(perft(&board, 3), 9_467);
    }

    #[test]
    fn test_perft_pos4_4() {
        let board = Board::from_fen(POS4).unwrap();
        assert_eq!(perft(&board, 4), 422_333);
    }

    // =========================================================================
    // is_square_attacked & in_check
    // =========================================================================

    #[test]
    fn test_startpos_not_in_check() {
        let board = Board::start_pos();
        assert!(!board.in_check(Color::White));
        assert!(!board.in_check(Color::Black));
    }

    #[test]
    fn test_in_check_simple() {
        // White king on e1, black rook on e8, nothing between
        let board = Board::from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(board.in_check(Color::White));
    }

    #[test]
    fn test_not_in_check_blocked() {
        // White king on e1, black rook on e8, blocker on e4
        let board = Board::from_fen("4r3/8/8/8/4P3/8/8/4K3 w - - 0 1").unwrap();
        assert!(!board.in_check(Color::White));
    }

    #[test]
    fn test_in_check_by_knight() {
        // White king on e1, black knight on d3
        let board = Board::from_fen("8/8/8/8/8/3n4/8/4K3 w - - 0 1").unwrap();
        assert!(board.in_check(Color::White));
    }

    #[test]
    fn test_in_check_by_pawn() {
        // White king on e4, black pawn on d5
        let board = Board::from_fen("8/8/8/3p4/4K3/8/8/8 w - - 0 1").unwrap();
        assert!(board.in_check(Color::White));
    }

    // =========================================================================
    // Hash consistency through legal moves
    // =========================================================================

    #[test]
    fn test_hash_consistency_after_legal_moves() {
        let board = Board::start_pos();
        let moves = board.generate_legal_moves();

        for mv in moves.as_slice() {
            let mut after = board.make_move(*mv);
            let incremental_hash = after.hash;
            after.recompute_hash();
            assert_eq!(
                incremental_hash,
                after.hash,
                "Hash mismatch after move {}",
                mv.to_uci()
            );
        }
    }

    #[test]
    fn test_hash_consistency_kiwipete() {
        let board = Board::from_fen(KIWIPETE).unwrap();
        let moves = board.generate_legal_moves();

        for mv in moves.as_slice() {
            let mut after = board.make_move(*mv);
            let incremental_hash = after.hash;
            after.recompute_hash();
            assert_eq!(
                incremental_hash,
                after.hash,
                "Hash mismatch after move {} in Kiwipete",
                mv.to_uci()
            );
        }
    }
}
