// =============================================================================
// eval.rs — Tapered Evaluation (PeSTO) and Positional Bonuses
// =============================================================================

use std::sync::OnceLock;

use crate::attacks;
use crate::board::Board;
use crate::types::{BitBoard, Color, Piece, Square};

// =============================================================================
// Constants and Mate Scores
// =============================================================================

pub const INFINITY: i32 = 32000;
pub const MATE_VALUE: i32 = 31000;
pub const DRAW_SCORE: i32 = 0;

pub fn mate_in(ply: usize) -> i32 {
    MATE_VALUE - ply as i32
}
pub fn mated_in(ply: usize) -> i32 {
    -MATE_VALUE + ply as i32
}
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_VALUE - 1000
}

// =============================================================================
// PeSTO Base Values
// =============================================================================

pub const PIECE_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0]; // Used by move ordering/pruning
const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

const MG_PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14,
    13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0,
];

const EG_PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82,
    84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13,
    8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
];

const MG_KNIGHT_TABLE: [i32; 64] = [
    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84,
    129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10,
    19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
];

const EG_KNIGHT_TABLE: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9,
    -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1,
    15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50,
    -64,
];

const MG_BISHOP_TABLE: [i32; 64] = [
    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50,
    37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18,
    10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21,
];

const EG_BISHOP_TABLE: [i32; 64] = [
    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0,
    4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17,
];

const MG_ROOK_TABLE: [i32; 64] = [
    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0,
    -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26,
];

const EG_ROOK_TABLE: [i32; 64] = [
    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13,
    1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9,
    -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
];

const MG_QUEEN_TABLE: [i32; 64] = [
    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47,
    57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2,
    14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50,
];

const EG_QUEEN_TABLE: [i32; 64] = [
    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41,
];

const MG_KING_TABLE: [i32; 64] = [
    -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6,
    22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14,
    -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24,
    14,
];

const EG_KING_TABLE: [i32; 64] = [
    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45,
    44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23,
    16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_TABLES: [&[i32; 64]; 6] = [
    &MG_PAWN_TABLE,
    &MG_KNIGHT_TABLE,
    &MG_BISHOP_TABLE,
    &MG_ROOK_TABLE,
    &MG_QUEEN_TABLE,
    &MG_KING_TABLE,
];

const EG_TABLES: [&[i32; 64]; 6] = [
    &EG_PAWN_TABLE,
    &EG_KNIGHT_TABLE,
    &EG_BISHOP_TABLE,
    &EG_ROOK_TABLE,
    &EG_QUEEN_TABLE,
    &EG_KING_TABLE,
];

// Game phase increments
const GAME_PHASE_INC: [i32; 6] = [
    0, // Pawn
    1, // Knight
    1, // Bishop
    2, // Rook
    4, // Queen
    0, // King
];

// =============================================================================
// Pawn Structure Evaluation
// =============================================================================

static PASSED_MASKS: OnceLock<[[BitBoard; 64]; 2]> = OnceLock::new();
static ISOLATED_MASKS: OnceLock<[BitBoard; 64]> = OnceLock::new();

fn init_pawn_masks() {
    PASSED_MASKS.get_or_init(|| {
        let mut masks = [[BitBoard::EMPTY; 64]; 2];
        for c in 0..2 {
            for s in 0..64 {
                let rank = s / 8;
                let file = s % 8;
                let mut bb = 0u64;

                if c == 0 {
                    for r in (rank + 1)..8 {
                        bb |= 0xFF << (r * 8);
                    }
                } else {
                    for r in 0..rank {
                        bb |= 0xFF << (r * 8);
                    }
                }

                let mut file_mask = 0x0101_0101_0101_0101 << file;
                if file > 0 {
                    file_mask |= 0x0101_0101_0101_0101 << (file - 1);
                }
                if file < 7 {
                    file_mask |= 0x0101_0101_0101_0101 << (file + 1);
                }

                masks[c][s] = BitBoard(bb & file_mask);
            }
        }
        masks
    });

    ISOLATED_MASKS.get_or_init(|| {
        let mut masks = [BitBoard::EMPTY; 64];
        for s in 0..64 {
            let file = s % 8;
            let mut file_mask = 0;
            if file > 0 {
                file_mask |= 0x0101_0101_0101_0101 << (file - 1);
            }
            if file < 7 {
                file_mask |= 0x0101_0101_0101_0101 << (file + 1);
            }
            masks[s] = BitBoard(file_mask);
        }
        masks
    });
}

const PASSED_PAWN_BONUS: [i32; 8] = [0, 5, 10, 20, 35, 60, 100, 0];
const ISOLATED_PAWN_PENALTY: i32 = -15;
const DOUBLED_PAWN_PENALTY: i32 = -10;

fn eval_pawns(board: &Board) -> (i32, i32) {
    let mut score = [0; 2];
    init_pawn_masks();
    let passed_masks = PASSED_MASKS.get().unwrap();
    let isolated_masks = ISOLATED_MASKS.get().unwrap();

    for color_idx in 0..2 {
        let mut pawns = board.pieces[color_idx][Piece::Pawn.index()];
        let opp_pawns = board.pieces[color_idx ^ 1][Piece::Pawn.index()];

        while pawns.is_not_empty() {
            let sq = pawns.pop_lsb();
            let sq_idx = sq.0 as usize;

            // Passed pawn
            if (passed_masks[color_idx][sq_idx] & opp_pawns).is_empty() {
                let rank = if color_idx == 0 {
                    sq_idx / 8
                } else {
                    7 - (sq_idx / 8)
                };
                score[color_idx] += PASSED_PAWN_BONUS[rank];
            }

            // Isolated pawn
            let our_pawns = board.pieces[color_idx][Piece::Pawn.index()];
            if (isolated_masks[sq_idx] & our_pawns).is_empty() {
                score[color_idx] += ISOLATED_PAWN_PENALTY;
            }

            // Doubled pawn (penalize if there is a friendly pawn in front on the same file)
            let file_mask = BitBoard(0x0101_0101_0101_0101 << (sq_idx % 8));
            let forward_mask = if color_idx == 0 {
                let mut bb = 0;
                for r in (sq_idx / 8 + 1)..8 {
                    bb |= 0xFF << (r * 8);
                }
                BitBoard(bb)
            } else {
                let mut bb = 0;
                for r in 0..(sq_idx / 8) {
                    bb |= 0xFF << (r * 8);
                }
                BitBoard(bb)
            };

            if (forward_mask & file_mask & our_pawns).is_not_empty() {
                score[color_idx] += DOUBLED_PAWN_PENALTY;
            }
        }
    }

    (score[0], score[1])
}

// =============================================================================
// Piece Mobility, King Safety, Positional Bonuses
// =============================================================================

fn eval_advanced(board: &Board) -> (i32, i32, i32, i32) {
    let mut mg = [0; 2];
    let mut eg = [0; 2];

    let k_weights = [0, 2, 2, 3, 5, 0];
    let mut king_attackers = [0; 2];
    let mut king_attack_weight = [0; 2];

    // Safely get kings, fallback to a1 if missing (e.g. some malformed test boards)
    let w_king = board.pieces[0][Piece::King.index()].lsb();
    let b_king = board.pieces[1][Piece::King.index()].lsb();

    let w_king_zone = attacks::king_attacks(w_king);
    let b_king_zone = attacks::king_attacks(b_king);

    let occ = board.occupied;

    for color_idx in 0..2 {
        let us = board.by_color[color_idx];
        let opp_king_zone = if color_idx == 0 {
            b_king_zone
        } else {
            w_king_zone
        };

        let w_pawns = board.pieces[0][Piece::Pawn.index()];
        let b_pawns = board.pieces[1][Piece::Pawn.index()];
        let our_pawns = if color_idx == 0 { w_pawns } else { b_pawns };
        let their_pawns = if color_idx == 0 { b_pawns } else { w_pawns };

        // Bishop pair
        if board.pieces[color_idx][Piece::Bishop.index()].has_multiple() {
            mg[color_idx] += 30;
            eg[color_idx] += 30;
        }

        // Loop over non-pawn, non-king pieces for mobility and king safety
        for p_idx in 1..5 {
            let mut pieces = board.pieces[color_idx][p_idx];
            while pieces.is_not_empty() {
                let sq = pieces.pop_lsb();
                let attacks = match p_idx {
                    1 => attacks::knight_attacks(sq),
                    2 => attacks::bishop_attacks(sq, occ),
                    3 => attacks::rook_attacks(sq, occ),
                    4 => attacks::queen_attacks(sq, occ),
                    _ => BitBoard::EMPTY,
                };

                // Mobility
                let safe_moves = (attacks & !us).popcount() as i32;
                let mob_score = match p_idx {
                    1 => (safe_moves - 4) * 4,
                    2 => (safe_moves - 7) * 5,
                    3 => (safe_moves - 7) * 3,
                    4 => (safe_moves - 14) * 2,
                    _ => 0,
                };
                mg[color_idx] += mob_score;
                eg[color_idx] += mob_score;

                // King safety
                let king_attacks = attacks & opp_king_zone;
                if king_attacks.is_not_empty() {
                    king_attackers[color_idx] += 1;
                    king_attack_weight[color_idx] += k_weights[p_idx];
                }

                // Rook open files
                if p_idx == 3 {
                    let file_mask = BitBoard(0x0101_0101_0101_0101 << (sq.0 % 8));
                    if (file_mask & our_pawns).is_empty() {
                        if (file_mask & their_pawns).is_empty() {
                            mg[color_idx] += 25;
                            eg[color_idx] += 25;
                        } else {
                            mg[color_idx] += 12;
                            eg[color_idx] += 12;
                        }
                    }
                }
            }
        }
    }

    for color_idx in 0..2 {
        let weight = king_attack_weight[color_idx];
        if king_attackers[color_idx] >= 2 && weight > 0 {
            let penalty = (weight * weight) / 2;
            mg[color_idx ^ 1] -= penalty;
        }
    }

    (mg[0], mg[1], eg[0], eg[1])
}

// =============================================================================
// Main Evaluation
// =============================================================================

/// Tapered evaluation using PeSTO's tables.
/// Returns a score from the perspective of `side_to_move`.
/// When NNUE is loaded and `nnue` state is provided, uses the neural network;
/// otherwise falls back to the handcrafted PeSTO evaluation.
pub fn evaluate(board: &Board, nnue: Option<&crate::nnue::NnueState>, ply: usize) -> i32 {
    // Use NNUE if weights are loaded and state is available
    if crate::nnue::is_loaded() {
        if let Some(ns) = nnue {
            return ns.evaluate(ply, board.side_to_move);
        }
    }

    // Fallback: handcrafted PeSTO evaluation
    evaluate_hce(board)
}

/// Handcrafted evaluation (PeSTO + positional bonuses).
/// Always available as a fallback.
fn evaluate_hce(board: &Board) -> i32 {
    let mut mg_score = [0; 2];
    let mut eg_score = [0; 2];
    let mut game_phase = 0;

    for sq_idx in 0..64 {
        let sq = Square(sq_idx);
        if let Some((color, piece)) = board.piece_at(sq) {
            let p_idx = piece.index();
            let c_idx = color.index();

            // Map the square index correctly.
            // LERF (0=a1). PeSTO tables are oriented with 0=a8.
            // White pieces evaluate towards rank 8, so we flip their rank.
            // Black pieces evaluate towards rank 1, so they use the table as-is.
            let table_idx = if color == Color::White {
                (sq.0 ^ 56) as usize
            } else {
                sq.0 as usize
            };

            let mg_val = MG_VALUE[p_idx] + MG_TABLES[p_idx][table_idx];
            let eg_val = EG_VALUE[p_idx] + EG_TABLES[p_idx][table_idx];

            mg_score[c_idx] += mg_val;
            eg_score[c_idx] += eg_val;
            game_phase += GAME_PHASE_INC[p_idx];
        }
    }

    let (w_pawn_eval, b_pawn_eval) = eval_pawns(board);
    mg_score[0] += w_pawn_eval;
    eg_score[0] += w_pawn_eval;
    mg_score[1] += b_pawn_eval;
    eg_score[1] += b_pawn_eval;

    let (w_mg_adv, b_mg_adv, w_eg_adv, b_eg_adv) = eval_advanced(board);
    mg_score[0] += w_mg_adv;
    eg_score[0] += w_eg_adv;
    mg_score[1] += b_mg_adv;
    eg_score[1] += b_eg_adv;

    let stm = board.side_to_move.index();
    let opp = stm ^ 1;

    let mg_diff = mg_score[stm] - mg_score[opp];
    let eg_diff = eg_score[stm] - eg_score[opp];

    // Cap game phase at 24 (all material present)
    let mg_phase = game_phase.min(24);
    let eg_phase = 24 - mg_phase;

    // Interpolate
    (mg_diff * mg_phase + eg_diff * eg_phase) / 24
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_start_pos() {
        let board = Board::start_pos();
        let score = evaluate(&board, None, 0);
        // Start pos is perfectly symmetrical in PeSTO
        assert_eq!(score, 0);
    }
}
