// =============================================================================
// search.rs — Core search: negamax, quiescence, iterative deepening, pruning
// =============================================================================
//
// Implements the full search stack per the Phase 4 guide:
// - Negamax with alpha-beta
// - PVS (Principal Variation Search)
// - Iterative deepening with aspiration windows
// - Transposition table integration
// - Move ordering: TT move, MVV-LVA, killers, history, counter moves
// - Quiescence search with stand-pat + delta pruning
// - Pruning: RFP, NMP, LMR, futility, LMP
// - Check extensions
// - Draw detection (repetition + 50-move rule)

use std::io::Write;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::syzygy::SyzygyAdapter;
use pyrrhic_rs::{DtzProbeValue, TableBases, WdlProbeResult};

use crate::board::Board;
use crate::eval::{self, DRAW_SCORE, INFINITY, MATE_VALUE, is_mate_score, mate_in, mated_in};
use crate::tt::{NodeBound, TranspositionTable, adjust_mate_for_tt, adjust_mate_from_tt};
use crate::types::{BitBoard, Move, MoveList, Piece, Square};

// =============================================================================
// LMR table (precomputed)
// =============================================================================

static LMR_TABLE: OnceLock<[[i32; 64]; 64]> = OnceLock::new();

fn init_lmr() -> [[i32; 64]; 64] {
    let mut table = [[0i32; 64]; 64];
    for depth in 1..64usize {
        for moves_searched in 1..64usize {
            table[depth][moves_searched] =
                (0.77 + (depth as f64).ln() * (moves_searched as f64).ln() / 2.36) as i32;
        }
    }
    table
}

#[inline]
fn lmr_reduction(depth: usize, move_count: usize) -> i32 {
    LMR_TABLE.get_or_init(init_lmr)[depth.min(63)][move_count.min(63)]
}

// When a quiet move appears to improve the static eval for the side to move,
// reduce less aggressively ("improving node" idea).
const IMPROVING_EVAL_MARGIN_1: i32 = 40;
const IMPROVING_EVAL_MARGIN_2: i32 = 80;

// =============================================================================
// MVV-LVA (Most Valuable Victim – Least Valuable Attacker)
// =============================================================================

const MVV_LVA_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

#[inline]
fn mvv_lva_score(attacker: Piece, victim: Piece) -> i32 {
    MVV_LVA_VALUES[victim.index()] * 10 - MVV_LVA_VALUES[attacker.index()]
}

// =============================================================================
// SEE (Static Exchange Evaluation)
// =============================================================================

/// Piece values for SEE (centipawns).
const SEE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

/// Static Exchange Evaluation — evaluates the outcome of a capture exchange
/// on a single square without making actual moves.
///
/// Returns the material gain (positive) or loss (negative) for the side
/// initiating the capture.
fn see(board: &Board, mv: Move) -> i32 {
    use crate::attacks;

    let from = mv.from_sq();
    let to = mv.to_sq();

    // Determine the initial captured piece value
    let mut gain = if mv.is_en_passant() {
        SEE_VALUES[Piece::Pawn.index()]
    } else if let Some((_, victim)) = board.piece_at(to) {
        SEE_VALUES[victim.index()]
    } else {
        // Quiet move — SEE is 0
        return 0;
    };

    // If it's a promotion, the gain includes the promotion upgrade
    if mv.is_promotion() {
        if let Some(promo) = mv.promotion_piece() {
            gain += SEE_VALUES[promo.index()] - SEE_VALUES[Piece::Pawn.index()];
        }
    }

    // Determine the initial attacker
    let (attacker_color, attacker_piece) = match board.piece_at(from) {
        Some(pair) => pair,
        None => return 0,
    };

    // A stack of gains at each exchange step (max depth = 32 is more than enough)
    let mut gains = [0i32; 32];
    let mut depth = 0usize;
    gains[0] = gain;

    // Simulate occupancy changes: remove pieces as they capture
    let mut occ = board.occupied;
    // Remove the initial attacker from occupancy
    occ = BitBoard(occ.0 & !(1u64 << from.0));

    // For en passant, also remove the captured pawn
    if mv.is_en_passant() {
        let ep_cap_sq = crate::types::Square::new(to.file(), from.rank());
        occ = BitBoard(occ.0 & !(1u64 << ep_cap_sq.0));
    }

    let mut current_piece_value = SEE_VALUES[attacker_piece.index()];
    let mut side = attacker_color.flip(); // Opponent captures next

    loop {
        depth += 1;
        if depth >= 32 {
            break;
        }

        // The next capture gains the previous attacker's value
        gains[depth] = current_piece_value - gains[depth - 1];

        // Pruning: if the side to move can't improve even with a best-case capture,
        // the exchange is settled
        if (-gains[depth]).max(gains[depth - 1]) < 0 {
            break;
        }

        // Find cheapest attacker for `side` on square `to`
        let their_pieces = board.by_color[side.index()];
        let attackers_to_sq = get_all_attackers(board, to, occ) & their_pieces & occ;

        if attackers_to_sq.is_empty() {
            break; // No more attackers — exchange is over
        }

        // Pick the cheapest attacker (Pawn < Knight < Bishop < Rook < Queen < King)
        let (cheapest_sq, cheapest_piece) = find_cheapest_attacker(board, attackers_to_sq, side);

        current_piece_value = SEE_VALUES[cheapest_piece.index()];

        // Remove this piece from occupancy (it moves to `to`)
        occ = BitBoard(occ.0 & !(1u64 << cheapest_sq.0));

        // Switch sides
        side = side.flip();
    }

    // Negamax the gains stack
    while depth > 1 {
        depth -= 1;
        gains[depth] = -((-gains[depth]).max(gains[depth + 1]));
    }

    gains[0]
}

/// Get all pieces of any color that attack a given square, considering occupancy.
#[inline]
fn get_all_attackers(board: &Board, sq: Square, occ: BitBoard) -> BitBoard {
    use crate::attacks;

    let pawns_w = board.pieces[0][Piece::Pawn.index()];
    let pawns_b = board.pieces[1][Piece::Pawn.index()];
    let knights = board.pieces[0][Piece::Knight.index()] | board.pieces[1][Piece::Knight.index()];
    let bishops = board.pieces[0][Piece::Bishop.index()] | board.pieces[1][Piece::Bishop.index()];
    let rooks = board.pieces[0][Piece::Rook.index()] | board.pieces[1][Piece::Rook.index()];
    let queens = board.pieces[0][Piece::Queen.index()] | board.pieces[1][Piece::Queen.index()];
    let kings = board.pieces[0][Piece::King.index()] | board.pieces[1][Piece::King.index()];

    let diag = bishops | queens;
    let orth = rooks | queens;

    // Pawn attacks: a white pawn attacks sq if sq is in its attack set (and vice versa)
    let pawn_atk = (attacks::pawn_attacks(crate::types::Color::Black, sq) & pawns_w)
        | (attacks::pawn_attacks(crate::types::Color::White, sq) & pawns_b);

    pawn_atk
        | (attacks::knight_attacks(sq) & knights)
        | (attacks::bishop_attacks(sq, occ) & diag)
        | (attacks::rook_attacks(sq, occ) & orth)
        | (attacks::king_attacks(sq) & kings)
}

/// Find the cheapest attacker from a set of attacker squares for a given side.
#[inline]
fn find_cheapest_attacker(
    board: &Board,
    attackers: BitBoard,
    side: crate::types::Color,
) -> (Square, Piece) {
    let side_idx = side.index();
    // Check pieces in order of value: Pawn, Knight, Bishop, Rook, Queen, King
    for &piece in &Piece::ALL {
        let piece_bb = board.pieces[side_idx][piece.index()];
        let overlap = attackers & piece_bb;
        if overlap.is_not_empty() {
            return (overlap.lsb(), piece);
        }
    }
    // Should never reach here if attackers is non-empty
    (attackers.lsb(), Piece::King)
}

/// Quick check: is the SEE of this capture >= threshold?
/// Used for pruning decisions without computing the full SEE value.
#[inline]
fn see_ge(board: &Board, mv: Move, threshold: i32) -> bool {
    see(board, mv) >= threshold
}

// =============================================================================
// Move ordering score constants
// =============================================================================

const TT_MOVE_SCORE: i32 = 2_000_000;
const GOOD_CAPTURE_BASE: i32 = 1_000_000;
const QUEEN_PROMO_SCORE: i32 = 999_000;
const KILLER1_SCORE: i32 = 900_000;
const KILLER2_SCORE: i32 = 800_000;
const COUNTER_MOVE_SCORE: i32 = 700_000;
const CONT_HISTORY_SCORE: i32 = 600_000;
const BAD_CAPTURE_BASE: i32 = -1_000_000;
const HISTORY_PRUNE_DEPTH_MAX: i32 = 5;
const HISTORY_PRUNE_SCORE: i32 = -4_000;

// =============================================================================
// SearchState — per-search mutable state
// =============================================================================

pub struct SearchState {
    // --- Stop control ---
    pub stop: Arc<AtomicBool>,
    pub syzygy: Option<Arc<TableBases<SyzygyAdapter>>>,

    // --- Thread identity ---
    pub thread_id: usize,

    // --- Per-search heuristics ---
    pub killers: [[Move; 2]; 128],
    pub history: [[i32; 64]; 64],
    pub counter_moves: [[Move; 64]; 64],
    /// Continuation history: [prev_piece][prev_to][to]
    pub cont_history: [[[i32; 64]; 64]; 6],

    // --- Stats ---
    pub nodes: u64,
    pub sel_depth: u8,

    // --- Timing ---
    pub start_time: Instant,
    pub soft_limit_ms: u64,
    pub hard_limit_ms: u64,

    // --- Node limit ---
    pub max_nodes: u64,

    // --- Root info ---
    pub max_depth: i32,
    pub root_depth: i32,
    pub best_move: Move,
    pub best_score: i32,

    // --- PV table (triangular) ---
    pub pv_length: [usize; 128],
    pub pv_table: [[Move; 128]; 128],

    // --- Position history for repetition detection ---
    pub position_history: Vec<u64>,

    // Static eval cache by ply for improving-node decisions.
    pub static_eval_stack: [i32; 128],

    // --- NNUE accumulator stack ---
    pub nnue: crate::nnue::NnueState,
}

impl SearchState {
    pub fn new(stop: Arc<AtomicBool>, syzygy: Option<Arc<TableBases<SyzygyAdapter>>>) -> Self {
        Self {
            stop,
            syzygy,
            thread_id: 0,
            killers: [[Move::NULL; 2]; 128],
            history: [[0i32; 64]; 64],
            counter_moves: [[Move::NULL; 64]; 64],
            cont_history: [[[0i32; 64]; 64]; 6],
            nodes: 0,
            sel_depth: 0,
            start_time: Instant::now(),
            soft_limit_ms: u64::MAX,
            hard_limit_ms: u64::MAX,
            max_nodes: u64::MAX,
            max_depth: 128,
            root_depth: 0,
            best_move: Move::NULL,
            best_score: -INFINITY,
            pv_length: [0; 128],
            pv_table: [[Move::NULL; 128]; 128],
            position_history: Vec::with_capacity(512),
            static_eval_stack: [0; 128],
            nnue: crate::nnue::NnueState::new(),
        }
    }

    #[inline]
    fn check_stop(&mut self) {
        if self.nodes & 4095 == 0 {
            if self.nodes >= self.max_nodes {
                self.stop.store(true, Ordering::Relaxed);
                return;
            }
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            if elapsed >= self.hard_limit_ms {
                self.stop.store(true, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn update_killers(&mut self, mv: Move, ply: usize) {
        if ply < 128 && self.killers[ply][0] != mv {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = mv;
        }
    }

    fn update_history(&mut self, mv: Move, depth: i32) {
        let from = mv.from_sq().0 as usize;
        let to = mv.to_sq().0 as usize;
        let bonus = depth * depth;
        self.history[from][to] += bonus;
        if self.history[from][to].abs() > 32_767 {
            // Gravity: halve all values to prevent overflow
            for row in &mut self.history {
                for v in row {
                    *v /= 2;
                }
            }
        }
    }

    fn penalize_history(&mut self, mv: Move, depth: i32) {
        let from = mv.from_sq().0 as usize;
        let to = mv.to_sq().0 as usize;
        let penalty = depth * depth;
        self.history[from][to] -= penalty;
        if self.history[from][to].abs() > 32_767 {
            for row in &mut self.history {
                for v in row {
                    *v /= 2;
                }
            }
        }
    }

    fn update_cont_history(&mut self, board: &Board, prev_mv: Move, mv: Move, depth: i32) {
        if prev_mv == Move::NULL || mv.is_capture() || mv.is_promotion() || mv.is_en_passant() {
            return;
        }
        if let Some((_, prev_piece)) = board.piece_at(prev_mv.to_sq()) {
            let pf = prev_mv.to_sq().0 as usize;
            let tf = mv.to_sq().0 as usize;
            let bonus = depth * depth;
            self.cont_history[prev_piece.index()][pf][tf] += bonus;
            if self.cont_history[prev_piece.index()][pf][tf].abs() > 32_767 {
                for plane in &mut self.cont_history {
                    for row in plane {
                        for v in row {
                            *v /= 2;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Draw detection
// =============================================================================

/// Check for twofold repetition within the position history.
fn is_repetition(hash: u64, halfmove: u8, history: &[u64]) -> bool {
    let len = history.len();
    let lookback = (halfmove as usize).min(len);

    // Check every 2nd position (same side to move) going back
    let mut i = 2usize;
    while i <= lookback && i <= len {
        if history[len - i] == hash {
            return true;
        }
        i += 2;
    }
    false
}

fn is_draw(board: &Board, history: &[u64]) -> bool {
    // 50-move rule
    if board.halfmove_clock >= 100 {
        return true;
    }
    // Repetition
    is_repetition(board.hash, board.halfmove_clock, history)
}

// =============================================================================
// PV table
// =============================================================================

#[inline]
fn update_pv(ss: &mut SearchState, ply: usize, mv: Move) {
    ss.pv_table[ply][ply] = mv;
    let child_len = if ply + 1 < 128 {
        ss.pv_length[ply + 1]
    } else {
        ply + 1
    };
    for i in (ply + 1)..child_len {
        ss.pv_table[ply][i] = ss.pv_table[ply + 1][i];
    }
    ss.pv_length[ply] = child_len;
}

// =============================================================================
// Move scoring
// =============================================================================

fn score_moves(
    board: &Board,
    moves: &MoveList,
    scores: &mut [i32; 256],
    tt_move: Move,
    killers: &[Move; 2],
    history: &[[i32; 64]; 64],
    counter_mv: Move,
    cont_hist: &[[[i32; 64]; 64]; 6],
    prev_move: Move,
) {
    for i in 0..moves.len() {
        let mv = moves[i];

        if mv.pack_16() == tt_move.pack_16() && tt_move != Move::NULL {
            scores[i] = TT_MOVE_SCORE;
        } else if mv.is_capture() || mv.is_en_passant() {
            // Use SEE to classify captures as good or bad
            let see_val = see(board, mv);
            if see_val >= 0 {
                // Good capture: base + SEE value (ensures winning captures sort highest)
                scores[i] = GOOD_CAPTURE_BASE + see_val;
            } else {
                // Bad/losing capture: sort below quiet moves
                scores[i] = BAD_CAPTURE_BASE + see_val;
            }
        } else if mv.is_promotion() {
            scores[i] = QUEEN_PROMO_SCORE;
        } else if mv.pack_16() == killers[0].pack_16() && killers[0] != Move::NULL {
            scores[i] = KILLER1_SCORE;
        } else if mv.pack_16() == killers[1].pack_16() && killers[1] != Move::NULL {
            scores[i] = KILLER2_SCORE;
        } else if mv.pack_16() == counter_mv.pack_16() && counter_mv != Move::NULL {
            scores[i] = COUNTER_MOVE_SCORE;
        } else {
            let mut h = history[mv.from_sq().0 as usize][mv.to_sq().0 as usize];
            if prev_move != Move::NULL {
                if let Some((_, prev_piece)) = board.piece_at(prev_move.to_sq()) {
                    let ch = cont_hist[prev_piece.index()][prev_move.to_sq().0 as usize]
                        [mv.to_sq().0 as usize];
                    if ch > h {
                        h = ch;
                    }
                }
            }
            scores[i] = if h > 0 { CONT_HISTORY_SCORE + h } else { h };
        }
    }
}

/// Partial sort: swap the highest-scoring remaining move into position `start`.
#[inline]
fn move_stage(score: i32) -> i32 {
    if score >= TT_MOVE_SCORE {
        0
    } else if score >= GOOD_CAPTURE_BASE {
        1
    } else if score >= KILLER1_SCORE {
        2
    } else if score >= CONT_HISTORY_SCORE {
        3
    } else if score >= 0 {
        4
    } else if score > BAD_CAPTURE_BASE {
        5
    } else {
        6
    }
}

/// Staged picker: TT move, good captures, killers/counter, quiets, bad captures.
#[inline]
fn pick_next(moves: &mut MoveList, scores: &mut [i32; 256], start: usize) {
    if start + 1 >= moves.len() {
        return;
    }
    let mut best_idx = start;
    for i in (start + 1)..moves.len() {
        let stage_i = move_stage(scores[i]);
        let stage_best = move_stage(scores[best_idx]);
        if stage_i < stage_best || (stage_i == stage_best && scores[i] > scores[best_idx]) {
            best_idx = i;
        }
    }
    if best_idx != start {
        // Swap moves
        let tmp = moves[start];
        moves[start] = moves[best_idx];
        moves[best_idx] = tmp;
        scores.swap(start, best_idx);
    }
}

// =============================================================================
// Score captures for quiescence
// =============================================================================

fn score_captures(board: &Board, moves: &MoveList, scores: &mut [i32; 256]) {
    for i in 0..moves.len() {
        let mv = moves[i];
        if mv.is_capture() || mv.is_en_passant() {
            let see_val = see(board, mv);
            if see_val >= 0 {
                scores[i] = GOOD_CAPTURE_BASE + see_val;
            } else {
                scores[i] = BAD_CAPTURE_BASE + see_val;
            }
        } else if mv.is_promotion() {
            scores[i] = QUEEN_PROMO_SCORE;
        } else {
            scores[i] = 0;
        }
    }
}

// =============================================================================
// Iterative Deepening
// =============================================================================

pub fn iterative_deepening(
    board: &mut Board,
    ss: &mut SearchState,
    tt: &TranspositionTable,
) -> Move {
    let mut best_move = Move::NULL;
    let mut prev_score = 0i32;

    // --- Syzygy DTZ Root Probe ---
    if let Some(ref tb) = ss.syzygy {
        if board.occupied.popcount() <= tb.max_pieces() && board.castling_rights.0 == 0 {
            let ep = board.en_passant_sq.map(|s| s.0 as u32).unwrap_or(0);
            // Must be the only thread using the tablebase to call probe_root
            if let Ok(dtz_result) = tb.probe_root(
                board.by_color[0].0,
                board.by_color[1].0,
                board.bb(crate::types::Color::White, Piece::King).0
                    | board.bb(crate::types::Color::Black, Piece::King).0,
                board.bb(crate::types::Color::White, Piece::Queen).0
                    | board.bb(crate::types::Color::Black, Piece::Queen).0,
                board.bb(crate::types::Color::White, Piece::Rook).0
                    | board.bb(crate::types::Color::Black, Piece::Rook).0,
                board.bb(crate::types::Color::White, Piece::Bishop).0
                    | board.bb(crate::types::Color::Black, Piece::Bishop).0,
                board.bb(crate::types::Color::White, Piece::Knight).0
                    | board.bb(crate::types::Color::Black, Piece::Knight).0,
                board.bb(crate::types::Color::White, Piece::Pawn).0
                    | board.bb(crate::types::Color::Black, Piece::Pawn).0,
                board.halfmove_clock as u32,
                ep,
                board.side_to_move == crate::types::Color::White,
            ) {
                // If it finds a perfect move, pick the best one and return
                // dtz_result.moves contains the moves and their DTZ values
                let mut best_dtz_move = None;
                for i in 0..dtz_result.num_moves {
                    if let DtzProbeValue::DtzResult(res) = dtz_result.moves[i] {
                        if let DtzProbeValue::DtzResult(root_res) = dtz_result.root {
                            if res.wdl == root_res.wdl {
                                // Match the Syzygy move to our Move type
                                let mut flags = Move::FLAG_QUIET;
                                if res.ep {
                                    flags = Move::FLAG_EP_CAPTURE;
                                } else if board
                                    .piece_at(crate::types::Square(res.to_square))
                                    .is_some()
                                {
                                    flags = Move::FLAG_CAPTURE;
                                }

                                if res.promotion != pyrrhic_rs::Piece::Pawn {
                                    flags = match res.promotion {
                                        pyrrhic_rs::Piece::Queen => Move::FLAG_PROMO_Q,
                                        pyrrhic_rs::Piece::Rook => Move::FLAG_PROMO_R,
                                        pyrrhic_rs::Piece::Bishop => Move::FLAG_PROMO_B,
                                        pyrrhic_rs::Piece::Knight => Move::FLAG_PROMO_N,
                                        _ => Move::FLAG_QUIET,
                                    };
                                }
                                best_dtz_move = Some(Move::new(
                                    crate::types::Square(res.from_square),
                                    crate::types::Square(res.to_square),
                                    flags,
                                ));
                                break;
                            }
                        }
                    }
                }

                if let Some(mv) = best_dtz_move {
                    let score = match dtz_result.root {
                        DtzProbeValue::DtzResult(res) => match res.wdl {
                            WdlProbeResult::Win => MATE_VALUE - 100,
                            WdlProbeResult::Loss => -MATE_VALUE + 100,
                            _ => 0,
                        },
                        _ => 0,
                    };
                    ss.nodes = 1;
                    print_info(ss, 1, score, tt);
                    return mv;
                }
            }
        }
    }

    // Initialize LMR table
    let _ = LMR_TABLE.get_or_init(init_lmr);

    // Helper threads (id > 0) start at a higher depth to diversify
    let start_depth = if ss.thread_id > 0 {
        (1 + (ss.thread_id as i32 % 4)).min(ss.max_depth)
    } else {
        1
    };

    for depth in start_depth..=ss.max_depth {
        ss.root_depth = depth;
        ss.pv_length[0] = 0;

        // Refresh NNUE accumulator at root
        ss.nnue.refresh(0, board);

        let score = aspiration_search(board, ss, tt, depth, prev_score);

        if ss.is_stopped() {
            break;
        }

        // This iteration completed — update best move
        if ss.pv_length[0] > 0 {
            best_move = ss.pv_table[0][0];
        }
        ss.best_move = best_move;
        ss.best_score = score;
        prev_score = score;

        // Only thread 0 prints info to the GUI
        if ss.thread_id == 0 {
            print_info(ss, depth, score, tt);
        }

        // Soft time limit: don't start next iteration if we've used >50% of soft time
        let elapsed = ss.start_time.elapsed().as_millis() as u64;
        if elapsed >= ss.soft_limit_ms {
            break;
        }

        // Early exit on forced mate
        if is_mate_score(score) && (MATE_VALUE - score.abs()) <= depth {
            break;
        }
    }

    best_move
}

// =============================================================================
// Aspiration Windows
// =============================================================================

fn aspiration_search(
    board: &mut Board,
    ss: &mut SearchState,
    tt: &TranspositionTable,
    depth: i32,
    prev_score: i32,
) -> i32 {
    // First few depths: search with full window
    if depth < 5 {
        return negamax(board, ss, tt, depth, 0, -INFINITY, INFINITY, Move::NULL);
    }

    let mut delta = (15 + depth * 8 + prev_score.abs() / 32).clamp(25, 150);
    let mut alpha = (prev_score - delta).max(-INFINITY);
    let mut beta = (prev_score + delta).min(INFINITY);
    let mut fail_highs = 0;
    let mut fail_lows = 0;

    loop {
        let score = negamax(board, ss, tt, depth, 0, alpha, beta, Move::NULL);

        if ss.is_stopped() {
            return score;
        }

        if score <= alpha {
            // Failed low: widen more aggressively on the low side.
            fail_lows += 1;
            beta = ((alpha + beta) / 2 + delta / 2).min(INFINITY);
            alpha = (score - delta * (fail_lows + 2)).max(-INFINITY);
            delta = (delta * 2).min(600);
        } else if score >= beta {
            // Failed high: widen more aggressively on the high side.
            fail_highs += 1;
            alpha = ((alpha + beta) / 2 - delta / 2).max(-INFINITY);
            beta = (score + delta * (fail_highs + 2)).min(INFINITY);
            delta = (delta * 2).min(600);
        } else {
            return score;
        }

        // After multiple failures, fall back to infinite window
        if delta > 500 {
            return negamax(board, ss, tt, depth, 0, -INFINITY, INFINITY, Move::NULL);
        }
    }
}

// =============================================================================
// Singular extension — search siblings excluding TT move
// =============================================================================

fn is_singular(
    board: &mut Board,
    ss: &mut SearchState,
    tt: &TranspositionTable,
    depth: i32,
    ply: usize,
    se_beta: i32,
    exclude: Move,
    prev_move: Move,
) -> bool {
    let us = board.side_to_move;
    let mut moves = board.generate_pseudo_legal_moves();
    let killers = if ply < 128 {
        ss.killers[ply]
    } else {
        [Move::NULL; 2]
    };
    let counter = if prev_move != Move::NULL {
        ss.counter_moves[prev_move.from_sq().0 as usize][prev_move.to_sq().0 as usize]
    } else {
        Move::NULL
    };
    let mut scores = [0i32; 256];
    score_moves(
        board,
        &moves,
        &mut scores,
        Move::NULL,
        &killers,
        &ss.history,
        counter,
        &ss.cont_history,
        prev_move,
    );

    let sing_depth = depth - 3;
    if sing_depth < 1 {
        return false;
    }

    for move_idx in 0..moves.len() {
        pick_next(&mut moves, &mut scores, move_idx);
        let mv = moves[move_idx];
        if mv.pack_16() == exclude.pack_16() {
            continue;
        }
        if mv.is_capture() || mv.is_en_passant() {
            if see(board, mv) < -80 {
                continue;
            }
        }

        let undo = board.do_move(mv);
        if board.in_check(us) {
            board.undo_move(undo);
            continue;
        }

        ss.nnue.apply_move(ply + 1, board, undo);
        ss.position_history.push(board.hash);
        let score = -negamax(
            board,
            ss,
            tt,
            sing_depth,
            ply + 1,
            -se_beta,
            -se_beta + 1,
            mv,
        );
        ss.position_history.pop();
        board.undo_move(undo);

        if ss.is_stopped() {
            return false;
        }
        if score >= se_beta {
            return false;
        }
    }
    true
}

// =============================================================================
// Negamax with Alpha-Beta, PVS, TT, and all pruning
// =============================================================================

fn negamax(
    board: &mut Board,
    ss: &mut SearchState,
    tt: &TranspositionTable,
    mut depth: i32,
    ply: usize,
    mut alpha: i32,
    beta: i32,
    prev_move: Move,
) -> i32 {
    let is_pv = (beta - alpha) > 1;
    let in_check = board.in_check(board.side_to_move);

    // Check extension: don't reduce search when in check
    if in_check {
        depth += 1;
    }

    // PV bookkeeping
    if ply < 128 {
        ss.pv_length[ply] = ply;
    }

    // Quiescence at horizon
    if depth <= 0 {
        return quiescence(board, ss, ply, alpha, beta);
    }

    // Max ply guard
    if ply >= 127 {
        return eval::evaluate(board, Some(&ss.nnue), ply);
    }

    // Draw detection (skip root)
    if ply > 0 && is_draw(board, &ss.position_history) {
        return DRAW_SCORE;
    }

    ss.nodes += 1;
    ss.check_stop();
    if ss.is_stopped() {
        return 0;
    }

    // =========================================================================
    // Syzygy WDL Probe
    // =========================================================================

    if let Some(ref tb) = ss.syzygy {
        if !in_check && depth >= 1 && board.occupied.popcount() <= tb.max_pieces() {
            let halfmove = board.halfmove_clock as u32;
            let ep = board.en_passant_sq.map(|s| s.0 as u32).unwrap_or(0);
            if halfmove == 0 && (board.castling_rights.0 == 0) {
                if let Ok(wdl) = tb.probe_wdl(
                    board.by_color[0].0,
                    board.by_color[1].0,
                    board.bb(crate::types::Color::White, Piece::King).0
                        | board.bb(crate::types::Color::Black, Piece::King).0,
                    board.bb(crate::types::Color::White, Piece::Queen).0
                        | board.bb(crate::types::Color::Black, Piece::Queen).0,
                    board.bb(crate::types::Color::White, Piece::Rook).0
                        | board.bb(crate::types::Color::Black, Piece::Rook).0,
                    board.bb(crate::types::Color::White, Piece::Bishop).0
                        | board.bb(crate::types::Color::Black, Piece::Bishop).0,
                    board.bb(crate::types::Color::White, Piece::Knight).0
                        | board.bb(crate::types::Color::Black, Piece::Knight).0,
                    board.bb(crate::types::Color::White, Piece::Pawn).0
                        | board.bb(crate::types::Color::Black, Piece::Pawn).0,
                    ep,
                    board.side_to_move == crate::types::Color::White,
                ) {
                    let tb_score = match wdl {
                        WdlProbeResult::Win => MATE_VALUE - 100 - ply as i32,
                        WdlProbeResult::Loss => -MATE_VALUE + 100 + ply as i32,
                        WdlProbeResult::Draw
                        | WdlProbeResult::CursedWin
                        | WdlProbeResult::BlessedLoss => 0,
                    };

                    // We can return immediately if it's a bound we can use
                    if wdl == WdlProbeResult::Win && tb_score >= beta {
                        return tb_score;
                    }
                    if wdl == WdlProbeResult::Loss && tb_score <= alpha {
                        return tb_score;
                    }
                    if (wdl == WdlProbeResult::Draw
                        || wdl == WdlProbeResult::CursedWin
                        || wdl == WdlProbeResult::BlessedLoss)
                    {
                        if tb_score >= beta {
                            return tb_score;
                        }
                        if tb_score <= alpha {
                            return tb_score;
                        }
                    }
                }
            }
        }
    }

    let original_alpha = alpha;

    // =========================================================================
    // TT Probe
    // =========================================================================

    let mut tt_move = Move::NULL;
    let mut tt_depth = 0i32;
    let mut tt_score_for_se = 0i32;
    let mut tt_bound_for_se = NodeBound::Upper;
    if let Some(entry) = tt.probe(board.hash) {
        tt_move = Move::unpack_16(entry.mv);
        tt_depth = entry.depth as i32;
        let tt_score = adjust_mate_from_tt(entry.score as i32, ply);
        tt_score_for_se = tt_score;
        tt_bound_for_se = NodeBound::from_u8(entry.bound);

        // Use TT cutoff only if depth sufficient and not at root
        if ply > 0 && !is_pv && entry.depth as i32 >= depth {
            match tt_bound_for_se {
                NodeBound::Exact => return tt_score,
                NodeBound::Lower => {
                    if tt_score >= beta {
                        return tt_score;
                    }
                }
                NodeBound::Upper => {
                    if tt_score <= alpha {
                        return tt_score;
                    }
                }
            }
        }
    }

    // =========================================================================
    // Internal Iterative Deepening (IID)
    // =========================================================================
    // If we have no TT move at a PV node with sufficient depth, do a quick
    // shallow search to find a reasonable move for ordering.

    if is_pv && tt_move == Move::NULL && depth >= 6 {
        negamax(board, ss, tt, depth - 4, ply, alpha, beta, prev_move);
        if let Some(entry) = tt.probe(board.hash) {
            tt_move = Move::unpack_16(entry.mv);
            tt_depth = entry.depth as i32;
            tt_score_for_se = adjust_mate_from_tt(entry.score as i32, ply);
            tt_bound_for_se = NodeBound::from_u8(entry.bound);
        }
    }

    let static_eval = eval::evaluate(board, Some(&ss.nnue), ply);
    if ply < 128 {
        ss.static_eval_stack[ply] = static_eval;
    }
    let improving = if ply >= 2 {
        static_eval > ss.static_eval_stack[ply - 2]
    } else {
        false
    };

    // =========================================================================
    // Razoring — forward prune into quiescence when far below alpha
    // =========================================================================

    if !is_pv && !in_check && depth >= 1 && depth <= 3 {
        let razor_margin = 200 * depth + if improving { 40 } else { 0 };
        if static_eval + razor_margin < alpha {
            return quiescence(board, ss, ply, alpha, beta);
        }
    }

    // =========================================================================
    // Reverse Futility Pruning (RFP / Static Null Move Pruning)
    // =========================================================================

    if !is_pv && !in_check && depth >= 1 && depth <= 8 {
        let rfp_margin = 80 * depth + if improving { 20 } else { 0 };
        if static_eval - rfp_margin >= beta {
            return static_eval;
        }
    }

    // =========================================================================
    // Null Move Pruning (NMP)
    // =========================================================================

    let us = board.side_to_move;
    let has_pieces = (board.pieces[us.index()][Piece::Knight.index()]
        | board.pieces[us.index()][Piece::Bishop.index()]
        | board.pieces[us.index()][Piece::Rook.index()]
        | board.pieces[us.index()][Piece::Queen.index()])
    .is_not_empty();

    if !is_pv && !in_check && has_pieces && depth >= 3 && ply > 0 {
        let r = 3 + depth / 6 + if !improving { 1 } else { 0 };
        let null_undo = board.do_null_move();
        ss.nnue.copy_ply(ply + 1);
        ss.position_history.push(board.hash);
        let null_score = -negamax(
            board,
            ss,
            tt,
            depth - 1 - r,
            ply + 1,
            -beta,
            -beta + 1,
            Move::NULL,
        );
        ss.position_history.pop();
        board.undo_null_move(null_undo);

        if null_score >= beta {
            if is_mate_score(null_score) {
                return beta;
            }
            return null_score;
        }
    }

    // =========================================================================
    // ProbCut — reduced-depth cutoff search on strong captures
    // =========================================================================

    if !is_pv && !in_check && depth >= 3 && static_eval >= beta + if improving { 120 } else { 90 } {
        let prob_depth = depth - 4;
        if prob_depth >= 1 {
            let prob_beta = beta + 1;
            let us = board.side_to_move;
            let mut caps = board.generate_captures();
            let mut cap_scores = [0i32; 256];
            score_captures(board, &caps, &mut cap_scores);
            for ci in 0..caps.len().min(8) {
                pick_next(&mut caps, &mut cap_scores, ci);
                let cap_mv = caps[ci];
                if see(board, cap_mv) < 1 {
                    continue;
                }
                let undo = board.do_move(cap_mv);
                if board.in_check(us) {
                    board.undo_move(undo);
                    continue;
                }
                ss.nnue.apply_move(ply + 1, board, undo);
                ss.position_history.push(board.hash);
                let prob_score = -negamax(
                    board,
                    ss,
                    tt,
                    prob_depth,
                    ply + 1,
                    -prob_beta,
                    -prob_beta + 1,
                    cap_mv,
                );
                ss.position_history.pop();
                board.undo_move(undo);
                if prob_score >= prob_beta {
                    return beta;
                }
            }
        }
    }

    // =========================================================================
    // Generate and score moves
    // =========================================================================

    let mut moves = board.generate_pseudo_legal_moves();
    let killers = if ply < 128 {
        ss.killers[ply]
    } else {
        [Move::NULL; 2]
    };
    let counter = if prev_move != Move::NULL {
        ss.counter_moves[prev_move.from_sq().0 as usize][prev_move.to_sq().0 as usize]
    } else {
        Move::NULL
    };
    let mut scores = [0i32; 256];
    score_moves(
        board,
        &moves,
        &mut scores,
        tt_move,
        &killers,
        &ss.history,
        counter,
        &ss.cont_history,
        prev_move,
    );

    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let mut legal_moves = 0usize;
    let mut quiets_tried: Vec<Move> = Vec::new();

    // LMP move count thresholds by depth
    const LMP_THRESHOLD: [usize; 9] = [0, 5, 10, 20, 40, 60, 80, 100, 120];

    // =========================================================================
    // Move loop
    // =========================================================================

    for move_idx in 0..moves.len() {
        pick_next(&mut moves, &mut scores, move_idx);
        let mv = moves[move_idx];
        let capture_see = if mv.is_capture() || mv.is_en_passant() {
            Some(see(board, mv))
        } else {
            None
        };

        if !is_pv && !in_check && depth <= 6 {
            if let Some(see_val) = capture_see {
                // Prune clearly losing late captures in shallow non-PV nodes.
                if move_idx > 0 && see_val < -(90 * depth) {
                    continue;
                }
            }
        }

        let mut extension = 0i32;
        if !in_check
            && depth >= 8
            && ply > 0
            && mv.pack_16() == tt_move.pack_16()
            && tt_move != Move::NULL
            && tt_depth >= depth - 3
            && matches!(tt_bound_for_se, NodeBound::Lower | NodeBound::Exact)
            && !is_mate_score(tt_score_for_se)
            && tt_score_for_se >= static_eval + 80
        {
            let se_beta = tt_score_for_se - 2 * depth;
            if is_singular(board, ss, tt, depth, ply, se_beta, tt_move, prev_move) {
                extension = if tt_score_for_se >= beta + 120 && depth >= 10 {
                    2
                } else {
                    1
                };
            }
        }

        let undo = board.do_move(mv);
        if board.in_check(us) {
            board.undo_move(undo);
            continue;
        }

        ss.nnue.apply_move(ply + 1, board, undo);

        legal_moves += 1;
        let is_quiet = !mv.is_capture() && !mv.is_promotion() && !mv.is_en_passant();
        let child_in_check = board.in_check(board.side_to_move);

        if !is_pv && !in_check && depth == 1 && is_quiet && !child_in_check {
            if static_eval + 100 <= alpha {
                board.undo_move(undo);
                continue;
            }
        }

        if !is_pv && !in_check && depth <= 4 && is_quiet && !child_in_check {
            let threshold = if (depth as usize) < LMP_THRESHOLD.len() {
                LMP_THRESHOLD[depth as usize]
            } else {
                256
            };
            if legal_moves > threshold {
                board.undo_move(undo);
                continue;
            }
        }

        if !is_pv && !in_check && is_quiet && !child_in_check && depth <= HISTORY_PRUNE_DEPTH_MAX {
            let hist = ss.history[mv.from_sq().0 as usize][mv.to_sq().0 as usize];
            let late_quiet = legal_moves > (8 + (depth as usize * 4));
            if hist <= HISTORY_PRUNE_SCORE
                && late_quiet
                && static_eval + if improving { 60 } else { 90 } <= alpha
            {
                board.undo_move(undo);
                continue;
            }
        }

        if is_quiet {
            quiets_tried.push(mv);
        }

        ss.position_history.push(board.hash);

        let score;
        if legal_moves == 1 {
            score = -negamax(
                board,
                ss,
                tt,
                depth - 1 + extension,
                ply + 1,
                -beta,
                -alpha,
                mv,
            );
        } else {
            let mut reduction = 0i32;

            if legal_moves >= 3 && depth >= 3 && !in_check && !child_in_check && is_quiet {
                reduction = lmr_reduction(depth as usize, legal_moves);
                let hist = ss.history[mv.from_sq().0 as usize][mv.to_sq().0 as usize];
                if hist < 0 {
                    reduction += 1;
                }
                if mv.pack_16() == killers[0].pack_16() || mv.pack_16() == killers[1].pack_16() {
                    reduction -= 1;
                }
                if is_pv {
                    reduction -= 1;
                }

                // "Improving node" adjustment:
                // - parent `static_eval` is from the current side-to-move perspective (`us`)
                // - child `eval::evaluate` is from the *next* side-to-move perspective (opponent),
                //   so compare using `-child_static_eval` as an estimate for our perspective.
                let child_static_eval = eval::evaluate(board, Some(&ss.nnue), ply + 1);
                let improve_for_us = (-child_static_eval) - static_eval;

                if (-child_static_eval) >= alpha - 25 {
                    // Seems already promising: don't reduce.
                    reduction = 0;
                } else if improve_for_us >= IMPROVING_EVAL_MARGIN_2 {
                    reduction = (reduction - 2).max(0);
                } else if improve_for_us >= IMPROVING_EVAL_MARGIN_1 {
                    reduction = (reduction - 1).max(0);
                }

                reduction = reduction.clamp(0, depth - 2);
            }

            let mut null_score = -negamax(
                board,
                ss,
                tt,
                depth - 1 + extension - reduction,
                ply + 1,
                -alpha - 1,
                -alpha,
                mv,
            );

            if null_score > alpha && reduction > 0 {
                null_score = -negamax(
                    board,
                    ss,
                    tt,
                    depth - 1 + extension,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    mv,
                );
            }

            if null_score > alpha && null_score < beta && is_pv {
                score = -negamax(
                    board,
                    ss,
                    tt,
                    depth - 1 + extension,
                    ply + 1,
                    -beta,
                    -alpha,
                    mv,
                );
            } else {
                score = null_score;
            }
        }

        ss.position_history.pop();
        board.undo_move(undo);

        if ss.is_stopped() {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = mv;

            if score > alpha {
                alpha = score;
                if ply < 128 {
                    update_pv(ss, ply, mv);
                }

                if score >= beta {
                    if is_quiet {
                        ss.update_killers(mv, ply);
                        ss.update_history(mv, depth);
                        ss.update_cont_history(board, prev_move, mv, depth);
                        if prev_move != Move::NULL {
                            ss.counter_moves[prev_move.from_sq().0 as usize]
                                [prev_move.to_sq().0 as usize] = mv;
                        }
                        for &q in &quiets_tried {
                            if q != mv {
                                ss.penalize_history(q, depth);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // =========================================================================
    // No legal moves: checkmate or stalemate
    // =========================================================================

    if legal_moves == 0 {
        return if in_check { mated_in(ply) } else { DRAW_SCORE };
    }

    // =========================================================================
    // TT Store
    // =========================================================================

    let bound = if best_score >= beta {
        NodeBound::Lower
    } else if best_score > original_alpha {
        NodeBound::Exact
    } else {
        NodeBound::Upper
    };

    tt.store(
        board.hash,
        adjust_mate_for_tt(best_score, ply) as i16,
        best_move,
        depth as i8,
        bound,
    );

    best_score
}

// =============================================================================
// Quiescence Search
// =============================================================================

fn quiescence(
    board: &mut Board,
    ss: &mut SearchState,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    ss.nodes += 1;
    ss.sel_depth = ss.sel_depth.max(ply as u8);

    if ss.is_stopped() {
        return 0;
    }

    let in_check = board.in_check(board.side_to_move);

    // Compute stand-pat once; -INFINITY when in check (no stand-pat allowed)
    let stand_pat = if in_check {
        -INFINITY
    } else {
        eval::evaluate(board, Some(&ss.nnue), ply)
    };

    // When not in check: stand-pat pruning
    if !in_check {
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    // Generate moves: all moves if in check, captures only otherwise
    let mut moves = if in_check {
        board.generate_pseudo_legal_moves()
    } else {
        board.generate_captures()
    };

    // Score captures by SEE
    let mut scores = [0i32; 256];
    score_captures(board, &moves, &mut scores);

    let mut legal_moves = 0usize;
    let us = board.side_to_move;

    for move_idx in 0..moves.len() {
        pick_next(&mut moves, &mut scores, move_idx);
        let mv = moves[move_idx];

        // Delta pruning: skip captures that can't possibly raise alpha
        if !in_check && mv.is_capture() && !mv.is_en_passant() {
            if let Some((_, victim)) = board.piece_at(mv.to_sq()) {
                if stand_pat + eval::PIECE_VALUE[victim.index()] + 200 < alpha {
                    continue;
                }
            }
        }

        // SEE pruning: skip losing captures
        if !in_check && (mv.is_capture() || mv.is_en_passant()) {
            if see(board, mv) < 0 {
                continue;
            }
        }

        let undo = board.do_move(mv);
        if board.in_check(us) {
            board.undo_move(undo);
            continue;
        }

        ss.nnue.apply_move(ply + 1, board, undo);
        legal_moves += 1;

        let score = -quiescence(board, ss, ply + 1, -beta, -alpha);
        board.undo_move(undo);

        if score >= beta {
            return score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    // In check with no legal moves = checkmate
    if in_check && legal_moves == 0 {
        return mated_in(ply);
    }

    alpha
}

// =============================================================================
// UCI info output
// =============================================================================

fn print_info(ss: &SearchState, depth: i32, score: i32, tt: &TranspositionTable) {
    let elapsed_ms = ss.start_time.elapsed().as_millis() as u64;
    let nps = if elapsed_ms > 0 {
        ss.nodes * 1000 / elapsed_ms
    } else {
        0
    };
    let hashfull = tt.hashfull();

    let score_str = if is_mate_score(score) {
        let mate_dist = (MATE_VALUE - score.abs() + 1) / 2;
        if score > 0 {
            format!("mate {}", mate_dist)
        } else {
            format!("mate -{}", mate_dist)
        }
    } else {
        format!("cp {}", score)
    };

    let pv_len = ss.pv_length[0];
    let pv_str: String = (0..pv_len)
        .map(|i| ss.pv_table[0][i].to_uci())
        .collect::<Vec<_>>()
        .join(" ");

    println!(
        "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
        depth, ss.sel_depth, score_str, ss.nodes, nps, hashfull, elapsed_ms, pv_str,
    );
    std::io::stdout().flush().unwrap();
}

// =============================================================================
// Bench — fixed positions at fixed depth
// =============================================================================

pub const BENCH_POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

pub fn run_bench(depth: i32, tt: std::sync::Arc<TranspositionTable>, threads: usize) {
    let start = Instant::now();
    let mut total_nodes = 0u64;

    for fen in BENCH_POSITIONS {
        let board = Board::from_fen(fen).unwrap();
        let stop = Arc::new(AtomicBool::new(false));

        let mut handles = vec![];
        for tid in 0..threads {
            let mut b = board.clone();
            let tt_clone = Arc::clone(&tt);
            let stop_clone = Arc::clone(&stop);

            handles.push(std::thread::spawn(move || {
                let mut ss = SearchState::new(stop_clone, None);
                ss.thread_id = tid;
                ss.max_depth = depth;
                ss.soft_limit_ms = u64::MAX;
                ss.hard_limit_ms = u64::MAX;
                ss.position_history.push(b.hash);

                iterative_deepening(&mut b, &mut ss, &tt_clone);
                ss.nodes
            }));
        }

        for h in handles {
            total_nodes += h.join().unwrap();
        }
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let nps = if elapsed_ms > 0 {
        total_nodes * 1000 / elapsed_ms
    } else {
        0
    };
    println!("{} nodes {} nps ({} threads)", total_nodes, nps, threads);
}

// =============================================================================
// Time management
// =============================================================================

pub struct TimeControl {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u32>,
    pub movetime: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
}

impl Default for TimeControl {
    fn default() -> Self {
        TimeControl {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movestogo: None,
            movetime: None,
            depth: None,
            nodes: None,
            infinite: false,
        }
    }
}

/// Compute soft and hard time limits from a TimeControl.
pub fn compute_limits(tc: &TimeControl, side: crate::types::Color) -> (u64, u64) {
    if tc.infinite {
        return (u64::MAX, u64::MAX);
    }

    if let Some(ms) = tc.movetime {
        let hard = (ms as f64 * 0.95) as u64;
        return (hard, hard);
    }

    let (my_time, my_inc) = match side {
        crate::types::Color::White => (tc.wtime.unwrap_or(30_000), tc.winc.unwrap_or(0)),
        crate::types::Color::Black => (tc.btime.unwrap_or(30_000), tc.binc.unwrap_or(0)),
    };

    let overhead = 50u64;

    if let Some(mtg) = tc.movestogo {
        let mtg = mtg.max(1) as f64;
        let base_soft = (my_time as f64 / (mtg + 2.0)) + my_inc as f64 * 0.60;
        let base_hard = (my_time as f64 / (mtg / 2.0).max(1.0)) + my_inc as f64 * 0.85;

        // Avoid pathological under-spend in long controls and over-spend in low time.
        let min_soft = ((my_time as f64) * 0.005).max(10.0);
        let max_soft = (my_time as f64) * 0.25;
        let soft_ms = base_soft.clamp(min_soft, max_soft) as u64;

        let hard_cap = my_time.saturating_sub(overhead).max(1);
        let hard_ms = (base_hard as u64)
            .min(hard_cap)
            .min(soft_ms.saturating_mul(5));
        return (soft_ms.min(my_time), hard_ms.min(my_time));
    }

    // Sudden death / increment: adapt expected moves left by remaining time.
    let moves_left = if my_time < 5_000 {
        18.0
    } else if my_time < 20_000 {
        28.0
    } else if my_time < 60_000 {
        36.0
    } else {
        44.0
    };

    let base_soft = my_time as f64 / moves_left + my_inc as f64 * 0.70;
    let panic_bonus = if my_time < 3_000 {
        my_inc as f64 * 0.30
    } else {
        0.0
    };
    let soft_ms = (base_soft + panic_bonus) as u64;

    let hard_cap = my_time.saturating_sub(overhead).max(1);
    let hard_ms = (soft_ms.saturating_mul(5)).min(hard_cap);

    (soft_ms.min(my_time), hard_ms.min(my_time))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_finds_mate_in_1() {
        // White to move, Qh7# is mate
        let mut board = Board::from_fen("6k1/5ppp/8/8/8/8/8/4Q2K w - - 0 1").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(8);
        let mut ss = SearchState::new(stop, None);
        ss.max_depth = 4;
        ss.position_history.push(board.hash);

        let best = iterative_deepening(&mut board, &mut ss, &tt);
        assert!(best != Move::NULL, "Should find a move");
        // The score should indicate mate
        assert!(
            ss.best_score >= MATE_VALUE - 10,
            "Should find a near-mate score"
        );
    }

    #[test]
    fn test_search_avoids_stalemate() {
        // Don't let a winning position turn into stalemate
        let mut board = Board::from_fen("k7/8/1K6/8/8/8/8/1Q6 w - - 0 1").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(8);
        let mut ss = SearchState::new(stop, None);
        ss.max_depth = 6;
        ss.position_history.push(board.hash);

        let best = iterative_deepening(&mut board, &mut ss, &tt);
        assert!(best != Move::NULL, "Should find a move");
        // Should not give stalemate — score should be positive
        assert!(
            ss.best_score > 0,
            "Should have a winning score, not stalemate"
        );
    }

    #[test]
    fn test_search_startpos_reasonable() {
        let mut board = Board::start_pos();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(8);
        let mut ss = SearchState::new(stop, None);
        ss.max_depth = 5;
        ss.position_history.push(board.hash);

        let best = iterative_deepening(&mut board, &mut ss, &tt);
        assert!(
            best != Move::NULL,
            "Should find a move from starting position"
        );
        // Score should be roughly symmetric
        assert!(
            ss.best_score.abs() < 200,
            "Starting position should be roughly equal"
        );
    }

    #[test]
    fn test_draw_detection_50_move() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 50").unwrap();
        assert!(is_draw(&board, &[]));
    }

    #[test]
    fn test_draw_detection_repetition() {
        let hash = 0xDEAD_BEEF_CAFE_1234u64;
        let history = vec![hash, 0x1111, hash, 0x2222];
        // Current position has same hash as history[2] (2 moves ago, same side)
        assert!(is_repetition(hash, 10, &history));
    }

    #[test]
    fn test_lmr_table_sane() {
        let t = LMR_TABLE.get_or_init(init_lmr);
        assert_eq!(t[0][0], 0, "No reduction at depth 0");
        assert!(
            t[10][10] > 0,
            "Should have nonzero reduction at depth 10, move 10"
        );
        assert!(t[10][10] < 10, "Reduction should be reasonable");
    }

    #[test]
    fn test_time_management_infinite() {
        let tc = TimeControl {
            infinite: true,
            ..Default::default()
        };
        let (soft, hard) = compute_limits(&tc, crate::types::Color::White);
        assert_eq!(soft, u64::MAX);
        assert_eq!(hard, u64::MAX);
    }

    #[test]
    fn test_time_management_movetime() {
        let tc = TimeControl {
            movetime: Some(1000),
            ..Default::default()
        };
        let (soft, hard) = compute_limits(&tc, crate::types::Color::White);
        assert!(soft >= 900 && soft <= 1000);
        assert_eq!(soft, hard);
    }
}
