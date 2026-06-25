// =============================================================================
// uci.rs — Universal Chess Interface protocol handler
// =============================================================================
//
// Minimum UCI implementation for GUI compatibility:
// uci, isready, ucinewgame, position, go, stop, quit, bench

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use crate::syzygy::SyzygyAdapter;
use polyglot_book_rs::PolyglotBook;
use pyrrhic_rs::TableBases;

use crate::attacks;
use crate::board::Board;
use crate::search::{self, SearchState, TimeControl, compute_limits, run_bench};
use crate::tt::TranspositionTable;
use crate::types::{Color, Move, Piece, Square};

const ENGINE_NAME: &str = "ChessEngine";
const ENGINE_AUTHOR: &str = "JP";

// =============================================================================
// Parse a UCI move string (e.g. "e2e4", "e7e8q") against a Board
// =============================================================================

fn parse_uci_move(board: &Board, uci: &str) -> Option<Move> {
    if uci.len() < 4 {
        return None;
    }

    let from = Square::from_algebraic(&uci[0..2])?;
    let to = Square::from_algebraic(&uci[2..4])?;
    let promo_char = if uci.len() > 4 {
        Some(uci.as_bytes()[4])
    } else {
        None
    };

    let moves = board.generate_legal_moves();
    for mv in moves.as_slice() {
        if mv.from_sq() == from && mv.to_sq() == to {
            match promo_char {
                Some(b'q') => {
                    if mv.promotion_piece() == Some(Piece::Queen) {
                        return Some(*mv);
                    }
                }
                Some(b'r') => {
                    if mv.promotion_piece() == Some(Piece::Rook) {
                        return Some(*mv);
                    }
                }
                Some(b'b') => {
                    if mv.promotion_piece() == Some(Piece::Bishop) {
                        return Some(*mv);
                    }
                }
                Some(b'n') => {
                    if mv.promotion_piece() == Some(Piece::Knight) {
                        return Some(*mv);
                    }
                }
                None => {
                    if !mv.is_promotion() {
                        return Some(*mv);
                    }
                }
                _ => continue,
            }
        }
    }
    None
}

// =============================================================================
// Parse "go" parameters into a TimeControl
// =============================================================================

fn parse_go(tokens: &[&str]) -> TimeControl {
    let mut tc = TimeControl::default();
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i] {
            "wtime" => {
                i += 1;
                tc.wtime = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "btime" => {
                i += 1;
                tc.btime = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "winc" => {
                i += 1;
                tc.winc = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "binc" => {
                i += 1;
                tc.binc = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "movestogo" => {
                i += 1;
                tc.movestogo = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "movetime" => {
                i += 1;
                tc.movetime = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "depth" => {
                i += 1;
                tc.depth = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "nodes" => {
                i += 1;
                tc.nodes = tokens.get(i).and_then(|s| s.parse().ok());
            }
            "infinite" => {
                tc.infinite = true;
            }
            _ => {}
        }
        i += 1;
    }

    tc
}

// =============================================================================
// UCI main loop
// =============================================================================

pub fn uci_loop() {
    // Initialize magic bitboard tables upfront
    attacks::init();

    // Try to auto-load NNUE weights
    crate::nnue::load_default_weights();

    let mut tt = Arc::new(TranspositionTable::new(64)); // 64 MB default
    let stop = Arc::new(AtomicBool::new(false));

    let mut board = Board::start_pos();
    let mut game_history: Vec<u64> = vec![board.hash];

    let mut search_handle: Option<thread::JoinHandle<()>> = None;
    let mut helper_handles: Vec<thread::JoinHandle<()>> = Vec::new();

    let mut num_threads: usize = 1;
    let mut own_book = true;
    let mut book: Option<PolyglotBook> = PolyglotBook::load("book.bin").ok();
    let mut syzygy: Option<Arc<TableBases<SyzygyAdapter>>> = None;

    // Stdin reading: we read synchronously on main thread.
    // "stop" is handled via the stop flag.
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            // =================================================================
            // uci
            // =================================================================
            "uci" => {
                println!("id name {}", ENGINE_NAME);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max 64");
                println!("option name EvalFile type string default nn.bin");
                println!("option name OwnBook type check default true");
                println!("option name BookPath type string default book.bin");
                println!("option name SyzygyPath type string default <empty>");
                println!("uciok");
                io::stdout().flush().unwrap();
            }

            // =================================================================
            // isready
            // =================================================================
            "isready" => {
                // Wait for any running search to finish
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
                println!("readyok");
                io::stdout().flush().unwrap();
            }

            // =================================================================
            // ucinewgame
            // =================================================================
            "ucinewgame" => {
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
                tt.clear();
                tt.advance_age();
                board = Board::start_pos();
                game_history = vec![board.hash];
            }

            // =================================================================
            // setoption
            // =================================================================
            "setoption" => {
                // Parse: setoption name Hash value 128
                if let (Some(name_idx), Some(val_idx)) = (
                    tokens.iter().position(|&t| t == "name"),
                    tokens.iter().position(|&t| t == "value"),
                ) {
                    let name = tokens[name_idx + 1..val_idx].join(" ");
                    let value = tokens[val_idx + 1..].join(" ");
                    let value = value.trim_matches('"');

                    if name.eq_ignore_ascii_case("Hash") {
                        if let Ok(mb) = value.parse::<usize>() {
                            // Wait for any running search before replacing TT
                            if let Some(handle) = search_handle.take() {
                                let _ = handle.join();
                            }
                            tt = Arc::new(TranspositionTable::new(mb.max(1).min(4096)));
                        }
                    } else if name.eq_ignore_ascii_case("Threads") {
                        if let Ok(n) = value.parse::<usize>() {
                            num_threads = n.max(1).min(64);
                        }
                    } else if name.eq_ignore_ascii_case("OwnBook") {
                        own_book = value.eq_ignore_ascii_case("true");
                    } else if name.eq_ignore_ascii_case("BookPath") {
                        book = PolyglotBook::load(value).ok();
                    } else if name.eq_ignore_ascii_case("EvalFile") {
                        crate::nnue::load_weights(value);
                    } else if name.eq_ignore_ascii_case("SyzygyPath") {
                        if let Some(handle) = search_handle.take() {
                            let _ = handle.join();
                        }
                        syzygy = TableBases::<SyzygyAdapter>::new(value).ok().map(Arc::new);
                    }
                }
            }

            // =================================================================
            // position
            // =================================================================
            "position" => {
                let mut move_start = 2;

                if tokens.get(1) == Some(&"startpos") {
                    board = Board::start_pos();
                    game_history = vec![board.hash];
                    move_start = 2;
                } else if tokens.get(1) == Some(&"fen") {
                    // Collect FEN tokens (up to "moves" or end)
                    let fen_end = tokens
                        .iter()
                        .position(|&t| t == "moves")
                        .unwrap_or(tokens.len());
                    let fen_str = tokens[2..fen_end].join(" ");
                    if let Ok(b) = Board::from_fen(&fen_str) {
                        board = b;
                        game_history = vec![board.hash];
                    }
                    move_start = fen_end;
                }

                // Apply moves
                if tokens.get(move_start) == Some(&"moves") {
                    for uci_mv in &tokens[move_start + 1..] {
                        if let Some(mv) = parse_uci_move(&board, uci_mv) {
                            board = board.make_move(mv);
                            game_history.push(board.hash);
                        }
                    }
                }
            }

            // =================================================================
            // go
            // =================================================================
            "go" => {
                // Wait for any previous search
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
                for h in helper_handles.drain(..) {
                    let _ = h.join();
                }

                stop.store(false, Ordering::SeqCst);
                tt.advance_age();

                let tc = parse_go(&tokens[1..]);
                let max_nodes = tc.nodes.unwrap_or(u64::MAX);
                // When nodes-only mode, don't impose time limits
                let (soft_ms, hard_ms) = if tc.nodes.is_some() && tc.wtime.is_none() && tc.btime.is_none() && tc.movetime.is_none() {
                    (u64::MAX, u64::MAX)
                } else {
                    compute_limits(&tc, board.side_to_move)
                };
                let max_depth = tc.depth.unwrap_or(128);

                // --- Opening Book Lookup ---
                if own_book {
                    if let Some(ref bk) = book {
                        let fen = board.to_fen();
                        if let Some(entry) = bk.get_best_move_from_fen(&fen) {
                            // Verify the returned move string is legal in our engine
                            let moves = board.generate_legal_moves();
                            if let Some(mv) = moves
                                .as_slice()
                                .iter()
                                .copied()
                                .find(|m| m.to_uci() == entry.move_string)
                            {
                                println!("bestmove {}", mv.to_uci());
                                io::stdout().flush().unwrap();
                                continue;
                            }
                        }
                    }
                }

                let mut board_copy = board;
                let history_copy = game_history.clone();
                let stop_clone = Arc::clone(&stop);
                let tt_clone = Arc::clone(&tt);
                let syzygy_clone = syzygy.clone();
                let start_time = Instant::now();

                // --- Spawn helper threads (Lazy SMP) ---
                for tid in 1..num_threads {
                    let mut h_board = board_copy;
                    let h_history = history_copy.clone();
                    let h_stop = Arc::clone(&stop);
                    let h_tt = Arc::clone(&tt);
                    let h_syzygy = syzygy.clone();
                    let h_start = start_time;

                    helper_handles.push(thread::spawn(move || {
                        let mut ss = SearchState::new(h_stop, h_syzygy);
                        ss.thread_id = tid;
                        ss.max_depth = max_depth;
                        ss.soft_limit_ms = soft_ms;
                        ss.hard_limit_ms = hard_ms;
                        ss.max_nodes = max_nodes;
                        ss.start_time = h_start;
                        ss.position_history = h_history;

                        search::iterative_deepening(&mut h_board, &mut ss, &h_tt);
                    }));
                }

                // --- Main thread (thread 0) ---
                search_handle = Some(thread::spawn(move || {
                    let mut ss = SearchState::new(stop_clone, syzygy_clone);
                    ss.thread_id = 0;
                    ss.max_depth = max_depth;
                    ss.soft_limit_ms = soft_ms;
                    ss.hard_limit_ms = hard_ms;
                    ss.max_nodes = max_nodes;
                    ss.start_time = start_time;
                    ss.position_history = history_copy;

                    let best = search::iterative_deepening(&mut board_copy, &mut ss, &tt_clone);

                    let best_uci = if best != Move::NULL {
                        best.to_uci()
                    } else {
                        // Fallback: pick first legal move
                        let moves = board_copy.generate_legal_moves();
                        if moves.len() > 0 {
                            moves[0].to_uci()
                        } else {
                            "0000".to_string()
                        }
                    };

                    println!("bestmove {}", best_uci);
                    io::stdout().flush().unwrap();
                }));
            }

            // =================================================================
            // stop
            // =================================================================
            "stop" => {
                stop.store(true, Ordering::SeqCst);
                for h in helper_handles.drain(..) {
                    let _ = h.join();
                }
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
            }

            // =================================================================
            // quit
            // =================================================================
            "quit" => {
                stop.store(true, Ordering::SeqCst);
                break;
            }

            // =================================================================
            // bench [depth]
            // =================================================================
            "bench" => {
                let depth = tokens
                    .get(1)
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(13);
                run_bench(depth, Arc::clone(&tt), num_threads);
            }

            // =================================================================
            // display (debug)
            // =================================================================
            "d" | "display" => {
                println!("{}", board);
                io::stdout().flush().unwrap();
            }

            // =================================================================
            // perft [depth] (debug)
            // =================================================================
            "perft" => {
                let depth = tokens
                    .get(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(5);
                let start = Instant::now();
                let nodes = crate::movegen::perft(&board, depth);
                let elapsed = start.elapsed();
                let nps = if elapsed.as_secs_f64() > 0.0 {
                    nodes as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                println!(
                    "perft({}) = {}  ({:.3}s, {:.1}M nps)",
                    depth,
                    nodes,
                    elapsed.as_secs_f64(),
                    nps / 1_000_000.0
                );
                io::stdout().flush().unwrap();
            }

            _ => {
                // Unknown command — ignore silently per UCI spec
            }
        }
    }
}
