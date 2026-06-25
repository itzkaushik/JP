mod attacks;
mod bits;
mod board;
mod eval;
mod movegen;
mod nnue;
mod search;
mod syzygy;
mod tt;
mod types;
mod uci;

fn main() {
    // Check for bench command: `./engine bench [depth]`
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "bench" {
        attacks::init();
        nnue::load_default_weights();
        let depth = args
            .get(2)
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(13);
        let threads = args
            .get(3)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);
        let tt = std::sync::Arc::new(tt::TranspositionTable::new(16));
        search::run_bench(depth, tt, threads);
        return;
    }

    // Default: enter UCI loop
    uci::uci_loop();
}
