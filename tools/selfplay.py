#!/usr/bin/env python3
"""
Self-Play Data Generation for ChessEngine NNUE Training
========================================================

Plays games engine-vs-engine via UCI, records positions with WDL labels
from the final game result (industry-standard bootstrap approach).

Usage:
    python selfplay.py --engine ../target/release/chessEngine.exe --games 1000
    python selfplay.py --engine ../target/release/chessEngine.exe --games 50000 \\
        --depth 8 --workers 4 --output data/selfplay.bin

Requirements: pip install python-chess
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

import chess

# Allow importing nnue_common from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnue_common import board_to_features, stm_index, game_result_wdl, write_position_record

# Magic header for self-play datasets
DATASET_MAGIC = b"SPNL"
DATASET_VERSION = 1


class UCIEngine:
    """Minimal UCI client for ChessEngine."""

    def __init__(self, path: str, hash_mb: int = 128, threads: int = 1, eval_file: str | None = None):
        self.path = path
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", timeout=30.0)

        if eval_file:
            self._send(f"setoption name EvalFile value {eval_file}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        self._send("setoption name OwnBook value false")
        self._send("isready")
        self._wait_for("readyok", timeout=30.0)

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_line(self, timeout: float = 60.0) -> str:
        assert self.proc.stdout is not None
        # Simple blocking read — engine responds quickly for go depth N
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Engine process ended unexpectedly")
        return line.strip()

    def _wait_for(self, token: str, timeout: float = 60.0) -> list[str]:
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line()
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"Timed out waiting for '{token}'")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def go_depth(self, board: chess.Board, depth: int) -> tuple[str, int | None]:
        """Search and return (bestmove_uci, score_cp_or_none)."""
        fen = board.fen()
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        score_cp: int | None = None
        bestmove = "0000"
        deadline = time.time() + max(120.0, depth * 10.0)

        while time.time() < deadline:
            line = self._read_line()
            if line.startswith("info ") and " score cp " in line:
                parts = line.split()
                try:
                    idx = parts.index("cp")
                    score_cp = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            if line.startswith("bestmove "):
                bestmove = line.split()[1]
                break

        return bestmove, score_cp

    def quit(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        self.proc.wait(timeout=5)


def random_opening(board: chess.Board, max_plies: int) -> None:
    """Play random legal moves for opening diversity."""
    n = random.randint(0, max_plies)
    for _ in range(n):
        moves = list(board.legal_moves)
        if not moves:
            return
        board.push(random.choice(moves))


def play_game(
    engine: UCIEngine,
    depth: int,
    max_plies: int,
    opening_plies: int,
    sample_rate: float,
) -> tuple[list[tuple], str]:
    """
    Play one self-play game.
    Returns (recorded_positions, result_string).
    Each recorded position: (w_feats, b_feats, stm, side_at_position)
    side_at_position is stored temporarily; WDL filled in after game ends.
    """
    board = chess.Board()
    random_opening(board, opening_plies)

    snapshots: list[tuple] = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if not board.is_check() and random.random() < sample_rate:
            w_feats, b_feats = board_to_features(board)
            snapshots.append((w_feats, b_feats, stm_index(board), board.turn))

        move_uci, _ = engine.go_depth(board, depth)
        if move_uci in ("0000", "(none)"):
            break
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
        ply += 1

    result = board.result(claim_draw=True) or "1/2-1/2"
    return snapshots, result


def label_positions(snapshots: list[tuple], result: str) -> list[tuple]:
    """Attach WDL labels from game result to each snapshot."""
    labeled = []
    for w_feats, b_feats, stm, turn_at_pos in snapshots:
        wdl = game_result_wdl(result, turn_at_pos)
        if wdl is None:
            continue
        labeled.append((w_feats, b_feats, stm, wdl))
    return labeled


def worker_play_games(
    engine_path: str,
    games: int,
    depth: int,
    hash_mb: int,
    threads: int,
    eval_file: str | None,
    max_plies: int,
    opening_plies: int,
    sample_rate: float,
    worker_id: int,
) -> list[tuple]:
    random.seed(worker_id * 10_007 + int(time.time()))
    engine = UCIEngine(engine_path, hash_mb=hash_mb, threads=threads, eval_file=eval_file)
    all_positions: list[tuple] = []

    for g in range(games):
        engine.new_game()
        snapshots, result = play_game(engine, depth, max_plies, opening_plies, sample_rate)
        all_positions.extend(label_positions(snapshots, result))

    engine.quit()
    return all_positions


def generate_selfplay(
    engine_path: str,
    output_file: str,
    games: int,
    depth: int = 8,
    workers: int = 1,
    hash_mb: int = 128,
    threads_per_engine: int = 1,
    eval_file: str | None = None,
    max_plies: int = 200,
    opening_plies: int = 8,
    sample_rate: float = 0.5,
) -> str:
    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    games_per_worker = games // workers
    extra = games % workers
    allocations = [games_per_worker + (1 if i < extra else 0) for i in range(workers)]

    print(f"Self-play: {games} games, depth {depth}, {workers} worker(s)")
    print(f"Engine: {engine_path}")
    print(f"Output: {output_file}")
    start = time.time()

    all_positions: list[tuple] = []

    if workers == 1:
        all_positions = worker_play_games(
            engine_path, games, depth, hash_mb, threads_per_engine, eval_file,
            max_plies, opening_plies, sample_rate, 0,
        )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for wid, ng in enumerate(allocations):
                if ng == 0:
                    continue
                futures.append(pool.submit(
                    worker_play_games,
                    engine_path, ng, depth, hash_mb, threads_per_engine, eval_file,
                    max_plies, opening_plies, sample_rate, wid,
                ))
            for fut in as_completed(futures):
                all_positions.extend(fut.result())

    elapsed = time.time() - start
    print(f"Collected {len(all_positions):,} positions from {games} games in {elapsed:.1f}s")

    with open(output_file, "wb") as f:
        f.write(DATASET_MAGIC)
        f.write(struct.pack("<I", DATASET_VERSION))
        f.write(struct.pack("<I", len(all_positions)))
        for w_feats, b_feats, stm, wdl in all_positions:
            write_position_record(f, w_feats, b_feats, stm, wdl)

    size = os.path.getsize(output_file)
    print(f"Wrote {output_file} ({size:,} bytes, {len(all_positions):,} positions)")
    return output_file


def default_engine_path() -> str:
    root = Path(__file__).resolve().parent.parent
    release = root / "target" / "release" / "chessEngine.exe"
    if release.is_file():
        return str(release)
    debug = root / "target" / "debug" / "chessEngine.exe"
    if debug.is_file():
        return str(debug)
    # Non-Windows
    release_unix = root / "target" / "release" / "chessEngine"
    if release_unix.is_file():
        return str(release_unix)
    return str(release)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play data generation for NNUE")
    parser.add_argument("--engine", default=default_engine_path(), help="Path to ChessEngine binary")
    parser.add_argument("--output", default="data/selfplay.bin", help="Output dataset path")
    parser.add_argument("--games", type=int, default=1000, help="Number of self-play games")
    parser.add_argument("--depth", type=int, default=8, help="Search depth per move")
    parser.add_argument("--workers", type=int, default=1, help="Parallel game workers")
    parser.add_argument("--hash", type=int, default=128, help="TT hash per engine (MB)")
    parser.add_argument("--threads", type=int, default=1, help="Threads per engine process")
    parser.add_argument("--eval-file", default=None, help="NNUE weights path (optional)")
    parser.add_argument("--max-plies", type=int, default=200, help="Max plies per game")
    parser.add_argument("--opening-plies", type=int, default=8, help="Max random opening plies")
    parser.add_argument("--sample-rate", type=float, default=0.5, help="Fraction of positions to record")

    args = parser.parse_args()
    generate_selfplay(
        engine_path=args.engine,
        output_file=args.output,
        games=args.games,
        depth=args.depth,
        workers=args.workers,
        hash_mb=args.hash,
        threads_per_engine=args.threads,
        eval_file=args.eval_file,
        max_plies=args.max_plies,
        opening_plies=args.opening_plies,
        sample_rate=args.sample_rate,
    )
