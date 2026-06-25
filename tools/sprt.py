#!/usr/bin/env python3
"""
SPRT — Sequential Probability Ratio Test for ChessEngine
===========================================================

Plays engine1 vs engine2 via UCI and stops when a statistical decision is reached.

H0: engine1 is NOT stronger than engine2 by elo0 points
H1: engine1 IS stronger than engine2 by elo1 points

Usage:
    # Test if current build gained +5 Elo vs baseline (Fishtest-style bounds)
    python sprt.py --engine1 ../target/release/chessEngine.exe \\
        --engine2 ../target/release/chessEngine.exe \\
        --elo0 0 --elo1 5

    # Regression test: ensure no more than 5 Elo loss
    python sprt.py --engine1 NEW.exe --engine2 OLD.exe --elo0 -5 --elo1 0

    # Fixed-depth nodes test (fair on any hardware)
    python sprt.py --engine1 ../target/release/chessEngine.exe \\
        --engine2 stockfish.exe --nodes 200000 --concurrency 2

Requirements: pip install python-chess
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uci_engine import UCIEngine


# =============================================================================
# SPRT math (cutechess / Fishtest compatible binomial model on decisive games)
# =============================================================================

def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass
class SprtState:
    elo0: float
    elo1: float
    alpha: float
    beta: float
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def __post_init__(self):
        self.p0 = elo_to_score(self.elo0)
        self.p1 = elo_to_score(self.elo1)
        self.llr_min = math.log(self.beta / (1.0 - self.alpha))
        self.llr_max = math.log((1.0 - self.beta) / self.alpha)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def llr(self) -> float:
        """Trinomial LLR: wins/losses use logistic model; draws weighted 0.5."""
        w, l, d = self.wins, self.losses, self.draws
        n = w + l + d
        if n == 0:
            return 0.0
        # Decisive-game component (primary signal)
        llr = 0.0
        if w + l > 0:
            llr = w * math.log(self.p1 / self.p0) + l * math.log((1.0 - self.p1) / (1.0 - self.p0))
        # Draw contribution: under equal strength draws are neutral; under H1 slightly fewer
        if d > 0:
            mid0 = (self.p0 + (1.0 - self.p0)) / 2.0
            mid1 = (self.p1 + (1.0 - self.p1)) / 2.0
            draw_ratio = d / n
            llr += d * math.log(mid1 / mid0) * 0.25
        return llr

    def update(self, result: float) -> str:
        """result: 1.0 win, 0.5 draw, 0.0 loss from engine1's perspective."""
        if result >= 1.0:
            self.wins += 1
        elif result <= 0.0:
            self.losses += 1
        else:
            self.draws += 1
        return self.status()

    def status(self) -> str:
        llr = self.llr
        if llr >= self.llr_max:
            return "H1"
        if llr <= self.llr_min:
            return "H0"
        return "continue"

    def estimated_elo(self) -> float:
        """Logistic Elo from score rate (includes draws as 0.5)."""
        if self.games == 0:
            return 0.0
        s = min(0.99, max(0.01, self.score_rate))
        return -400.0 * math.log10(1.0 / s - 1.0)


# =============================================================================
# Openings
# =============================================================================

def load_openings(path: str | None, count: int = 64) -> list[str]:
    fens: list[str] = []
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and "FEN" in line:
                    continue
                fen = line.split(" | ")[0].strip() if " | " in line else line
                if " " not in fen:
                    continue
                try:
                    chess.Board(fen)
                    fens.append(fen)
                except ValueError:
                    print(f"Warning: skipping invalid FEN: {fen}", file=sys.stderr)
    if not fens:
        rng = random.Random(42)
        board = chess.Board()
        for i in range(count):
            b = chess.Board()
            plies = rng.randint(0, 14)
            for _ in range(plies):
                moves = list(b.legal_moves)
                if not moves:
                    break
                b.push(rng.choice(moves))
            fens.append(b.fen())
    return fens


# =============================================================================
# Play one game
# =============================================================================

def play_game(
    engine1: UCIEngine,
    engine2: UCIEngine,
    opening_fen: str,
    depth: int | None,
    movetime_ms: int | None,
    nodes: int | None,
    max_plies: int = 300,
) -> float:
    """
    Play one game. engine1 plays White.
    Returns result from engine1 perspective: 1.0 / 0.5 / 0.0
    """
    board = chess.Board(opening_fen)
    engines = [engine1, engine2]
    engine1.new_game()
    engine2.new_game()

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        side = 0 if board.turn == chess.WHITE else 1
        fen = board.fen()
        eng = engines[side]
        if nodes is not None:
            uci = eng.go_nodes(fen, nodes)
        elif movetime_ms is not None:
            uci = eng.go_movetime(fen, movetime_ms)
        elif depth is not None:
            uci = eng.go_depth(fen, depth)
        else:
            uci = eng.go_depth(fen, 8)

        if uci in ("0000", "(none)"):
            break
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


# =============================================================================
# SPRT tournament
# =============================================================================

def run_sprt(
    engine1_path: str,
    engine2_path: str,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_games: int = 20000,
    depth: int | None = 8,
    movetime_ms: int | None = None,
    nodes: int | None = None,
    hash_mb: int = 128,
    threads: int = 1,
    eval1: str | None = None,
    eval2: str | None = None,
    openings_file: str | None = None,
    seed: int = 42,
) -> SprtState:
    rng = random.Random(seed)
    openings = load_openings(openings_file)
    sprt = SprtState(elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)

    e1_cwd = str(Path(engine1_path).resolve().parent)
    e2_cwd = str(Path(engine2_path).resolve().parent)

    print("=" * 60)
    print("SPRT Test")
    print("=" * 60)
    print(f"  Engine1 (candidate): {engine1_path}")
    if eval1:
        print(f"    EvalFile: {eval1}")
    print(f"  Engine2 (baseline):  {engine2_path}")
    if eval2:
        print(f"    EvalFile: {eval2}")
    print(f"  H0: Elo <= {elo0:+.1f}  |  H1: Elo >= {elo1:+.1f}")
    print(f"  alpha={alpha}  beta={beta}  max_games={max_games}")
    if nodes:
        print(f"  TC: nodes {nodes}")
    elif movetime_ms:
        print(f"  TC: movetime {movetime_ms}ms")
    else:
        print(f"  TC: depth {depth}")
    print(f"  Openings: {len(openings)} positions")
    print("=" * 60)

    engine1 = UCIEngine(engine1_path, hash_mb=hash_mb, threads=threads, eval_file=eval1, cwd=e1_cwd)
    engine2 = UCIEngine(engine2_path, hash_mb=hash_mb, threads=threads, eval_file=eval2, cwd=e2_cwd)

    start = time.time()
    game_num = 0

    try:
        while sprt.games < max_games:
            game_num += 1
            opening = rng.choice(openings)
            white_first = rng.random() < 0.5

            if white_first:
                result = play_game(engine1, engine2, opening, depth, movetime_ms, nodes)
            else:
                result = 1.0 - play_game(engine2, engine1, opening, depth, movetime_ms, nodes)

            status = sprt.update(result)
            elapsed = time.time() - start
            n = sprt.games
            nps_games = n / elapsed if elapsed > 0 else 0

            print(
                f"Game {game_num:5d}  W:{sprt.wins:4d} L:{sprt.losses:4d} D:{sprt.draws:4d}  "
                f"score:{sprt.score_rate*100:5.1f}%  Elo:{sprt.estimated_elo():+6.1f}  "
                f"LLR:{sprt.llr:+7.3f}  [{sprt.llr_min:+.3f}, {sprt.llr_max:+.3f}]  "
                f"({nps_games:.2f} games/s)",
                flush=True,
            )

            if status == "H1":
                print()
                print(f"*** H1 ACCEPTED — Engine1 is >= {elo1:+.0f} Elo stronger (95% confidence) ***")
                print(f"    Final: {sprt.wins}W {sprt.losses}L {sprt.draws}D  "
                      f"est.Elo {sprt.estimated_elo():+.1f}  after {n} games")
                break
            if status == "H0":
                print()
                print(f"*** H0 ACCEPTED — Engine1 is NOT >= {elo1:+.0f} Elo stronger ***")
                print(f"    Final: {sprt.wins}W {sprt.losses}L {sprt.draws}D  "
                      f"est.Elo {sprt.estimated_elo():+.1f}  after {n} games")
                break
        else:
            print()
            print(f"*** MAX GAMES ({max_games}) — no decision ***")
            print(f"    Final: {sprt.wins}W {sprt.losses}L {sprt.draws}D  "
                  f"est.Elo {sprt.estimated_elo():+.1f}  LLR {sprt.llr:+.3f}")
    finally:
        engine1.quit()
        engine2.quit()

    return sprt


def default_engine() -> str:
    root = Path(__file__).resolve().parent.parent
    for name in ("chessEngine.exe", "chessEngine"):
        p = root / "target" / "release" / name
        if p.is_file():
            return str(p)
    raise FileNotFoundError("Release engine not found. Run: cargo build --release")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPRT testing for ChessEngine")
    parser.add_argument("--engine1", default=None, help="Candidate engine (tested as stronger)")
    parser.add_argument("--engine2", default=None, help="Baseline engine")
    parser.add_argument("--eval1", default=None, help="NNUE file for engine1")
    parser.add_argument("--eval2", default=None, help="NNUE file for engine2")
    parser.add_argument("--elo0", type=float, default=0.0, help="H0 bound (Elo)")
    parser.add_argument("--elo1", type=float, default=5.0, help="H1 bound (Elo)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=8, help="Fixed depth TC")
    parser.add_argument("--movetime", type=int, default=None, help="Movetime ms (overrides depth)")
    parser.add_argument("--nodes", type=int, default=None, help="Fixed nodes TC (overrides depth/movetime)")
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--openings", default=None, help="EPD/FEN list file")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    e1 = args.engine1 or default_engine()
    e2 = args.engine2 or e1

    depth = None if (args.movetime or args.nodes) else args.depth

    run_sprt(
        engine1_path=e1,
        engine2_path=e2,
        elo0=args.elo0,
        elo1=args.elo1,
        alpha=args.alpha,
        beta=args.beta,
        max_games=args.max_games,
        depth=depth,
        movetime_ms=args.movetime,
        nodes=args.nodes,
        hash_mb=args.hash,
        threads=args.threads,
        eval1=args.eval1,
        eval2=args.eval2,
        openings_file=args.openings,
        seed=args.seed,
    )
