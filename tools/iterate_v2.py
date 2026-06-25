#!/usr/bin/env python3
"""
Automated V2 loop:
  selfplay -> train -> mini-SPRT -> promote best nn.bin
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> str:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def parse_sprt_result(text: str) -> tuple[bool, str]:
    if "H1 ACCEPTED" in text:
        return True, "H1 accepted"
    if "H0 ACCEPTED" in text:
        return False, "H0 accepted"
    m = re.search(r"est\.Elo\s+([+-]?\d+(?:\.\d+)?)", text)
    if m and float(m.group(1)) > 0.0:
        return True, f"no decision, positive est Elo {m.group(1)}"
    return False, "no decision / non-positive est Elo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative selfplay+SPRT loop")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sprt-games", type=int, default=80)
    parser.add_argument("--sprt-depth", type=int, default=6)
    parser.add_argument("--sprt-elo0", type=float, default=0.0)
    parser.add_argument("--sprt-elo1", type=float, default=5.0)
    args = parser.parse_args()

    tools = Path(__file__).resolve().parent
    root = tools.parent
    runs = tools / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    engine = root / "target" / "release" / "chessEngine.exe"
    if not engine.exists():
        raise FileNotFoundError(f"Engine not found: {engine}")

    best_net = root / "nn.bin"
    if not best_net.exists():
        print("No baseline nn.bin found; first accepted run will create it.")

    summary_lines: list[str] = []

    for i in range(1, args.cycles + 1):
        run_dir = runs / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_file = run_dir / "selfplay.bin"
        cand_net = run_dir / "nn_candidate.bin"

        print(f"\n=== Cycle {i}/{args.cycles} ===")
        train_cmd = [
            sys.executable,
            "train_selfplay.py",
            "--all",
            "--engine",
            str(engine),
            "--data",
            str(data_file),
            "--output",
            str(cand_net),
            "--games",
            str(args.games),
            "--depth",
            str(args.depth),
            "--workers",
            str(args.workers),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        if best_net.exists():
            train_cmd += ["--eval-file", str(best_net)]
        run(train_cmd, tools)

        if not best_net.exists():
            shutil.copy2(cand_net, best_net)
            msg = f"Cycle {i}: accepted (bootstrap, no baseline) -> {best_net}"
            print(msg)
            summary_lines.append(msg)
            continue

        sprt_cmd = [
            sys.executable,
            "sprt.py",
            "--engine1",
            str(engine),
            "--engine2",
            str(engine),
            "--eval1",
            str(cand_net),
            "--eval2",
            str(best_net),
            "--openings",
            "openings.epd",
            "--elo0",
            str(args.sprt_elo0),
            "--elo1",
            str(args.sprt_elo1),
            "--depth",
            str(args.sprt_depth),
            "--max-games",
            str(args.sprt_games),
            "--hash",
            "64",
            "--threads",
            "1",
        ]
        out = run(sprt_cmd, tools)
        accepted, reason = parse_sprt_result(out)
        if accepted:
            backup = runs / "best_previous.nn.bin"
            if best_net.exists():
                shutil.copy2(best_net, backup)
            shutil.copy2(cand_net, best_net)
            msg = f"Cycle {i}: ACCEPTED ({reason})"
        else:
            msg = f"Cycle {i}: REJECTED ({reason})"
        print(msg)
        summary_lines.append(msg)

    summary = runs / "summary.txt"
    summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nDone. Summary written to: {summary}")


if __name__ == "__main__":
    main()

