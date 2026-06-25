#!/usr/bin/env python3
"""Validate and deduplicate FEN lines in an openings file."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess


def clean_openings(path: Path) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(path)

    seen: set[str] = set()
    kept: list[str] = []
    total = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        total += 1
        fen = line.split(" | ")[0].strip()
        try:
            chess.Board(fen)
        except ValueError:
            continue
        if fen in seen:
            continue
        seen.add(fen)
        kept.append(fen)

    output = "# Cleaned openings (valid unique FENs)\n" + "\n".join(kept) + "\n"
    path.write_text(output, encoding="utf-8")
    return total, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean invalid/duplicate FEN openings")
    parser.add_argument("--file", default="openings.epd", help="Openings file path")
    args = parser.parse_args()

    path = Path(args.file)
    total, kept = clean_openings(path)
    print(f"Processed {total} lines, kept {kept} valid unique FENs -> {path}")


if __name__ == "__main__":
    main()

