#!/usr/bin/env python3
"""Shared minimal UCI client for ChessEngine tooling."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


class UCIEngine:
    """Minimal UCI client."""

    def __init__(
        self,
        path: str,
        hash_mb: int = 128,
        threads: int = 1,
        eval_file: str | None = None,
        cwd: str | None = None,
    ):
        self.path = str(Path(path).resolve())
        self.cwd = cwd or str(Path(self.path).parent)
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        self._send("uci")
        self._wait_for("uciok", timeout=60.0)

        if eval_file:
            eval_path = str(Path(eval_file).resolve())
            self._send(f'setoption name EvalFile value "{eval_path}"')
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        self._send("setoption name OwnBook value false")
        self._send("isready")
        self._wait_for("readyok", timeout=60.0)

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Engine exited unexpectedly: {self.path}")
        return line.strip()

    def _wait_for(self, token: str, timeout: float = 60.0) -> list[str]:
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line()
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"Timed out waiting for '{token}' from {self.path}")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=60.0)

    def position_fen(self, fen: str) -> None:
        self._send(f"position fen {fen}")

    def go(self, go_cmd: str, timeout_sec: float = 300.0) -> str:
        """Send go command and return bestmove UCI string."""
        self._send(go_cmd)
        bestmove = "0000"
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            line = self._read_line()
            if line.startswith("bestmove "):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else "0000"
                break
        return bestmove

    def go_depth(self, fen: str, depth: int) -> str:
        self.position_fen(fen)
        return self.go(f"go depth {depth}", timeout_sec=max(120.0, depth * 15.0))

    def go_movetime(self, fen: str, ms: int) -> str:
        self.position_fen(fen)
        return self.go(f"go movetime {ms}", timeout_sec=ms / 1000.0 + 30.0)

    def go_nodes(self, fen: str, nodes: int) -> str:
        self.position_fen(fen)
        return self.go(f"go nodes {nodes}", timeout_sec=300.0)

    def quit(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
