#!/usr/bin/env python3
"""Shared NNUE constants, feature extraction, model, and weight export."""

from __future__ import annotations

import struct
import os

import numpy as np
import torch
import torch.nn as nn

# Must match nnue.rs
INPUT_SIZE = 768
L1_SIZE = 256
L2_SIZE = 32
L3_SIZE = 32
QA = 255
QB = 64
SCALE = 400


def board_to_features(board) -> tuple[list[int], list[int]]:
    """Sparse HalfKA-style features for white and black perspectives."""
    import chess

    white_features: list[int] = []
    black_features: list[int] = []

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        color_idx = 0 if piece.color == chess.WHITE else 1
        piece_idx = piece.piece_type - 1
        white_features.append(color_idx * 384 + piece_idx * 64 + sq)
        black_features.append((color_idx ^ 1) * 384 + piece_idx * 64 + (sq ^ 56))

    return white_features, black_features


def stm_index(board) -> int:
    import chess

    return 0 if board.turn == chess.WHITE else 1


def game_result_wdl(result: str, side_to_move) -> float | None:
    """WDL target from side-to-move perspective: 1.0 win, 0.5 draw, 0.0 loss."""
    import chess

    if result == "1-0":
        return 1.0 if side_to_move == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if side_to_move == chess.BLACK else 0.0
    if result == "1/2-1/2":
        return 0.5
    return None


class ChessNNUE(nn.Module):
    """(768 -> 256)x2 -> 32 -> 32 -> 1"""

    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(INPUT_SIZE, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, L3_SIZE)
        self.out = nn.Linear(L3_SIZE, 1)

    def forward(self, white_features, black_features, stm):
        w_acc = torch.clamp(self.ft(white_features), 0.0, 1.0)
        b_acc = torch.clamp(self.ft(black_features), 0.0, 1.0)
        stm_expanded = stm.unsqueeze(1)
        us_acc = torch.where(stm_expanded == 0, w_acc, b_acc)
        them_acc = torch.where(stm_expanded == 0, b_acc, w_acc)
        combined = torch.cat([us_acc, them_acc], dim=1)
        x = torch.clamp(self.l2(combined), 0.0, 1.0)
        x = torch.clamp(self.l3(x), 0.0, 1.0)
        return self.out(x).squeeze(1)


def sigmoid_wdl(x, scale: float = SCALE):
    return torch.sigmoid(x / scale)


def wdl_loss(predictions, game_results):
  """Binary cross-entropy style loss on WDL probabilities."""
  return torch.nn.functional.binary_cross_entropy_with_logits(predictions / SCALE, game_results)


def export_weights(model: ChessNNUE, output_file: str) -> None:
    """Export quantized weights matching nnue.rs binary layout."""
    with open(output_file, "wb") as f:
        f.write(b"NNUE")
        f.write(struct.pack("<I", 1))

        ft_weight = model.ft.weight.data.cpu().numpy()
        ft_bias = model.ft.bias.data.cpu().numpy()
        ft_weight_q = np.clip(np.round(ft_weight * QA), -32768, 32767).astype(np.int16)
        ft_bias_q = np.clip(np.round(ft_bias * QA), -32768, 32767).astype(np.int16)

        for i in range(INPUT_SIZE):
            for j in range(L1_SIZE):
                f.write(struct.pack("<h", int(ft_weight_q[j, i])))
        for j in range(L1_SIZE):
            f.write(struct.pack("<h", int(ft_bias_q[j])))

        l2_weight = model.l2.weight.data.cpu().numpy()
        l2_bias = model.l2.bias.data.cpu().numpy()
        l2_weight_q = np.clip(np.round(l2_weight * QB), -128, 127).astype(np.int8)
        l2_bias_q = np.clip(np.round(l2_bias * QA * QB), -2_147_483_648, 2_147_483_647).astype(np.int32)
        for j in range(L2_SIZE):
            for i in range(L1_SIZE * 2):
                f.write(struct.pack("<b", int(l2_weight_q[j, i])))
        for j in range(L2_SIZE):
            f.write(struct.pack("<i", int(l2_bias_q[j])))

        l3_weight = model.l3.weight.data.cpu().numpy()
        l3_bias = model.l3.bias.data.cpu().numpy()
        l3_weight_q = np.clip(np.round(l3_weight * QB), -128, 127).astype(np.int8)
        l3_bias_q = np.clip(np.round(l3_bias * QA * QB), -2_147_483_648, 2_147_483_647).astype(np.int32)
        for j in range(L3_SIZE):
            for i in range(L2_SIZE):
                f.write(struct.pack("<b", int(l3_weight_q[j, i])))
        for j in range(L3_SIZE):
            f.write(struct.pack("<i", int(l3_bias_q[j])))

        out_weight = model.out.weight.data.cpu().numpy()
        out_bias = model.out.bias.data.cpu().numpy()
        out_weight_q = np.clip(np.round(out_weight[0] * QB), -128, 127).astype(np.int8)
        out_bias_q = int(np.clip(np.round(out_bias[0] * QA * QB), -2_147_483_648, 2_147_483_647))
        for j in range(L3_SIZE):
            f.write(struct.pack("<b", int(out_weight_q[j])))
        f.write(struct.pack("<i", out_bias_q))

    print(f"Exported {output_file} ({os.path.getsize(output_file):,} bytes)")


def write_position_record(f, w_feats, b_feats, stm: int, wdl: float) -> None:
    """Write one sparse training position to a binary stream."""
    f.write(struct.pack("<B", len(w_feats)))
    for idx in w_feats:
        f.write(struct.pack("<H", idx))
    f.write(struct.pack("<B", len(b_feats)))
    for idx in b_feats:
        f.write(struct.pack("<H", idx))
    f.write(struct.pack("<Bf", stm, wdl))


def read_position_records(path: str):
    """Yield (w_feats, b_feats, stm, wdl) from a self-play dataset file."""
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            n_w = struct.unpack("<B", f.read(1))[0]
            w_feats = [struct.unpack("<H", f.read(2))[0] for _ in range(n_w)]
            n_b = struct.unpack("<B", f.read(1))[0]
            b_feats = [struct.unpack("<H", f.read(2))[0] for _ in range(n_b)]
            stm, wdl = struct.unpack("<Bf", f.read(5))
            yield w_feats, b_feats, stm, wdl
