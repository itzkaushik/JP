#!/usr/bin/env python3
"""
NNUE Training on Self-Play Data
================================

End-to-end pipeline: self-play -> WDL training -> nn.bin export.

Usage:
    # Full pipeline
    python train_selfplay.py --all --games 5000 --epochs 20

    # Generate only
    python train_selfplay.py --generate --games 10000 --depth 8

    # Train on existing data
    python train_selfplay.py --train --data data/selfplay.bin --output ../nn.bin

Requirements: pip install python-chess torch numpy
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnue_common import (
    INPUT_SIZE,
    ChessNNUE,
    wdl_loss,
    export_weights,
    read_position_records,
)
from selfplay import DATASET_MAGIC, DATASET_VERSION, generate_selfplay, default_engine_path


class SelfPlayDataset(Dataset):
    def __init__(self, data_file: str, max_positions: int | None = None):
        self.samples: list[tuple] = []
        with open(data_file, "rb") as f:
            magic = f.read(4)
            if magic != DATASET_MAGIC:
                raise ValueError(f"Invalid dataset magic: {magic!r} (expected {DATASET_MAGIC!r})")
            version = struct.unpack("<I", f.read(4))[0]
            if version != DATASET_VERSION:
                raise ValueError(f"Unsupported dataset version: {version}")
            count = struct.unpack("<I", f.read(4))[0]
            limit = min(count, max_positions) if max_positions else count
            for i in range(limit):
                n_w = struct.unpack("<B", f.read(1))[0]
                w_feats = [struct.unpack("<H", f.read(2))[0] for _ in range(n_w)]
                n_b = struct.unpack("<B", f.read(1))[0]
                b_feats = [struct.unpack("<H", f.read(2))[0] for _ in range(n_b)]
                stm, wdl = struct.unpack("<Bf", f.read(5))
                self.samples.append((w_feats, b_feats, stm, wdl))
                if (i + 1) % 500_000 == 0:
                    print(f"  Loaded {i + 1:,}/{limit:,} positions")
        print(f"Loaded {len(self.samples):,} positions from {data_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch):
    batch_size = len(batch)
    w_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    b_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    stm_list = []
    wdl_list = []

    for i, (w_feats, b_feats, stm, wdl) in enumerate(batch):
        if w_feats:
            w_dense[i, w_feats] = 1.0
        if b_feats:
            b_dense[i, b_feats] = 1.0
        stm_list.append(stm)
        wdl_list.append(wdl)

    return (
        w_dense,
        b_dense,
        torch.tensor(stm_list, dtype=torch.float32),
        torch.tensor(wdl_list, dtype=torch.float32),
    )


def train_network(
    data_file: str,
    output_file: str,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 0.001,
    max_positions: int | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = SelfPlayDataset(data_file, max_positions=max_positions)
    random.shuffle(dataset.samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ChessNNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

    print(f"Epochs: {epochs}, batch: {batch_size}, params: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for w_feats, b_feats, stm, wdl in loader:
            w_feats = w_feats.to(device)
            b_feats = b_feats.to(device)
            stm = stm.to(device)
            wdl = wdl.to(device)

            pred = model(w_feats, b_feats, stm)
            loss = wdl_loss(pred, wdl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        marker = " *" if avg < best_loss else ""
        if avg < best_loss:
            best_loss = avg
        print(f"Epoch {epoch + 1:3d}/{epochs}  Loss: {avg:.6f}  LR: {scheduler.get_last_lr()[0]:.6f}{marker}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    export_weights(model, output_file)


def main():
    parser = argparse.ArgumentParser(description="Self-play NNUE training pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate self-play data")
    parser.add_argument("--train", action="store_true", help="Train on self-play data")
    parser.add_argument("--all", action="store_true", help="Generate + train")
    parser.add_argument("--engine", default=default_engine_path(), help="Engine binary path")
    parser.add_argument("--data", default="data/selfplay.bin", help="Dataset path")
    parser.add_argument("--output", default="../nn.bin", help="Output nn.bin path")
    parser.add_argument("--games", type=int, default=5000, help="Self-play games to generate")
    parser.add_argument("--depth", type=int, default=8, help="Search depth per move")
    parser.add_argument("--workers", type=int, default=1, help="Parallel self-play workers")
    parser.add_argument("--eval-file", default=None, help="Existing nn.bin for iterative training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-positions", type=int, default=None, help="Cap positions loaded for training")

    args = parser.parse_args()
    if args.all:
        args.generate = True
        args.train = True

    if not args.generate and not args.train:
        parser.print_help()
        sys.exit(1)

    tools_dir = Path(__file__).resolve().parent
    data_path = args.data if os.path.isabs(args.data) else str(tools_dir / args.data)
    output_path = args.output if os.path.isabs(args.output) else str((tools_dir / args.output).resolve())

    if args.generate:
        generate_selfplay(
            engine_path=args.engine,
            output_file=data_path,
            games=args.games,
            depth=args.depth,
            workers=args.workers,
            eval_file=args.eval_file,
        )

    if args.train:
        if not os.path.isfile(data_path):
            print(f"Dataset not found: {data_path}")
            sys.exit(1)
        train_network(
            data_path,
            output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_positions=args.max_positions,
        )
        print(f"\nWeights saved to {output_path}")
        print("Reload engine with: setoption name EvalFile value nn.bin")


if __name__ == "__main__":
    main()
