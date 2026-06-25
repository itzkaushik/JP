#!/usr/bin/env python3
"""
NNUE Training on Lichess Elite Data
=====================================

Trains an NNUE network on real game data from the Lichess Elite Database.
Uses the WDL (Win/Draw/Loss) blended loss — the industry-standard approach.

Architecture: (768 → 256)×2 → 32 → 32 → 1

Usage:
    python train_lichess.py --pgn data/lichess_elite_2024-01.pgn --output nn.bin
    python train_lichess.py --pgn data/*.pgn --output nn.bin --epochs 30

Requirements: pip install python-chess torch numpy
"""

import argparse
import glob
import math
import struct
import os
import random
import sys
import time

import chess
import chess.pgn
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    sys.exit(1)

# =============================================================================
# Constants (must match nnue.rs)
# =============================================================================

INPUT_SIZE = 768    # 12 pieces × 64 squares
L1_SIZE = 256       # Accumulator
L2_SIZE = 32
L3_SIZE = 32
QA = 255            # Accumulator quantization scale
QB = 64             # Weight quantization scale
SCALE = 400         # Sigmoid scaling factor (cp → WDL probability)

# =============================================================================
# Feature extraction
# =============================================================================

def board_to_features(board: chess.Board):
    """Convert board to sparse feature indices for both perspectives."""
    white_features = []
    black_features = []
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            color_idx = 0 if piece.color == chess.WHITE else 1
            piece_idx = piece.piece_type - 1  # 0-based
            
            w_idx = color_idx * 384 + piece_idx * 64 + sq
            white_features.append(w_idx)
            
            b_idx = (color_idx ^ 1) * 384 + piece_idx * 64 + (sq ^ 56)
            black_features.append(b_idx)
    
    return white_features, black_features


def features_to_dense(white_features, black_features):
    """Convert sparse feature indices to dense binary tensors."""
    w = np.zeros(INPUT_SIZE, dtype=np.float32)
    b = np.zeros(INPUT_SIZE, dtype=np.float32)
    for idx in white_features:
        w[idx] = 1.0
    for idx in black_features:
        b[idx] = 1.0
    return w, b

# =============================================================================
# PGN parsing — extract positions + game results
# =============================================================================

def result_to_score(result_str: str, board_turn: bool) -> float:
    """Convert game result to WDL value from the side-to-move perspective.
    
    Returns: 1.0 = side-to-move wins, 0.5 = draw, 0.0 = side-to-move loses
    """
    if result_str == "1-0":
        return 1.0 if board_turn == chess.WHITE else 0.0
    elif result_str == "0-1":
        return 1.0 if board_turn == chess.BLACK else 0.0
    elif result_str == "1/2-1/2":
        return 0.5
    else:
        return None  # Unknown result — skip


def parse_pgn_files(pgn_files: list, max_positions: int = 10_000_000,
                    min_move: int = 8, skip_prob: float = 0.7) -> list:
    """Parse PGN files, extracting positions with game outcomes.
    
    Args:
        pgn_files: List of PGN file paths
        max_positions: Maximum positions to extract
        min_move: Skip positions before this move number (avoid opening book)
        skip_prob: Probability of skipping a position (for diversity)
    
    Returns:
        List of (white_features, black_features, stm, game_result)
    """
    data = []
    games_parsed = 0
    start_time = time.time()
    
    for pgn_file in pgn_files:
        print(f"Parsing {pgn_file}...")
        
        with open(pgn_file, 'r', errors='replace') as f:
            while len(data) < max_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                result_str = game.headers.get("Result", "*")
                if result_str == "*":
                    continue  # Incomplete game
                
                games_parsed += 1
                
                board = game.board()
                move_num = 0
                
                for move in game.mainline_moves():
                    board.push(move)
                    move_num += 1
                    
                    # Skip early moves (opening book territory)
                    if move_num < min_move * 2:  # min_move is in full moves
                        continue
                    
                    # Random skip for diversity (don't take every position)
                    if random.random() < skip_prob:
                        continue
                    
                    # Skip positions in check (noisy for training)
                    if board.is_check():
                        continue
                    
                    # Skip positions with very few pieces (endgame tablebases better)
                    if len(board.piece_map()) < 5:
                        continue
                    
                    # Get game result from side-to-move perspective
                    game_result = result_to_score(result_str, board.turn)
                    if game_result is None:
                        continue
                    
                    # Extract features
                    w_feats, b_feats = board_to_features(board)
                    stm = 0.0 if board.turn == chess.WHITE else 1.0
                    
                    data.append((w_feats, b_feats, stm, game_result))
                
                if games_parsed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = games_parsed / max(elapsed, 0.01)
                    print(f"  {games_parsed:,} games, {len(data):,} positions ({rate:.0f} games/s)")
                
                if len(data) >= max_positions:
                    break
    
    elapsed = time.time() - start_time
    print(f"\nParsed {games_parsed:,} games -> {len(data):,} positions in {elapsed:.1f}s")
    return data

# =============================================================================
# PyTorch model (same architecture as nnue.rs)
# =============================================================================

class ChessNNUE(nn.Module):
    """NNUE-style network: (768 → 256)×2 → 32 → 32 → 1"""
    
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
        x = self.out(x)
        
        return x.squeeze(1)

# =============================================================================
# WDL Loss Function (industry standard)
# =============================================================================

def sigmoid_wdl(x, scale=SCALE):
    """Convert centipawn score to WDL probability using sigmoid."""
    return torch.sigmoid(x / scale)


def wdl_loss(predictions, game_results, lambda_wdl=0.8):
    """WDL blended loss function.
    
    The target is a blend of:
      - game_results: actual outcome (1.0/0.5/0.0)
      - The network's output is passed through sigmoid to get predicted WDL
    
    Args:
        predictions: raw network output (centipawns)
        game_results: game outcome from side-to-move perspective
        lambda_wdl: weight for game outcome (0.8 = 80% game, 20% self-consistency)
    """
    pred_wdl = sigmoid_wdl(predictions)
    target = game_results  # Pure WDL target
    
    loss = (pred_wdl - target) ** 2
    return loss.mean()

# =============================================================================
# Training
# =============================================================================

class SparseChessDataset(Dataset):
    def __init__(self, data: list):
        self.data = data
        self.n = len(data)
        
    def __len__(self):
        return self.n
        
    def __getitem__(self, idx):
        # Just return the sparse tuples
        return self.data[idx]

def chess_collate(batch):
    batch_size = len(batch)
    w_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    b_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    stm_list = []
    result_list = []
    
    for i, (w_feats, b_feats, stm, result) in enumerate(batch):
        # We can still do a python loop here, but it's only over the batch
        # For maximum speed we use scatter_ or direct indexing
        if w_feats:
            w_dense[i, w_feats] = 1.0
        if b_feats:
            b_dense[i, b_feats] = 1.0
        stm_list.append(stm)
        result_list.append(result)
        
    return (w_dense, b_dense, 
            torch.tensor(stm_list, dtype=torch.float32), 
            torch.tensor(result_list, dtype=torch.float32))

def train_network(data: list, output_file: str, epochs: int = 10,
                  batch_size: int = 16384, lr: float = 0.001):
    """Train the NNUE network on parsed game data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    dataset = SparseChessDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       collate_fn=chess_collate, num_workers=0, pin_memory=False)
    
    model = ChessNNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining for {epochs} epochs, batch_size={batch_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Dataset: {len(data):,} positions\n")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for w_feats, b_feats, stm, results in loader:
            w_feats = w_feats.to(device)
            b_feats = b_feats.to(device)
            stm = stm.to(device)
            results = results.to(device)
            
            predictions = model(w_feats, b_feats, stm)
            loss = wdl_loss(predictions, results, lambda_wdl=0.8)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        
        marker = " *" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {avg_loss:.6f}  LR: {lr_now:.6f}{marker}")
    
    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Exporting quantized weights to {output_file}...")
    export_weights(model, output_file)
    print("Done!")

# =============================================================================
# Weight export (quantization) — must match nnue.rs binary format
# =============================================================================

def export_weights(model: ChessNNUE, output_file: str):
    """Export model weights in quantized binary format matching nnue.rs."""
    
    with open(output_file, 'wb') as f:
        # Magic: "NNUE"
        f.write(b'NNUE')
        # Version: 1
        f.write(struct.pack('<I', 1))
        
        # Feature transformer weights: INPUT_SIZE × L1_SIZE as i16
        ft_weight = model.ft.weight.data.cpu().numpy()  # (L1_SIZE, INPUT_SIZE)
        ft_bias = model.ft.bias.data.cpu().numpy()      # (L1_SIZE,)
        
        ft_weight_q = np.clip(np.round(ft_weight * QA), -32768, 32767).astype(np.int16)
        ft_bias_q = np.clip(np.round(ft_bias * QA), -32768, 32767).astype(np.int16)
        
        for i in range(INPUT_SIZE):
            for j in range(L1_SIZE):
                f.write(struct.pack('<h', int(ft_weight_q[j, i])))
        
        for j in range(L1_SIZE):
            f.write(struct.pack('<h', int(ft_bias_q[j])))
        
        # L2 weights: (L1_SIZE*2) × L2_SIZE as i8
        l2_weight = model.l2.weight.data.cpu().numpy()
        l2_bias = model.l2.bias.data.cpu().numpy()
        
        l2_weight_q = np.clip(np.round(l2_weight * QB), -128, 127).astype(np.int8)
        l2_bias_q = np.clip(np.round(l2_bias * QA * QB), -2147483648, 2147483647).astype(np.int32)
        
        for j in range(L2_SIZE):
            for i in range(L1_SIZE * 2):
                f.write(struct.pack('<b', int(l2_weight_q[j, i])))
        
        for j in range(L2_SIZE):
            f.write(struct.pack('<i', int(l2_bias_q[j])))
        
        # L3 weights: L2_SIZE × L3_SIZE as i8
        l3_weight = model.l3.weight.data.cpu().numpy()
        l3_bias = model.l3.bias.data.cpu().numpy()
        
        l3_weight_q = np.clip(np.round(l3_weight * QB), -128, 127).astype(np.int8)
        l3_bias_q = np.clip(np.round(l3_bias * QA * QB), -2147483648, 2147483647).astype(np.int32)
        
        for j in range(L3_SIZE):
            for i in range(L2_SIZE):
                f.write(struct.pack('<b', int(l3_weight_q[j, i])))
        
        for j in range(L3_SIZE):
            f.write(struct.pack('<i', int(l3_bias_q[j])))
        
        # Output weights: L3_SIZE as i8
        out_weight = model.out.weight.data.cpu().numpy()
        out_bias = model.out.bias.data.cpu().numpy()
        
        out_weight_q = np.clip(np.round(out_weight[0] * QB), -128, 127).astype(np.int8)
        out_bias_q = int(np.clip(np.round(out_bias[0] * QA * QB), -2147483648, 2147483647))
        
        for j in range(L3_SIZE):
            f.write(struct.pack('<b', int(out_weight_q[j])))
        
        f.write(struct.pack('<i', out_bias_q))
    
    file_size = os.path.getsize(output_file)
    print(f"Exported {output_file} ({file_size:,} bytes)")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NNUE Training on Lichess Elite Data')
    parser.add_argument('--pgn', nargs='+', required=True, help='PGN file(s) to parse')
    parser.add_argument('--output', default='nn.bin', help='Output weights file')
    parser.add_argument('--max-positions', type=int, default=5_000_000, help='Max positions to extract')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16384, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min-move', type=int, default=8, help='Skip positions before this move')
    parser.add_argument('--skip-prob', type=float, default=0.7, help='Prob of skipping a position')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    pgn_files = []
    for pattern in args.pgn:
        matched = glob.glob(pattern)
        if matched:
            pgn_files.extend(matched)
        else:
            pgn_files.append(pattern)
    
    print(f"PGN files: {pgn_files}")
    print(f"Max positions: {args.max_positions:,}")
    print(f"Skip positions before move {args.min_move}")
    print()
    
    # Parse PGNs
    data = parse_pgn_files(pgn_files, 
                          max_positions=args.max_positions,
                          min_move=args.min_move,
                          skip_prob=args.skip_prob)
    
    if not data:
        print("No training data extracted! Check PGN files.")
        sys.exit(1)
    
    # Shuffle data
    random.shuffle(data)
    
    # Train
    train_network(data, args.output, 
                 epochs=args.epochs, 
                 batch_size=args.batch_size, 
                 lr=args.lr)
    
    print("\nAll done!")
