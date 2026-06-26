#!/usr/bin/env python3
"""
NNUE Training Script for ChessEngine
=====================================

Architecture: (768 → 256)×2 → 32 → 32 → 1

This script:
1. Generates training data by evaluating random positions with the engine's HCE
2. Trains a neural network to approximate the evaluation
3. Exports quantized weights to a binary .bin file

Usage:
    python train.py --generate    # Generate training data
    python train.py --train       # Train network on generated data
    python train.py --all         # Generate + Train + Export

Requirements: pip install python-chess torch numpy
"""

import argparse
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
    from torch.utils.data import Dataset, DataLoader
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
SCALE = 400         # Output scale

# =============================================================================
# Feature extraction
# =============================================================================

def board_to_features(board: chess.Board):
    """Convert a python-chess Board to a 768-element binary feature vector.
    
    Feature index = color * 384 + piece_type * 64 + square
    where piece_type is 0-5 (pawn..king) and color is 0 (white) or 1 (black).
    
    Returns features from BOTH perspectives (white and black).
    """
    white_features = []
    black_features = []
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            color_idx = 0 if piece.color == chess.WHITE else 1
            piece_idx = piece.piece_type - 1  # chess.PAWN=1, so -1 to get 0-based
            
            # White perspective: straightforward
            w_idx = color_idx * 384 + piece_idx * 64 + sq
            white_features.append(w_idx)
            
            # Black perspective: swap colors and mirror square
            b_idx = (color_idx ^ 1) * 384 + piece_idx * 64 + (sq ^ 56)
            black_features.append(b_idx)
    
    return white_features, black_features


def features_to_tensor(white_features, black_features):
    """Convert feature indices to dense tensors."""
    w = torch.zeros(INPUT_SIZE)
    b = torch.zeros(INPUT_SIZE)
    for idx in white_features:
        w[idx] = 1.0
    for idx in black_features:
        b[idx] = 1.0
    return w, b

# =============================================================================
# Data generation
# =============================================================================

# PeSTO piece-square tables for generating training labels
# (Simplified — matches our engine's eval approximately)
MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]

MG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
]

MG_KNIGHT_TABLE = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]

MG_BISHOP_TABLE = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

MG_ROOK_TABLE = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
]

MG_QUEEN_TABLE = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

MG_KING_TABLE = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

EG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

EG_KNIGHT_TABLE = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

EG_BISHOP_TABLE = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

EG_ROOK_TABLE = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

EG_QUEEN_TABLE = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

EG_KING_TABLE = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

MG_TABLES = [MG_PAWN_TABLE, MG_KNIGHT_TABLE, MG_BISHOP_TABLE, MG_ROOK_TABLE, MG_QUEEN_TABLE, MG_KING_TABLE]
EG_TABLES = [EG_PAWN_TABLE, EG_KNIGHT_TABLE, EG_BISHOP_TABLE, EG_ROOK_TABLE, EG_QUEEN_TABLE, EG_KING_TABLE]

GAME_PHASE_INC = [0, 1, 1, 2, 4, 0]

def simple_eval(board: chess.Board) -> float:
    """Simple PeSTO-like evaluation for labeling positions.
    Returns score in centipawns from White's perspective."""
    mg_score = [0, 0]
    eg_score = [0, 0]
    game_phase = 0
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        
        color_idx = 0 if piece.color == chess.WHITE else 1
        piece_idx = piece.piece_type - 1  # 0-based
        
        # PeSTO table lookup: white pieces flip rank, black uses as-is
        table_idx = sq ^ 56 if piece.color == chess.WHITE else sq
        
        mg_val = MG_VALUE[piece_idx] + MG_TABLES[piece_idx][table_idx]
        eg_val = EG_VALUE[piece_idx] + EG_TABLES[piece_idx][table_idx]
        
        mg_score[color_idx] += mg_val
        eg_score[color_idx] += eg_val
        game_phase += GAME_PHASE_INC[piece_idx]
    
    mg_phase = min(game_phase, 24)
    eg_phase = 24 - mg_phase
    
    mg_diff = mg_score[0] - mg_score[1]
    eg_diff = eg_score[0] - eg_score[1]
    
    score = (mg_diff * mg_phase + eg_diff * eg_phase) / 24.0
    return score


def generate_random_position(board: chess.Board, random_moves: int = 8):
    """Play random legal moves from starting position to get a diverse position."""
    board.reset()
    for _ in range(random_moves):
        legal = list(board.legal_moves)
        if not legal:
            return False
        board.push(random.choice(legal))
    return True


def generate_training_data(output_file: str, num_positions: int = 5_000_000):
    """Generate training data: random positions with PeSTO evaluations."""
    print(f"Generating {num_positions:,} training positions...")
    
    board = chess.Board()
    data = []
    
    start_time = time.time()
    
    for i in range(num_positions):
        # Generate random position (4-12 random moves from start)
        n_moves = random.randint(4, 12)
        if not generate_random_position(board, n_moves):
            continue
        
        # Skip positions where the side to move is in check
        if board.is_check():
            continue
        
        # Skip positions with too few pieces
        if len(board.piece_map()) < 6:
            continue
        
        # Evaluate from White's perspective
        score = simple_eval(board)
        
        # Get features
        w_feats, b_feats = board_to_features(board)
        stm = 0 if board.turn == chess.WHITE else 1
        
        data.append((w_feats, b_feats, stm, score))
        
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Generated {i+1:,} positions ({rate:.0f}/s)")
    
    # Save to binary file
    print(f"Saving {len(data):,} positions to {output_file}...")
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<I', len(data)))  # Number of entries
        
        for w_feats, b_feats, stm, score in data:
            # Write: num_white_features (u8), white indices (u16 each),
            #        num_black_features (u8), black indices (u16 each),
            #        stm (u8), score (f32)
            f.write(struct.pack('<B', len(w_feats)))
            for idx in w_feats:
                f.write(struct.pack('<H', idx))
            f.write(struct.pack('<B', len(b_feats)))
            for idx in b_feats:
                f.write(struct.pack('<H', idx))
            f.write(struct.pack('<B', stm))
            f.write(struct.pack('<f', score))
    
    elapsed = time.time() - start_time
    print(f"Done! Generated {len(data):,} positions in {elapsed:.1f}s")
    return output_file

# =============================================================================
# PyTorch model
# =============================================================================

class ChessNNUE(nn.Module):
    """NNUE-style network: (768 → 256)×2 → 32 → 32 → 1"""
    
    def __init__(self):
        super().__init__()
        # Feature transformer (shared weights, separate accumulators)
        self.ft = nn.Linear(INPUT_SIZE, L1_SIZE)
        # Hidden layers
        self.l2 = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l3 = nn.Linear(L2_SIZE, L3_SIZE)
        self.out = nn.Linear(L3_SIZE, 1)
    
    def forward(self, white_features, black_features, stm):
        """
        white_features: (batch, 768) binary
        black_features: (batch, 768) binary
        stm: (batch, 1) - 0 for white, 1 for black
        """
        # Apply feature transformer to both perspectives
        w_acc = torch.clamp(self.ft(white_features), 0.0, 1.0)  # CReLU [0, 1]
        b_acc = torch.clamp(self.ft(black_features), 0.0, 1.0)
        
        # Concatenate: side-to-move first, then opponent
        # When stm=0 (White): [white_acc, black_acc]
        # When stm=1 (Black): [black_acc, white_acc]
        stm_expanded = stm.unsqueeze(1)  # (batch, 1)
        
        us_acc = torch.where(stm_expanded == 0, w_acc, b_acc)
        them_acc = torch.where(stm_expanded == 0, b_acc, w_acc)
        
        combined = torch.cat([us_acc, them_acc], dim=1)  # (batch, 512)
        
        # Hidden layers with CReLU
        x = torch.clamp(self.l2(combined), 0.0, 1.0)
        x = torch.clamp(self.l3(x), 0.0, 1.0)
        x = self.out(x)
        
        return x.squeeze(1)  # (batch,)

# =============================================================================
# Dataset
# =============================================================================

class ChessDataset(Dataset):
    def __init__(self, data_file: str):
        print(f"Loading training data from {data_file}...")
        self.samples = []
        
        with open(data_file, 'rb') as f:
            n_entries = struct.unpack('<I', f.read(4))[0]
            
            for i in range(n_entries):
                # Read white features
                n_w = struct.unpack('<B', f.read(1))[0]
                w_indices = [struct.unpack('<H', f.read(2))[0] for _ in range(n_w)]
                
                # Read black features
                n_b = struct.unpack('<B', f.read(1))[0]
                b_indices = [struct.unpack('<H', f.read(2))[0] for _ in range(n_b)]
                
                stm = struct.unpack('<B', f.read(1))[0]
                score = struct.unpack('<f', f.read(4))[0]
                
                self.samples.append((w_indices, b_indices, stm, score))
                
                if (i + 1) % 500_000 == 0:
                    print(f"  Loaded {i+1:,}/{n_entries:,} positions")
        
        print(f"Loaded {len(self.samples):,} positions")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_batch(batch):
    batch_size = len(batch)
    w_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    b_dense = torch.zeros(batch_size, INPUT_SIZE, dtype=torch.float32)
    stm_list = []
    scores_list = []

    for i, (w_feats, b_feats, stm, score) in enumerate(batch):
        if w_feats:
            w_dense[i, w_feats] = 1.0
        if b_feats:
            b_dense[i, b_feats] = 1.0
        stm_list.append(stm)
        scores_list.append(score)

    return (
        w_dense,
        b_dense,
        torch.tensor(stm_list, dtype=torch.float32),
        torch.tensor(scores_list, dtype=torch.float32),
    )

# =============================================================================
# Training
# =============================================================================

def train_network(data_file: str, output_file: str, epochs: int = 30, 
                  batch_size: int = 16384, lr: float = 0.001):
    """Train the NNUE network."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    dataset = ChessDataset(data_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       collate_fn=collate_batch, num_workers=0, pin_memory=True)
    
    model = ChessNNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    
    print(f"\nTraining for {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for w_feats, b_feats, stm, targets in loader:
            w_feats = w_feats.to(device)
            b_feats = b_feats.to(device)
            stm = stm.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(w_feats, b_feats, stm)
            
            # MSE loss (score prediction)
            loss = nn.functional.mse_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:3d}/{epochs}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Export weights
    print(f"\nExporting quantized weights to {output_file}...")
    export_weights(model, output_file)
    print("Done!")

# =============================================================================
# Weight export (quantization)
# =============================================================================

def export_weights(model: ChessNNUE, output_file: str):
    """Export model weights in quantized binary format matching nnue.rs."""
    
    with open(output_file, 'wb') as f:
        # Magic: "NNUE"
        f.write(b'NNUE')
        # Version: 1
        f.write(struct.pack('<I', 1))
        
        # Feature transformer weights: INPUT_SIZE × L1_SIZE as i16
        # The PyTorch layer is (INPUT_SIZE, L1_SIZE), weight shape = (L1_SIZE, INPUT_SIZE)
        ft_weight = model.ft.weight.data.cpu().numpy()  # (L1_SIZE, INPUT_SIZE)
        ft_bias = model.ft.bias.data.cpu().numpy()      # (L1_SIZE,)
        
        # Quantize: multiply by QA (255), round to i16
        ft_weight_q = np.clip(np.round(ft_weight * QA), -32768, 32767).astype(np.int16)
        ft_bias_q = np.clip(np.round(ft_bias * QA), -32768, 32767).astype(np.int16)
        
        # Write FT weights: for each input feature, write L1_SIZE i16 values
        for i in range(INPUT_SIZE):
            for j in range(L1_SIZE):
                f.write(struct.pack('<h', int(ft_weight_q[j, i])))
        
        # Write FT biases: L1_SIZE i16 values
        for j in range(L1_SIZE):
            f.write(struct.pack('<h', int(ft_bias_q[j])))
        
        # L2 weights: (L1_SIZE*2) × L2_SIZE as i8
        l2_weight = model.l2.weight.data.cpu().numpy()  # (L2_SIZE, L1_SIZE*2)
        l2_bias = model.l2.bias.data.cpu().numpy()      # (L2_SIZE,)
        
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
        out_weight = model.out.weight.data.cpu().numpy()  # (1, L3_SIZE)
        out_bias = model.out.bias.data.cpu().numpy()      # (1,)
        
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
    parser = argparse.ArgumentParser(description='NNUE Training for ChessEngine')
    parser.add_argument('--generate', action='store_true', help='Generate training data')
    parser.add_argument('--train', action='store_true', help='Train network')
    parser.add_argument('--all', action='store_true', help='Generate + Train + Export')
    parser.add_argument('--data-file', default='training_data.bin', help='Training data file')
    parser.add_argument('--output', default='nn.bin', help='Output weights file')
    parser.add_argument('--positions', type=int, default=5_000_000, help='Number of positions to generate')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16384, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.all:
        args.generate = True
        args.train = True
    
    if not args.generate and not args.train:
        parser.print_help()
        sys.exit(1)
    
    if args.generate:
        generate_training_data(args.data_file, args.positions)
    
    if args.train:
        train_network(args.data_file, args.output, 
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    
    print("\nAll done!")
