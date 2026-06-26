// =============================================================================
// nnue.rs — NNUE (Efficiently Updatable Neural Network) Evaluation
// =============================================================================
//
// Architecture: (768 → 256)×2 → 32 → 32 → 1
//
// Input features: 768 = 12 piece types × 64 squares (piece-square)
//   Feature index = color * 384 + piece * 64 + square
//   Two perspectives: White and Black (Black mirrors squares vertically)
//
// The accumulator (first layer) is incrementally updated when pieces move,
// avoiding a full matrix multiply on every evaluation call.
//
// Quantization:
//   - Feature transformer weights/biases: i16 (QA = 255)
//   - Hidden layer weights: i8 (QB = 64)
//   - Biases for hidden/output: i32

use std::path::PathBuf;
use std::{env, fs};

use crate::types::{Color, Piece, Square};

// =============================================================================
// Network dimensions
// =============================================================================

pub const INPUT_SIZE: usize = 768; // 12 pieces × 64 squares
pub const L1_SIZE: usize = 256; // Accumulator (feature transformer output)
pub const L2_SIZE: usize = 32; // Hidden layer 2
pub const L3_SIZE: usize = 32; // Hidden layer 3

/// Quantization scale for the accumulator layer (i16 weights).
const QA: i32 = 255;
/// Quantization scale for hidden layer weights (i8).
const QB: i32 = 64;
/// Output scaling: final value is divided by this to get centipawns.
const SCALE: i32 = 400;

// =============================================================================
// NNUE Accumulator
// =============================================================================

/// The incrementally-updated hidden state for the first layer.
/// Each perspective (White, Black) has its own 256-element accumulator.
#[derive(Clone, Copy)]
pub struct NnueAccumulator {
    pub white: [i16; L1_SIZE],
    pub black: [i16; L1_SIZE],
}

impl NnueAccumulator {
    /// Create a zero-initialized accumulator (must be refreshed before use).
    pub const fn empty() -> Self {
        NnueAccumulator {
            white: [0i16; L1_SIZE],
            black: [0i16; L1_SIZE],
        }
    }
}

// =============================================================================
// Feature indexing
// =============================================================================

/// Compute the feature index for a piece on a square from a given perspective.
///
/// From White's perspective:
///   index = (color * 6 + piece) * 64 + square
///
/// From Black's perspective:
///   - Colors are swapped (opponent's pieces become "ours")
///   - Square is vertically mirrored (sq ^ 56)
///   index = ((color^1) * 6 + piece) * 64 + (square ^ 56)
#[inline]
pub fn feature_index_white(color: Color, piece: Piece, sq: Square) -> usize {
    color.index() * 384 + piece.index() * 64 + sq.0 as usize
}

#[inline]
pub fn feature_index_black(color: Color, piece: Piece, sq: Square) -> usize {
    (color.index() ^ 1) * 384 + piece.index() * 64 + (sq.0 ^ 56) as usize
}

// =============================================================================
// NNUE Parameters (loaded from file)
// =============================================================================

/// The full set of neural network weights and biases.
pub struct NnueParams {
    // Feature transformer: 768 × 256 weights + 256 biases (all i16)
    pub ft_weights: Box<[[i16; L1_SIZE]; INPUT_SIZE]>,
    pub ft_biases: Box<[i16; L1_SIZE]>,

    // Hidden layer 2: 512 inputs (256×2 perspectives) → 32 outputs (row-major)
    pub l2_weights: Box<[[i8; L1_SIZE * 2]; L2_SIZE]>,
    pub l2_biases: Box<[i32; L2_SIZE]>,

    // Hidden layer 3: 32 → 32 (row-major)
    pub l3_weights: Box<[[i8; L2_SIZE]; L3_SIZE]>,
    pub l3_biases: Box<[i32; L3_SIZE]>,

    // Output layer: 32 → 1
    pub out_weights: Box<[i8; L3_SIZE]>,
    pub out_bias: i32,
}

// =============================================================================
// NNUE State (global, loaded from file)
// =============================================================================

use std::sync::OnceLock;
static NNUE_PARAMS: OnceLock<NnueParams> = OnceLock::new();

/// Load NNUE weights from a binary file. Returns true on success.
pub fn load_weights(path: &str) -> bool {
    match load_weights_from_file(path) {
        Ok(params) => {
            let _ = NNUE_PARAMS.set(params);
            true
        }
        Err(e) => {
            eprintln!(
                "info string Failed to load NNUE weights from '{}': {}",
                path, e
            );
            false
        }
    }
}

/// Load the default network from common runtime locations.
pub fn load_default_weights() -> bool {
    if is_loaded() {
        return true;
    }

    let mut candidates = Vec::new();
    candidates.push(PathBuf::from("nn.bin"));

    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("nn.bin"));
            if let Some(target_dir) = exe_dir.parent() {
                if let Some(project_dir) = target_dir.parent() {
                    candidates.push(project_dir.join("nn.bin"));
                }
            }
        }
    }

    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        let path = candidate.to_string_lossy();
        match load_weights_from_file(&path) {
            Ok(params) => {
                let _ = NNUE_PARAMS.set(params);
                return true;
            }
            Err(e) => {
                eprintln!(
                    "info string Failed to load NNUE weights from '{}': {}",
                    path, e
                );
            }
        }
    }

    false
}

/// Check if NNUE weights are loaded and available.
#[inline]
pub fn is_loaded() -> bool {
    NNUE_PARAMS.get().is_some()
}

/// Get a reference to the loaded parameters (panics if not loaded).
#[inline]
fn params() -> &'static NnueParams {
    NNUE_PARAMS.get().expect("NNUE weights not loaded")
}

// =============================================================================
// Weight file I/O
// =============================================================================

/// Binary file format:
///   Magic: "NNUE" (4 bytes)
///   Version: u32 (4 bytes) — must be 1
///   FT weights: INPUT_SIZE * L1_SIZE * 2 bytes (i16 LE)
///   FT biases:  L1_SIZE * 2 bytes (i16 LE)
///   L2 weights: (L1_SIZE*2) * L2_SIZE bytes (i8)
///   L2 biases:  L2_SIZE * 4 bytes (i32 LE)
///   L3 weights: L2_SIZE * L3_SIZE bytes (i8)
///   L3 biases:  L3_SIZE * 4 bytes (i32 LE)
///   Out weights: L3_SIZE bytes (i8)
///   Out bias:   4 bytes (i32 LE)

const MAGIC: [u8; 4] = *b"NNUE";
const VERSION: u32 = 1;

fn load_weights_from_file(path: &str) -> Result<NnueParams, String> {
    let data = fs::read(path).map_err(|e| format!("IO error: {}", e))?;
    let mut cursor = 0usize;

    // Helper to read bytes
    let read_bytes = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        if *cursor + count > data.len() {
            return Err("Unexpected end of file".to_string());
        }
        let slice = &data[*cursor..*cursor + count];
        *cursor += count;
        Ok(slice)
    };

    // Read and verify magic
    let magic = read_bytes(&mut cursor, 4)?;
    if magic != MAGIC {
        return Err(format!("Invalid magic: expected 'NNUE', got {:?}", magic));
    }

    // Read and verify version
    let ver_bytes = read_bytes(&mut cursor, 4)?;
    let version = u32::from_le_bytes(ver_bytes.try_into().unwrap());
    if version != VERSION {
        return Err(format!("Unsupported version: {}", version));
    }

    // Read FT weights: INPUT_SIZE × L1_SIZE i16 values
    let mut ft_weights = Box::new([[0i16; L1_SIZE]; INPUT_SIZE]);
    for i in 0..INPUT_SIZE {
        for j in 0..L1_SIZE {
            let bytes = read_bytes(&mut cursor, 2)?;
            ft_weights[i][j] = i16::from_le_bytes(bytes.try_into().unwrap());
        }
    }

    // Read FT biases: L1_SIZE i16 values
    let mut ft_biases = Box::new([0i16; L1_SIZE]);
    for j in 0..L1_SIZE {
        let bytes = read_bytes(&mut cursor, 2)?;
        ft_biases[j] = i16::from_le_bytes(bytes.try_into().unwrap());
    }

    // Read L2 weights: L2_SIZE rows of (L1_SIZE*2) i8 values
    let mut l2_weights = Box::new([[0i8; L1_SIZE * 2]; L2_SIZE]);
    for j in 0..L2_SIZE {
        for i in 0..(L1_SIZE * 2) {
            let bytes = read_bytes(&mut cursor, 1)?;
            l2_weights[j][i] = bytes[0] as i8;
        }
    }

    // Read L2 biases: L2_SIZE i32 values
    let mut l2_biases = Box::new([0i32; L2_SIZE]);
    for j in 0..L2_SIZE {
        let bytes = read_bytes(&mut cursor, 4)?;
        l2_biases[j] = i32::from_le_bytes(bytes.try_into().unwrap());
    }

    // Read L3 weights: L3_SIZE rows of L2_SIZE i8 values
    let mut l3_weights = Box::new([[0i8; L2_SIZE]; L3_SIZE]);
    for j in 0..L3_SIZE {
        for i in 0..L2_SIZE {
            let bytes = read_bytes(&mut cursor, 1)?;
            l3_weights[j][i] = bytes[0] as i8;
        }
    }

    // Read L3 biases: L3_SIZE i32 values
    let mut l3_biases = Box::new([0i32; L3_SIZE]);
    for j in 0..L3_SIZE {
        let bytes = read_bytes(&mut cursor, 4)?;
        l3_biases[j] = i32::from_le_bytes(bytes.try_into().unwrap());
    }

    // Read output weights: L3_SIZE i8 values
    let mut out_weights = Box::new([0i8; L3_SIZE]);
    for j in 0..L3_SIZE {
        let bytes = read_bytes(&mut cursor, 1)?;
        out_weights[j] = bytes[0] as i8;
    }

    // Read output bias: i32
    let bias_bytes = read_bytes(&mut cursor, 4)?;
    let out_bias = i32::from_le_bytes(bias_bytes.try_into().unwrap());

    Ok(NnueParams {
        ft_weights,
        ft_biases,
        l2_weights,
        l2_biases,
        l3_weights,
        l3_biases,
        out_weights,
        out_bias,
    })
}

// =============================================================================
// Accumulator operations
// =============================================================================

/// Compute the accumulator from scratch for the given board.
/// This iterates over all pieces and sums the corresponding weight rows.
pub fn refresh_accumulator(board: &crate::board::Board) -> NnueAccumulator {
    let p = params();
    let mut acc = NnueAccumulator {
        white: *p.ft_biases,
        black: *p.ft_biases,
    };

    // Iterate over all pieces on the board
    for sq_idx in 0..64u8 {
        let sq = Square(sq_idx);
        if let Some((color, piece)) = board.piece_at(sq) {
            let wi = feature_index_white(color, piece, sq);
            let bi = feature_index_black(color, piece, sq);
            for j in 0..L1_SIZE {
                acc.white[j] += p.ft_weights[wi][j];
                acc.black[j] += p.ft_weights[bi][j];
            }
        }
    }

    acc
}

/// Add a piece feature to the accumulator (piece placed on square).
#[inline]
pub fn acc_add(acc: &mut NnueAccumulator, color: Color, piece: Piece, sq: Square) {
    if !is_loaded() {
        return;
    }
    let p = params();
    let wi = feature_index_white(color, piece, sq);
    let bi = feature_index_black(color, piece, sq);
    for j in 0..L1_SIZE {
        acc.white[j] += p.ft_weights[wi][j];
        acc.black[j] += p.ft_weights[bi][j];
    }
}

/// Remove a piece feature from the accumulator (piece removed from square).
#[inline]
pub fn acc_remove(acc: &mut NnueAccumulator, color: Color, piece: Piece, sq: Square) {
    if !is_loaded() {
        return;
    }
    let p = params();
    let wi = feature_index_white(color, piece, sq);
    let bi = feature_index_black(color, piece, sq);
    for j in 0..L1_SIZE {
        acc.white[j] -= p.ft_weights[wi][j];
        acc.black[j] -= p.ft_weights[bi][j];
    }
}

// =============================================================================
// Forward pass (inference)
// =============================================================================

/// Clipped ReLU: clamp(x, 0, QA) where QA = 255.
#[inline]
fn crelu(x: i16) -> i32 {
    (x as i32).clamp(0, QA)
}

/// Run the full NNUE forward pass on the accumulator.
/// Returns the evaluation in centipawns from the perspective of `side_to_move`.
pub fn forward_scalar(acc: &NnueAccumulator, side_to_move: Color) -> i32 {
    let p = params();

    // --- Arrange perspectives ---
    // The side-to-move's accumulator comes first, then the opponent's
    let (us_acc, them_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    // --- Layer 2: CReLU(accumulator) × L2 weights + L2 biases ---
    let mut l2_out = [0i32; L2_SIZE];
    for j in 0..L2_SIZE {
        let mut sum = p.l2_biases[j];

        // First half: our perspective (0..L1_SIZE)
        for i in 0..L1_SIZE {
            sum += crelu(us_acc[i]) * p.l2_weights[j][i] as i32;
        }
        // Second half: their perspective (L1_SIZE..L1_SIZE*2)
        for i in 0..L1_SIZE {
            sum += crelu(them_acc[i]) * p.l2_weights[j][L1_SIZE + i] as i32;
        }

        l2_out[j] = sum;
    }

    // --- Layer 3: CReLU(L2 output / (QA * QB)) × L3 weights + L3 biases ---
    let mut l3_out = [0i32; L3_SIZE];
    for j in 0..L3_SIZE {
        let mut sum = p.l3_biases[j];
        for i in 0..L2_SIZE {
            // De-quantize L2 output, apply CReLU, re-quantize
            let activated = (l2_out[i] / QB).clamp(0, QA);
            sum += activated * p.l3_weights[j][i] as i32;
        }
        l3_out[j] = sum;
    }

    // --- Output: CReLU(L3 output / (QA * QB)) × out_weights + out_bias ---
    let mut output = p.out_bias;
    for i in 0..L3_SIZE {
        let activated = (l3_out[i] / QB).clamp(0, QA);
        output += activated * p.out_weights[i] as i32;
    }

    // Scale to centipawns
    output * SCALE / (QA * QB)
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn forward_avx2(acc: &NnueAccumulator, side_to_move: Color) -> i32 {
    let p = params();

    let (us_acc, them_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let mut l2_out = [0i32; L2_SIZE];
    let ones = _mm256_set1_epi16(1);

    for j in 0..L2_SIZE {
        let mut sum_vec = _mm256_setzero_si256();

        let us_ptr = us_acc.as_ptr() as *const __m256i;
        let them_ptr = them_acc.as_ptr() as *const __m256i;
        let w_ptr = p.l2_weights[j].as_ptr() as *const __m256i;

        let mut process_acc = |acc_ptr: *const __m256i, w_offset: isize| unsafe {
            for i in 0..8 {
                let u1 = _mm256_loadu_si256(acc_ptr.offset(i * 2));
                let u2 = _mm256_loadu_si256(acc_ptr.offset(i * 2 + 1));

                let packed = _mm256_packus_epi16(u1, u2);
                let u8_vec = _mm256_permute4x64_epi64(packed, 0xD8);

                let weights = _mm256_loadu_si256(w_ptr.offset(w_offset + i));

                let madd = _mm256_maddubs_epi16(u8_vec, weights);
                let madd32 = _mm256_madd_epi16(madd, ones);

                sum_vec = _mm256_add_epi32(sum_vec, madd32);
            }
        };

        process_acc(us_ptr, 0);
        process_acc(them_ptr, 8);

        let x128 = _mm_add_epi32(
            _mm256_castsi256_si128(sum_vec),
            _mm256_extracti128_si256(sum_vec, 1),
        );
        let x64 = _mm_add_epi32(x128, _mm_shuffle_epi32(x128, 0x0E));
        let x32 = _mm_add_epi32(x64, _mm_shuffle_epi32(x64, 0x01));
        let final_sum = _mm_cvtsi128_si32(x32);

        l2_out[j] = p.l2_biases[j] + final_sum;
    }

    let mut l3_out = [0i32; L3_SIZE];
    for j in 0..L3_SIZE {
        let mut sum = p.l3_biases[j];
        for i in 0..L2_SIZE {
            let activated = (l2_out[i] / QB).clamp(0, QA);
            sum += activated * p.l3_weights[j][i] as i32;
        }
        l3_out[j] = sum;
    }

    let mut output = p.out_bias;
    for i in 0..L3_SIZE {
        let activated = (l3_out[i] / QB).clamp(0, QA);
        output += activated * p.out_weights[i] as i32;
    }

    output * SCALE / (QA * QB)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
pub unsafe fn forward_avx512vnni(acc: &NnueAccumulator, side_to_move: Color) -> i32 {
    let p = params();

    let (us_acc, them_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let mut l2_out = [0i32; L2_SIZE];
    // Permutation mask to linearize the 512-bit packus output
    let perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    for j in 0..L2_SIZE {
        let mut sum_vec = _mm512_setzero_si512();

        let us_ptr = us_acc.as_ptr() as *const __m512i;
        let them_ptr = them_acc.as_ptr() as *const __m512i;
        let w_ptr = p.l2_weights[j].as_ptr() as *const __m512i;

        let mut process_acc = |acc_ptr: *const __m512i, w_offset: isize| unsafe {
            for i in 0..4 {
                let u1 = _mm512_loadu_si512(acc_ptr.offset(i * 2));
                let u2 = _mm512_loadu_si512(acc_ptr.offset(i * 2 + 1));

                let packed = _mm512_packus_epi16(u1, u2);
                let u8_vec = _mm512_permutexvar_epi64(perm_idx, packed);

                let weights = _mm512_loadu_si512(w_ptr.offset(w_offset + i));

                // VNNI fuses multiply and accumulate for u8 * i8 -> i32
                sum_vec = _mm512_dpbusds_epi32(sum_vec, u8_vec, weights);
            }
        };

        process_acc(us_ptr, 0);
        process_acc(them_ptr, 4);

        // Horizontal sum of the 16 i32s in sum_vec
        let x256 = _mm256_add_epi32(
            _mm512_castsi512_si256(sum_vec),
            _mm512_extracti64x4_epi64(sum_vec, 1),
        );
        let x128 = _mm_add_epi32(
            _mm256_castsi256_si128(x256),
            _mm256_extracti128_si256(x256, 1),
        );
        let x64 = _mm_add_epi32(x128, _mm_shuffle_epi32(x128, 0x0E));
        let x32 = _mm_add_epi32(x64, _mm_shuffle_epi32(x64, 0x01));
        let final_sum = _mm_cvtsi128_si32(x32);

        l2_out[j] = p.l2_biases[j] + final_sum;
    }

    let mut l3_out = [0i32; L3_SIZE];
    for j in 0..L3_SIZE {
        let mut sum = p.l3_biases[j];
        for i in 0..L2_SIZE {
            let activated = (l2_out[i] / QB).clamp(0, QA);
            sum += activated * p.l3_weights[j][i] as i32;
        }
        l3_out[j] = sum;
    }

    let mut output = p.out_bias;
    for i in 0..L3_SIZE {
        let activated = (l3_out[i] / QB).clamp(0, QA);
        output += activated * p.out_weights[i] as i32;
    }

    output * SCALE / (QA * QB)
}

#[inline(always)]
pub fn forward(acc: &NnueAccumulator, side_to_move: Color) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512vnni")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
    {
        return unsafe { forward_avx512vnni(acc, side_to_move) };
    }

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { forward_avx2(acc, side_to_move) };
    }

    forward_scalar(acc, side_to_move)
}

// =============================================================================
// High-level evaluation entry point
// =============================================================================

/// Evaluate the board using NNUE. Returns score in centipawns from side-to-move.
/// Computes the accumulator from scratch (non-incremental, for fallback/testing).
pub fn evaluate_from_scratch(board: &crate::board::Board) -> i32 {
    let acc = refresh_accumulator(board);
    forward(&acc, board.side_to_move)
}

// =============================================================================
// NnueState — External accumulator stack for the search
// =============================================================================

const MAX_PLY: usize = 128;

/// Maintains a stack of NNUE accumulators, one per search ply.
/// The search refreshes at the root and updates incrementally per move.
pub struct NnueState {
    accumulators: Vec<NnueAccumulator>,
}

impl NnueState {
    pub fn new() -> Self {
        let mut accumulators = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            accumulators.push(NnueAccumulator::empty());
        }
        NnueState { accumulators }
    }

    /// Refresh the accumulator at a given ply from scratch.
    /// Call this at the root of the search.
    pub fn refresh(&mut self, ply: usize, board: &crate::board::Board) {
        if !is_loaded() {
            return;
        }
        self.accumulators[ply] = refresh_accumulator(board);
    }

    /// Copy accumulator from parent ply (null move — no piece changes).
    pub fn copy_ply(&mut self, ply: usize) {
        if !is_loaded() || ply == 0 {
            return;
        }
        self.accumulators[ply] = self.accumulators[ply - 1];
    }

    /// Incrementally update accumulator for a move that has already been made.
    pub fn apply_move(
        &mut self,
        ply: usize,
        board: &crate::board::Board,
        undo: crate::board::Undo,
    ) {
        if !is_loaded() || ply == 0 {
            return;
        }
        let parent_ply = ply - 1;
        let mut acc = self.accumulators[parent_ply];

        let mv = undo.mv;
        let from = mv.from_sq();
        let to = mv.to_sq();
        let us = board.side_to_move.flip();

        // Remove moving piece from source
        acc_remove(&mut acc, us, undo.moved_piece, from);

        // Remove captured piece
        if let Some((color, piece, sq)) = undo.captured {
            acc_remove(&mut acc, color, piece, sq);
        }

        // Place piece on destination
        if let Some(promo) = mv.promotion_piece() {
            acc_add(&mut acc, us, promo, to);
        } else {
            acc_add(&mut acc, us, undo.moved_piece, to);
        }

        // Castling: move the rook
        if mv.is_castle() {
            match (us, mv.flags()) {
                (Color::White, crate::types::Move::FLAG_KS_CASTLE) => {
                    acc_remove(&mut acc, Color::White, Piece::Rook, Square(7)); // h1
                    acc_add(&mut acc, Color::White, Piece::Rook, Square(5)); // f1
                }
                (Color::White, crate::types::Move::FLAG_QS_CASTLE) => {
                    acc_remove(&mut acc, Color::White, Piece::Rook, Square(0)); // a1
                    acc_add(&mut acc, Color::White, Piece::Rook, Square(3)); // d1
                }
                (Color::Black, crate::types::Move::FLAG_KS_CASTLE) => {
                    acc_remove(&mut acc, Color::Black, Piece::Rook, Square(63)); // h8
                    acc_add(&mut acc, Color::Black, Piece::Rook, Square(61)); // f8
                }
                (Color::Black, crate::types::Move::FLAG_QS_CASTLE) => {
                    acc_remove(&mut acc, Color::Black, Piece::Rook, Square(56)); // a8
                    acc_add(&mut acc, Color::Black, Piece::Rook, Square(59)); // d8
                }
                _ => {}
            }
        }

        self.accumulators[ply] = acc;
    }

    /// Evaluate using the accumulator at the given ply.
    #[inline]
    pub fn evaluate(&self, ply: usize, side_to_move: Color) -> i32 {
        forward(&self.accumulators[ply], side_to_move)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn load_test_net() -> bool {
        if !std::path::Path::new("nn.bin").is_file() {
            return false;
        }
        let _ = load_weights("nn.bin");
        is_loaded()
    }

    fn assert_incremental_matches_refresh(fen: &str, uci: &str) {
        if !load_test_net() {
            return;
        }

        let mut board = Board::from_fen(fen).unwrap();
        let mv = board
            .generate_legal_moves()
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("move {} not legal in {}", uci, fen));

        let mut state = NnueState::new();
        state.refresh(0, &board);

        let undo = board.do_move(mv);
        state.apply_move(1, &board, undo);

        let refreshed = refresh_accumulator(&board);
        assert_eq!(state.accumulators[1].white, refreshed.white);
        assert_eq!(state.accumulators[1].black, refreshed.black);
    }

    #[test]
    fn test_feature_index_white_bounds() {
        for &color in &[Color::White, Color::Black] {
            for &piece in &Piece::ALL {
                for sq in 0..64u8 {
                    let idx = feature_index_white(color, piece, Square(sq));
                    assert!(idx < INPUT_SIZE, "White index {} out of range", idx);
                }
            }
        }
    }

    #[test]
    fn test_feature_index_black_bounds() {
        for &color in &[Color::White, Color::Black] {
            for &piece in &Piece::ALL {
                for sq in 0..64u8 {
                    let idx = feature_index_black(color, piece, Square(sq));
                    assert!(idx < INPUT_SIZE, "Black index {} out of range", idx);
                }
            }
        }
    }

    #[test]
    fn test_feature_indices_symmetry() {
        // A white pawn on e2 from white's perspective should match
        // a black pawn on e7 from black's perspective (mirrored)
        let w_idx = feature_index_white(Color::White, Piece::Pawn, Square(12)); // e2
        let b_idx = feature_index_black(Color::Black, Piece::Pawn, Square(52)); // e7 → mirrors to e2
        assert_eq!(w_idx, b_idx, "Symmetric features should match");
    }

    #[test]
    fn test_incremental_quiet_matches_refresh() {
        assert_incremental_matches_refresh(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4",
        );
    }

    #[test]
    fn test_incremental_capture_matches_refresh() {
        assert_incremental_matches_refresh(
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2",
            "d4e5",
        );
    }

    #[test]
    fn test_incremental_castle_matches_refresh() {
        assert_incremental_matches_refresh("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1");
    }

    #[test]
    fn test_incremental_promotion_matches_refresh() {
        assert_incremental_matches_refresh("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7g8q");
    }

    #[test]
    fn test_incremental_en_passant_matches_refresh() {
        assert_incremental_matches_refresh("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
    }
}
