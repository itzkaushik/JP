use crate::attacks;
use crate::types::BitBoard;
use pyrrhic_rs::{Color as SyzygyColor, EngineAdapter};

#[derive(Clone)]
pub struct SyzygyAdapter;

impl EngineAdapter for SyzygyAdapter {
    fn pawn_attacks(color: SyzygyColor, sq: u64) -> u64 {
        let our_color = if color == SyzygyColor::Black {
            crate::types::Color::Black
        } else {
            crate::types::Color::White
        };
        attacks::pawn_attacks(our_color, crate::types::Square(sq as u8)).0
    }

    fn knight_attacks(sq: u64) -> u64 {
        attacks::knight_attacks(crate::types::Square(sq as u8)).0
    }

    fn bishop_attacks(sq: u64, occ: u64) -> u64 {
        attacks::bishop_attacks(crate::types::Square(sq as u8), BitBoard(occ)).0
    }

    fn rook_attacks(sq: u64, occ: u64) -> u64 {
        attacks::rook_attacks(crate::types::Square(sq as u8), BitBoard(occ)).0
    }

    fn king_attacks(sq: u64) -> u64 {
        attacks::king_attacks(crate::types::Square(sq as u8)).0
    }

    fn queen_attacks(sq: u64, occ: u64) -> u64 {
        (attacks::bishop_attacks(crate::types::Square(sq as u8), BitBoard(occ))
            | attacks::rook_attacks(crate::types::Square(sq as u8), BitBoard(occ)))
        .0
    }
}
