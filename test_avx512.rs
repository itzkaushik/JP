use std::arch::x86_64::*;
#[target_feature(enable = "avx512vnni", enable = "avx512f", enable = "avx512bw")]
unsafe fn test() {
    let a = _mm512_setzero_si512();
    let b = _mm512_setzero_si512();
    let c = _mm512_setzero_si512();
    let _ = _mm512_dpbusds_epi32(a, b, c);
}
fn main() {}
