use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn test_pack() {
    let mut a_arr = [0i16; 16];
    let mut b_arr = [0i16; 16];
    for i in 0..16 {
        a_arr[i] = i as i16;
        b_arr[i] = (i + 16) as i16;
    }
    
    let a = _mm256_loadu_si256(a_arr.as_ptr() as *const __m256i);
    let b = _mm256_loadu_si256(b_arr.as_ptr() as *const __m256i);
    
    let packed = _mm256_packus_epi16(a, b);
    let permuted = _mm256_permute4x64_epi64(packed, 0xD8);
    
    let mut out = [0u8; 32];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, permuted);
    
    println!("{:?}", out);
}

fn main() {
    unsafe { test_pack(); }
}
