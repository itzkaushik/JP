use std::arch::x86_64::*;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn test_pack512() {
    let mut a_arr = [0i16; 32];
    let mut b_arr = [0i16; 32];
    for i in 0..32 {
        a_arr[i] = i as i16;
        b_arr[i] = (i + 32) as i16;
    }
    
    let a = _mm512_loadu_si512(a_arr.as_ptr() as *const __m512i);
    let b = _mm512_loadu_si512(b_arr.as_ptr() as *const __m512i);
    
    let packed = _mm512_packus_epi16(a, b);
    let perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0); 
    let permuted = _mm512_permutexvar_epi64(perm_idx, packed);
    
    let mut out = [0u8; 64];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, permuted);
    
    println!("{:?}", out.to_vec());
}

fn main() {
    unsafe { test_pack512(); }
}
