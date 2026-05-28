#![no_std]
use core::cell::UnsafeCell;

// We use an UnsafeCell to hold the Catalytic Tape, allowing the outer 
// host universe to read and write directly into our memory space, bypassing safety limits.
struct SyncUnsafeCell<T>(UnsafeCell<T>);
unsafe impl<T> Sync for SyncUnsafeCell<T> {}

static TAPE: SyncUnsafeCell<[u8; 256]> = SyncUnsafeCell(UnsafeCell::new([0; 256]));

#[unsafe(no_mangle)]
pub extern "C" fn get_tape_ptr() -> *mut u8 {
    TAPE.0.get() as *mut u8
}

fn f_func(half: &[u8; 128], seed: u32) -> [u8; 128] {
    let mut out = [0u8; 128];
    for i in 0..128 {
        let dist = (i as i32 - 64).abs() as u32;
        let weight = 255 / (1 + dist);
        // The quantum seed dictates the magnitude of the gravitational warp
        out[i] = ((half[i] as u32 + weight * seed) % 256) as u8;
    }
    out
}

fn calc_variance(tape: &[u8; 256]) -> f64 {
    let sum: u32 = tape.iter().map(|&x| x as u32).sum();
    let mean = sum as f64 / 256.0;
    
    let mut var_sum = 0.0;
    for &x in tape.iter() {
        let diff = x as f64 - mean;
        var_sum += diff * diff;
    }
    var_sum / 256.0
}

#[unsafe(no_mangle)]
pub extern "C" fn apply_gravity_well(seed: u32) -> f64 {
    unsafe {
        let tape = &mut *TAPE.0.get();
        let initial_variance = calc_variance(tape);
        
        let mut l = [0u8; 128];
        let mut r = [0u8; 128];
        l.copy_from_slice(&tape[0..128]);
        r.copy_from_slice(&tape[128..256]);
        
        // Feistel Round 1
        let f_l = f_func(&l, seed);
        for i in 0..128 { r[i] ^= f_l[i]; }
        
        // Feistel Round 2
        let f_r = f_func(&r, seed);
        for i in 0..128 { l[i] ^= f_r[i]; }
        
        tape[0..128].copy_from_slice(&l);
        tape[128..256].copy_from_slice(&r);
        
        let final_variance = calc_variance(tape);
        let shift = final_variance - initial_variance;
        if shift < 0.0 { -shift } else { shift }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn inverse_gravity_well(seed: u32) {
    unsafe {
        let tape = &mut *TAPE.0.get();
        let mut l = [0u8; 128];
        let mut r = [0u8; 128];
        l.copy_from_slice(&tape[0..128]);
        r.copy_from_slice(&tape[128..256]);
        
        // Inverse Feistel Round 2
        let f_r = f_func(&r, seed);
        for i in 0..128 { l[i] ^= f_r[i]; }
        
        // Inverse Feistel Round 1
        let f_l = f_func(&l, seed);
        for i in 0..128 { r[i] ^= f_l[i]; }
        
        tape[0..128].copy_from_slice(&l);
        tape[128..256].copy_from_slice(&r);
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
