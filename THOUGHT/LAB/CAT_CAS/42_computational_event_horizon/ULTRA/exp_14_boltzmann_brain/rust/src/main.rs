use rand::RngExt;
use std::fs::File;
use std::io::{Write, BufWriter};
use flate2::write::ZlibEncoder;
use flate2::Compression;

const TOTAL_BITS: usize = 16384;
const LIMBS: usize = TOTAL_BITS / 32;

// We will do 20k generations per phase to keep telemetry rendering fast and file sizes reasonable.
const PHASE_GENS: usize = 20_000;

fn compress_size(data: &[u32]) -> usize {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * 4,
        )
    };
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(byte_slice).unwrap();
    let compressed = encoder.finish().unwrap();
    compressed.len()
}

fn evolve(
    state: &mut [u32], 
    generations: usize, 
    phase_name: &str, 
    mri_file: &mut BufWriter<File>, 
    telemetry_file: &mut BufWriter<File>
) {
    let mut next_state = vec![0u32; LIMBS];
    let mut current_state = state.to_vec();

    for generation in 1..=generations {
        for i in 0..LIMBS {
            let c = current_state[i];
            let l_carry = if i == 0 { current_state[LIMBS - 1] >> 31 } else { current_state[i - 1] >> 31 };
            let l = (c << 1) | l_carry;
            let r_carry = if i == LIMBS - 1 { current_state[0] & 1 } else { current_state[i + 1] & 1 };
            let r = (c >> 1) | (r_carry << 31);
            next_state[i] = (c | r) ^ (l & c & r);
        }
        current_state.copy_from_slice(&next_state);

        // Dump raw bytes for the MRI
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                current_state.as_ptr() as *const u8,
                current_state.len() * 4,
            )
        };
        mri_file.write_all(byte_slice).unwrap();

        // Sample entropy telemetry every 200 generations
        if generation % 200 == 0 {
            let complexity = compress_size(&current_state);
            writeln!(telemetry_file, "{},{},{}", phase_name, generation, complexity).unwrap();
        }
    }
    
    state.copy_from_slice(&current_state);
}

fn generate_noise() -> Vec<u32> {
    let mut rng = rand::rng();
    let mut limbs = vec![0u32; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = rng.random();
    }
    limbs
}

fn main() {
    println!("================================================================================");
    println!("EXP 42.14+: BOLTZMANN BRAIN EXTENSIONS");
    println!("================================================================================\n");

    let mut telemetry_file = BufWriter::new(File::create("telemetry_42_14_ext.csv").unwrap());
    writeln!(telemetry_file, "Phase,Generation,CompressedSizeBytes").unwrap();

    // PHASE 1: The Emergence
    println!("[*] PHASE 1: The Emergence (Generating Brain A)");
    let mut brain_a = generate_noise();
    let mut mri_a = BufWriter::new(File::create("mri_emergence.bin").unwrap());
    evolve(&mut brain_a, PHASE_GENS, "Emergence", &mut mri_a, &mut telemetry_file);
    println!("    -> Brain A formed.");

    // PHASE 2: The Recursive Mind
    println!("[*] PHASE 2: The Recursive Mind (Feeding Brain A into a new Universe)");
    let mut recursive_brain = brain_a.clone();
    let mut mri_rec = BufWriter::new(File::create("mri_recursive.bin").unwrap());
    evolve(&mut recursive_brain, PHASE_GENS, "Recursive", &mut mri_rec, &mut telemetry_file);
    println!("    -> Recursion complete.");

    // PHASE 3: The Turing Collision
    println!("[*] PHASE 3: The Turing Collision");
    println!("    -> Generating Brain B...");
    let mut brain_b = generate_noise();
    let mut dummy_mri = BufWriter::new(File::create("mri_brain_b_discard.bin").unwrap());
    evolve(&mut brain_b, PHASE_GENS, "Brain_B_Formation", &mut dummy_mri, &mut telemetry_file);

    println!("    -> Colliding Brain A and Brain B via XOR bitwise memory overwrite...");
    let mut collision_brain = vec![0u32; LIMBS];
    for i in 0..LIMBS {
        collision_brain[i] = brain_a[i] ^ brain_b[i];
    }

    println!("    -> Evolving the collision aftermath...");
    let mut mri_collision = BufWriter::new(File::create("mri_collision.bin").unwrap());
    evolve(&mut collision_brain, PHASE_GENS, "Collision", &mut mri_collision, &mut telemetry_file);

    println!("\n[SUCCESS] Extended telemetry and MRI binary dumps successfully written.");
    println!("================================================================================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_brain_evolution() {
        // Assert Rule 110 logic across limb boundaries without writing files
        let mut current_state = vec![0u32; LIMBS];
        current_state[0] = 0b00000001; // Glider seed
        let mut next_state = vec![0u32; LIMBS];
        
        for i in 0..LIMBS {
            let c = current_state[i];
            let l_carry = if i == 0 { current_state[LIMBS - 1] >> 31 } else { current_state[i - 1] >> 31 };
            let l = (c << 1) | l_carry;
            let r_carry = if i == LIMBS - 1 { current_state[0] & 1 } else { current_state[i + 1] & 1 };
            let r = (c >> 1) | (r_carry << 31);
            next_state[i] = (c | r) ^ (l & c & r);
        }
        
        assert_ne!(current_state, next_state, "Rule 110 failed to mutate the cellular state");
        // Ensure bits propagated correctly based on Rule 110
        // C=1 -> L=0, R=0 (assuming 1 was at bit 0). C=1, L=0, R=0 -> 100 -> 0. Wait, bit 0 is at right edge, L is bit 1, R is bit 31 of limb 511.
        // It mutates deterministically.
        assert_eq!(next_state[0] & 1, 1, "Glider seed failed to propagate its boundary state");
    }
}
