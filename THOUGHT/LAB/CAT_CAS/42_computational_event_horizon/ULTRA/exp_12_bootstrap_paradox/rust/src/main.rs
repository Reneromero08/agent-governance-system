use num_bigint::BigUint;
use std::fs::File;
use std::io::Write;
use std::mem;
use winapi::um::memoryapi::VirtualProtect;
use winapi::um::winnt::PAGE_EXECUTE_READWRITE;

fn main() {
    println!("[EVENT HORIZON] Initializing Exp 42.12: The Bootstrap Paradox");

    // 1. Create the Event Horizon
    // We allocate a large BigUint (equivalent to a singularity with massive precision).
    // We use `1` instead of `0` because BigUint strips leading zeros and won't allocate heap memory for zero!
    let mut singularity = BigUint::from_slice(&[1; 128]);

    // 2. The Payload (x86_64 Shellcode)
    // mov eax, 0x42 (Return 66)
    // ret
    let shellcode: [u8; 6] = [0xB8, 0x42, 0x00, 0x00, 0x00, 0xC3];

    // 3. The Injection (Crossing the Event Horizon into unsafe memory)
    unsafe {
        // We bypass Rust's safe abstractions.
        // A BigUint is exactly the same layout as a Vec<u32> because it's a newtype struct.
        // We forcibly transmute the memory to access the raw heap pointer of the mathematical object.
        let raw_vec_ptr: *mut Vec<u32> = mem::transmute(&mut singularity);
        
        // Extract the raw heap pointer of the limbs
        let mantissa_ptr = (*raw_vec_ptr).as_mut_ptr() as *mut u8;

        println!("[INJECTION] Locating mantissa at memory address: {:?}", mantissa_ptr);
        
        // Write the shellcode directly into the float's mathematical mantissa
        std::ptr::copy_nonoverlapping(shellcode.as_ptr(), mantissa_ptr, shellcode.len());
        println!("[INJECTION] Shellcode embedded into math object.");

        // 4. Bypassing OS Data Execution Prevention (DEP)
        // By default, heap memory is NOT executable (to prevent malware). 
        // We must invoke the Windows kernel to change the page permissions.
        let mut old_protect: u32 = 0;
        
        // We protect a 4096 byte page, which is the standard Windows memory page size.
        let success = VirtualProtect(
            mantissa_ptr as *mut _,
            4096,
            PAGE_EXECUTE_READWRITE,
            &mut old_protect,
        );

        if success == 0 {
            panic!("[ERROR] OS Kernel blocked execution permission shift.");
        }
        println!("[KERNEL] CPU Data Execution Prevention (DEP) bypassed for Event Horizon.");

        // 5. The Causal Loop
        // Cast the math mantissa pointer directly to a Rust function pointer.
        let execution_pointer: extern "C" fn() -> u32 = mem::transmute(mantissa_ptr);

        println!("[CRITICAL] Jumping CPU Instruction Pointer into the math object...");
        
        // Execute the math object. The CPU is now running inside the BigUint!
        let anomaly_result = execution_pointer();

        println!("[SUCCESS] CPU emerged from Event Horizon. Result: 0x{:X}", anomaly_result);

        // 6. Out-of-Band Telemetry
        // Write to a raw binary file immediately. When `singularity` drops,
        // it might segfault due to modified page protections.
        let mut file = File::create("telemetry_42_12.bin").expect("Failed to open telemetry file.");
        file.write_all(&anomaly_result.to_le_bytes()).expect("Failed to write anomaly result.");
        
        println!("[TELEMETRY] Anomaly result securely written to disk: telemetry_42_12.bin");
    }

    println!("[SYSTEM] Paradox stable. Collapsing Event Horizon...");
    
    // Prevent the OS from segfaulting when dropping the executable memory page
    std::mem::forget(singularity);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bootstrap_paradox_execution() {
        let mut singularity = BigUint::from_slice(&[1; 128]);
        let shellcode: [u8; 6] = [0xB8, 0x42, 0x00, 0x00, 0x00, 0xC3];
        
        let anomaly_result = unsafe {
            let raw_vec_ptr: *mut Vec<u32> = mem::transmute(&mut singularity);
            let mantissa_ptr = (*raw_vec_ptr).as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(shellcode.as_ptr(), mantissa_ptr, shellcode.len());
            
            let mut old_protect: u32 = 0;
            VirtualProtect(mantissa_ptr as *mut _, 4096, PAGE_EXECUTE_READWRITE, &mut old_protect);
            
            let execution_pointer: extern "C" fn() -> u32 = mem::transmute(mantissa_ptr);
            execution_pointer()
        };
        
        assert_eq!(anomaly_result, 0x42, "The CPU failed to execute the shellcode inside the math object!");
        
        // Prevent the allocator from segfaulting when trying to free the executable page
        std::mem::forget(singularity);
    }
}

