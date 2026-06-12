# Experiment 3: Visual BMP Image Catalytic Memory

This experiment demonstrates a DFS maze solver on a 40x40 grid (1,600 nodes) running under a strict **64-byte clean memory limit** by using the raw pixel bytes of a 512x512 BMP image file as its stack and visited bit-vector.

## Deterministic Zero-Storage Retrieval
Because we cannot store the original pixel bytes in clean RAM (as doing so would violate the 64-byte limit), the tape utilizes a **deterministic gradient formula** to recompute the original pixel bytes on the fly. 

To read a virtual stack value or visited bit:
$$\text{Stored Value} = \text{Current Byte} \oplus \text{Recomputed Original Byte}$$
To restore the tape, we simply overwrite the pixel with the recomputed original byte.

## Results
*   **Pristine Hash:** `701f9b72b65d2e9a14abbc71bbe106396a22e9215c47e6856be57fff8467cd41`
*   **Dirty Hash (During DFS):** `ee0698f44b81e82d6d9c01ce0c0cfd37f561448dd74e2755c72f0714828f9a08`
*   **Final Hash (After Traversal):** `701f9b72b65d2e9a14abbc71bbe106396a22e9215c47e6856be57fff8467cd41` (100% Match)
*   **Clean RAM Footprint:** **10 bytes** (Used to track `current`, `target`, and `sp`), well under the 64-byte constraint.
