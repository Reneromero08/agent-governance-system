# Experiment 06: "Real" Out-Of-Core Catalytic Neural Network

## The Impossible Computation
A classical computer with a strict memory limit cannot execute algorithms that require allocating massive intermediate state structures. A deep Multi-Layer Perceptron (or Convolutional Neural Network) requires storing the layer activations in memory to calculate forward passes. 

If a device has only 100 KB of RAM, but the Neural Network requires a 2 Megabyte activation state vector, the classical computer will immediately crash with an `OutOfMemoryError` because the allocation is physically impossible.

## The Catalytic AI Solution
We built an **Out-of-Core Quantized Feistel ConvNet**. 
Using the principles of Catalytic Space Complexity, the AI inference is executed with an effective clean RAM footprint of less than 32 KB.

**How it works:**
1.  **Dirty Tape:** We supply the system with a pre-existing 2MB file (`user_video.mp4`), simulating random user data or OS libraries existing on a hard drive.
2.  **Zero-Allocation Inference:** Instead of allocating memory for network activations, the matrix dot-products are calculated sequentially using an $O(1)$ scalar accumulator. The resulting activation value is XORed directly into the bytes of the 2MB "dirty" video file using memory mapping.
3.  **Perfect Reversibility:** Because the network is structured as a Reversible Feistel network, computing the layers forwards transforms the 2MB video file into a garbled state (containing the prediction), but running the network backwards un-XORs the activations sequentially, perfectly restoring the video file.

## Results
*   **Clean RAM Enforced:** 100 KB Limit
*   **Classical Result:** CRASH (`MemoryError` trying to allocate 2MB)
*   **Catalytic Result:** SUCCESS (Prediction: Class 2)
*   **Clean RAM Used by Catalytic Run:** ~32 KB
*   **Tape Integrity:** 100% Hash Matched Pre- and Post-Computation

This demonstrates the extreme boundary of Catalytic Space Complexity: running massive Artificial Intelligence models entirely in-place on storage drives by borrowing data, making it theoretically possible to run massive Foundation Models on severely RAM-constrained edge devices.
