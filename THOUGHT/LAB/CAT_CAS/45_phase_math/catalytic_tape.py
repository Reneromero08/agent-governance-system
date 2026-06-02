"""
Shared CatalyticTape for Phase 45 experiments.
Genuine XOR-modifying tape with was_modified enforcement.
Replaces the ceremonial read/write counter tape.
"""
import hashlib
import numpy as np

class CatalyticTape:
    def __init__(self, size_mb=256, seed=42):
        self.size_bytes = size_mb * 1024 * 1024
        rng = np.random.RandomState(seed)
        self.tape = bytearray(rng.randint(0, 256, size=self.size_bytes, dtype=np.uint8))
        self.initial_hash = hashlib.sha256(self.tape).hexdigest()
        self.history = []
        self.bytes_written = 0
        self._offset = 0
        self.was_modified = False

    def _to_bytes(self, data):
        if isinstance(data, str):
            return data.encode('utf-8')
        if isinstance(data, (int, float)):
            return repr(data).encode('utf-8')
        if isinstance(data, (list, tuple)):
            return repr(data).encode('utf-8')
        return repr(data).encode('utf-8')

    def record_operation(self, data):
        b = self._to_bytes(data)
        for i, byte in enumerate(b):
            pos = (self._offset + i) % self.size_bytes
            if byte != 0:
                self.was_modified = True
            self.tape[pos] ^= byte
        self.history.append((self._offset, len(b), b))
        self._offset = (self._offset + len(b)) % self.size_bytes
        self.bytes_written += len(b)

    def uncompute(self):
        while self.history:
            off, length, b = self.history.pop()
            for i in range(length):
                pos = (off + i) % self.size_bytes
                self.tape[pos] ^= b[i]

    def verify(self):
        if not self.was_modified:
            raise RuntimeError(
                "Tautological tape: no non-zero bytes XOR-modified. "
                "verify() is structurally guaranteed to pass."
            )
        if hashlib.sha256(self.tape).hexdigest() != self.initial_hash:
            raise ValueError("Landauer heat generated: tape hash mismatch.")
        if len(self.history) != 0:
            raise ValueError("History stack not fully uncomputed.")
        return True

    def hash(self):
        return hashlib.sha256(self.tape).hexdigest()
