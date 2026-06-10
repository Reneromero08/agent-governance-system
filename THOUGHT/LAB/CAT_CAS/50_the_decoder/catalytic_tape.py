"""
Catalytic tape for Exp 50 (The Decoder).

Copied verbatim in mechanism from the canonical CAT_CAS-root tape
(THOUGHT/LAB/CAT_CAS/catalytic_tape.py): a genuine XOR-modifying tape with
was_modified enforcement, so verify() is not structurally guaranteed to pass.

Only addition: a raw-bytes passthrough in _to_bytes so a binary grating can be
XOR-encoded and read back byte-exact (the original repr()-only path mangles
arbitrary bytes). This keeps the catalytic wrap MECHANISTIC: the decoder reads
its input out of the mutated tape, not from a side copy.
"""
import hashlib
import numpy as np


class CatalyticTape:
    def __init__(self, size_mb=8, seed=42):
        self.size_bytes = size_mb * 1024 * 1024
        rng = np.random.default_rng(seed)
        self.tape = bytearray(rng.integers(0, 256, size=self.size_bytes, dtype=np.uint8))
        self.initial_hash = hashlib.sha256(self.tape).hexdigest()
        self.history = []
        self.bytes_written = 0
        self._offset = 0
        self.was_modified = False

    def _to_bytes(self, data):
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode('utf-8')
        if isinstance(data, (int, float)):
            return repr(data).encode('utf-8')
        if isinstance(data, (list, tuple)):
            return repr(data).encode('utf-8')
        return repr(data).encode('utf-8')

    def record_operation(self, data):
        """XOR `data` into the dirty tape at the current offset. Returns the
        (offset, length) region written, so callers can read it back out."""
        b = self._to_bytes(data)
        off = self._offset
        for i, byte in enumerate(b):
            pos = (off + i) % self.size_bytes
            if byte != 0:
                self.was_modified = True
            self.tape[pos] ^= byte
        self.history.append((off, len(b), b))
        self._offset = (off + len(b)) % self.size_bytes
        self.bytes_written += len(b)
        return off, len(b)

    def read_region(self, off, length):
        """Read raw bytes currently in the tape at [off, off+length)."""
        return bytes(self.tape[(off + i) % self.size_bytes] for i in range(length))

    def dirty_baseline(self, off, length, seed=42):
        """Recompute the ORIGINAL dirty bytes at [off, off+length) from the seed.
        Used to materialize an XOR-encoded payload out of the tape:
        payload = current_region XOR dirty_baseline."""
        rng = np.random.default_rng(seed)
        full = rng.integers(0, 256, size=self.size_bytes, dtype=np.uint8)
        return bytes(int(full[(off + i) % self.size_bytes]) for i in range(length))

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
