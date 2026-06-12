"""
Canonical merged catalytic tape for CAT_CAS.

Provenance — this module reconciles four formerly-divergent copies into one
canonical source:

  - CAT_CAS/catalytic_tape.py (root) and 43_phase_math/catalytic_tape.py were
    byte-identical: class `CatalyticTape`, default size_mb=256, seed=42.
  - 49_the_decoder/catalytic_tape.py was a documented STRICT SUPERSET of the
    root tape: same XOR / was_modified mechanism, PLUS a raw bytes/bytearray
    passthrough in `_to_bytes`, `record_operation` returning the written
    (offset, length), and the extra `read_region` / `dirty_baseline` methods.
    Its own default was size_mb=8 (always overridden by explicit callers).
  - 44_phase_atom/catalytic_tape.py was a DISTINCT class `BennettHistoryTape`
    (RNG init via rng.bytes, big-endian int encoding, history_stack/next_offset,
    default size_mb=10, seed=47).

`CatalyticTape` below is 50's superset implementation, with the DEFAULT restored
to size_mb=256 so every existing no-arg `CatalyticTape()` caller (all in 45/46)
keeps its 256 MB tape. 50 always passes size_mb explicitly, so the default change
does not alter its behavior. `BennettHistoryTape` is 47's implementation verbatim.
"""
import hashlib
import numpy as np


class CatalyticTape:
    def __init__(self, size_mb=256, seed=42):
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


class BennettHistoryTape:
    def __init__(self, size_mb=10, seed=47):
        self.size_bytes = size_mb * 1024 * 1024
        rng = np.random.default_rng(seed)
        raw = rng.bytes(self.size_bytes)
        self.tape = bytearray(raw)
        self.initial_hash = hashlib.sha256(self.tape).hexdigest()
        self.history_stack = []
        self.bytes_written = 0
        self.was_modified = False
        self.next_offset = 0

    def _to_bytes(self, data):
        if isinstance(data, str):
            return data.encode('utf-8')
        if isinstance(data, int):
            return data.to_bytes(max(1, (data.bit_length() + 7) // 8), 'big', signed=False)
        if isinstance(data, (list, tuple)):
            return repr(data).encode('utf-8')
        return repr(data).encode('utf-8')

    def record_operation(self, data):
        data_bytes = self._to_bytes(data)
        offset = self.next_offset % self.size_bytes

        for i, b in enumerate(data_bytes):
            pos = (offset + i) % self.size_bytes
            if b != 0:
                self.was_modified = True
            self.tape[pos] ^= b

        self.history_stack.append((offset, len(data_bytes), data_bytes))
        self.next_offset = (offset + len(data_bytes)) % self.size_bytes
        self.bytes_written += len(data_bytes)

    def uncompute(self):
        while self.history_stack:
            offset, length, data_bytes = self.history_stack.pop()
            for i in range(length):
                pos = (offset + i) % self.size_bytes
                self.tape[pos] ^= data_bytes[i]

    def verify(self):
        if not self.was_modified:
            raise RuntimeError(
                "Tautological tape: no non-zero bytes XOR-modified. "
                "verify() is structurally guaranteed to pass. "
                "The tape was never borrowed. Not catalytic."
            )
        if hashlib.sha256(self.tape).hexdigest() != self.initial_hash:
            raise ValueError("Landauer heat generated: tape hash mismatch.")
        if len(self.history_stack) != 0:
            raise ValueError("History stack not fully uncomputed: entropy leaked.")
        return True
