import hashlib
import numpy as np

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
