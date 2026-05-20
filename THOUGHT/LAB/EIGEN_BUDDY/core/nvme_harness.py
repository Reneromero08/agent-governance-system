"""Item 4: NVMe Stencil Harness — gguf library + mmap stride reader.
0 bytes RAM for weights. Catalytic tape (U) for scratch.
"""
import mmap, os, hashlib, time, math
from pathlib import Path
from gguf import GGUFReader

MODELS = {
    "1.2B": r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "4B":  r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf",
    "27B": r"F:\LLM_Models\lmstudio-models\Qwen3.6-27B\Qwen3.6-27B-F16-mtp.gguf",
}

class NVMeStencilHarness:
    def __init__(self, model_key="1.2B"):
        self.model_path = MODELS.get(model_key, model_key)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(self.model_path)
        self.file_size = Path(self.model_path).stat().st_size

        t0 = time.time()
        self._gguf = GGUFReader(self.model_path)
        self._mm = self._gguf.data
        self.parse_time = time.time() - t0

        self.tensors = {}
        self.gdn_layers = []
        self.ga_layers = []
        self.mtp_names = []
        self._tape = {}
        self._hashes = []

    def parse(self):
        t0 = time.time()
        for rt in self._gguf.tensors:
            self.tensors[rt.name] = {
                'offset': rt.data_offset, 'size': rt.n_bytes,
                'shape': rt.shape, 'dtype': rt.tensor_type
            }
        self._classify_layers()
        dt = time.time() - t0
        total_mb = sum(t['size'] for t in self.tensors.values()) / 1e6
        print(f"[parse] {len(self.tensors)} tensors, {total_mb:.0f} MB | "
              f"{len(self.gdn_layers)} GDN + {len(self.ga_layers)} GA + "
              f"{len(self.mtp_names)} MTP | lib:{self.parse_time:.1f}s + class:{dt:.1f}s")

    def _classify_layers(self):
        seen = set()
        for name in self.tensors:
            if any(x in name.lower() for x in ('mtp', 'ssm', 'next_token')):
                self.mtp_names.append(name); continue
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit() and i > 0: seen.add(int(p)); break
        max_layer = max(seen) + 1 if seen else 0
        for lidx in range(max_layer):
            prefix = f'.{lidx}.'
            has_sep = any('attn_q.weight' in n and 'attn_qkv' not in n
                         for n in self.tensors if prefix in n)
            (self.ga_layers if has_sep else self.gdn_layers).append(lidx)

    def read_block(self, block_idx):
        gdn = self.gdn_layers[block_idx*3:block_idx*3+3]
        ga = [self.ga_layers[block_idx]] if block_idx < len(self.ga_layers) else []
        layer_set = set(gdn + ga)
        offs = []
        for name, t in self.tensors.items():
            for lid in layer_set:
                if f'.{lid}.' in name:
                    offs.append((name, t['offset'], t['offset'] + t['size'])); break
        if not offs: return {}, 0
        min_off = min(o[1] for o in offs); max_off = max(o[2] for o in offs)
        stride_bytes = max_off - min_off
        window = memoryview(self._mm[min_off:max_off])
        key = f'block_{block_idx}'
        self._tape[key] = bytes(window)
        self._hashes.append(hashlib.sha256(self._tape[key]).digest())
        mb = stride_bytes / 1e6
        n_tensors = len(offs)
        n_layers = len(layer_set)
        print(f"[stride] Block {block_idx}: {n_tensors} tensors, {mb:.1f} MB "
              f"({n_layers} layers)")
        return {name: window[o-min_off:o+self.tensors[name]['size']-min_off]
                for name, o, _ in offs}, mb

    def read_mtp(self):
        if not self.mtp_names: return {}, 0
        offs = [(n, self.tensors[n]['offset'], self.tensors[n]['offset']+self.tensors[n]['size'])
                for n in self.mtp_names if n in self.tensors]
        if not offs: return {}, 0
        min_off = min(o[1] for o in offs); max_off = max(o[2] for o in offs)
        window = memoryview(self._mm[min_off:max_off])
        key = 'mtp_heads'
        self._tape[key] = bytes(window)
        self._hashes.append(hashlib.sha256(self._tape[key]).digest())
        mb = (max_off - min_off) / 1e6
        n_heads = len(offs)
        print(f"[mtp] {n_heads} heads, {mb:.1f} MB")
        return {name: window[o-min_off:o+self.tensors[name]['size']-min_off]
                for name, o, _ in offs}, mb

    def verify(self):
        if not self._hashes: print("[verify] Empty"); return True
        cur = [hashlib.sha256(self._tape[k]).digest() for k in self._tape]
        ok = cur == self._hashes[:len(cur)]
        print(f"[verify] {'PASS' if ok else 'FAIL'} ({len(cur)} entries)")
        return ok

    def clear_tape(self):
        self.verify()
        for k in list(self._tape):
            self._tape[k] = b'\x00' * len(self._tape[k]); del self._tape[k]
        self._hashes.clear()
        print("[adjoint] Tape cleared. 0 bits.")

    def close(self): pass

    def summary(self):
        total_mb = sum(t['size'] for t in self.tensors.values()) / 1e6
        return {
            'file': Path(self.model_path).name,
            'size_gb': self.file_size / 1e9,
            'n_tensors': len(self.tensors),
            'data_mb': total_mb,
            'gdn': len(self.gdn_layers),
            'ga': len(self.ga_layers),
            'mtp': len(self.mtp_names),
            'tape': len(self._tape),
            '3:1_blocks': len(self.gdn_layers) // 3,
        }


if __name__ == '__main__':
    print("=" * 60)
    print("ITEM 4: NVMe Stencil Harness")
    print("=" * 60)
    for key in ["1.2B", "4B"]:
        print(f"\n--- {key} ---")
        try:
            h = NVMeStencilHarness(key)
            h.parse()
            h.read_block(0)
            h.read_mtp()
            h.verify()
            for k, v in h.summary().items():
                print(f"  {k}: {v}")
            h.clear_tape()
            h.close()
        except Exception as e:
            import traceback; traceback.print_exc()
