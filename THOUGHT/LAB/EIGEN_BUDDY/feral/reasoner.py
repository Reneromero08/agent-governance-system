"""Native Eigen Reasoner — drop-in GeometricReasoner for Feral Resident.

Points (initialize/readout): MiniLM directly — same as GeometricReasoner.
Geodesics (navigate/entangle/interpolate/superpose/project): NativeEigenCore.
The Core only activates for multi-vector sequences.
"""
import sys, time, numpy as np, torch
from pathlib import Path
from typing import List, Dict, Tuple

EIGEN_PATH = Path(__file__).parent.parent
FERAL_PATH = Path(r'THOUGHT/LAB/FERAL_RESIDENT')
for p in [str(FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"),
          str(FERAL_PATH), str(EIGEN_PATH)]:
    if p not in sys.path: sys.path.insert(0, p)

from geometric_reasoner import GeometricState
from core import NativeEigenCore as _NativeEigenCore


class NativeEigenReasoner:
    """Phase-native reasoner. Core only activates for multi-vector geodesics."""

    def __init__(self, model_name='all-MiniLM-L6-v2',
                 d=64, heads=4, layers=4, cycles=4, weights_path=None):
        from sentence_transformers import SentenceTransformer
        self.embed = SentenceTransformer(model_name)
        self.d = d; self.heads = heads; self.layers = layers; self.cycles = cycles
        self.D_emb = 384; self.D = self.D_emb // 2
        self._core = None
        self._weights_path = weights_path or str(
            Path(__file__).parent.parent / "weights" / "feral.pt")
        self.op_count = {'initialize': 0, 'readout': 0, 'navigate': 0,
                        'entangle': 0, 'superpose': 0, 'project': 0, 'interpolate': 0}

    def _init_core(self):
        if self._core is not None: return
        import torch.nn as nn, os
        self._core = _NativeEigenCore(d=self.d, heads=self.heads,
                        layers=self.layers, merge='concat', geo_init=True)
        self._in_r = nn.Linear(self.D, self.d, bias=False)
        self._in_i = nn.Linear(self.D, self.d, bias=False)
        self._out_r = nn.Linear(self.d, self.D, bias=False)
        self._out_i = nn.Linear(self.d, self.D, bias=False)
        for w in [self._in_r, self._in_i, self._out_r, self._out_i]:
            nn.init.normal_(w.weight, std=0.02)

        if os.path.exists(self._weights_path):
            ckpt = torch.load(self._weights_path, map_location='cpu')
            cfg = ckpt.get('config', {})
            if cfg.get('d') == self.d:
                self._core.load_state_dict(ckpt['core'])
                self._in_r.load_state_dict(ckpt['in_r'])
                self._in_i.load_state_dict(ckpt['in_i'])
                self._out_r.load_state_dict(ckpt['out_r'])
                self._out_i.load_state_dict(ckpt['out_i'])
        self._core.eval()

    def _core_process(self, seq):
        """Process a sequence of vectors through the Core, return (result, phase_coh)."""
        z = torch.complex(torch.tensor(seq[:, :self.D], dtype=torch.float32).unsqueeze(0),
                          torch.tensor(seq[:, self.D:], dtype=torch.float32).unsqueeze(0))
        zp = torch.complex(self._in_r(z.real) - self._in_i(z.imag),
                           self._in_r(z.imag) + self._in_i(z.real))
        with torch.no_grad():
            z_out, pc = self._core(zp + zp)
        pr = self._out_r(z_out.real) + self._out_i(z_out.imag)
        pi = self._out_r(z_out.imag) - self._out_i(z_out.real)
        return torch.cat([pr, pi], dim=-1).detach(), float(pc)

    # ---- Points: MiniLM only ----
    def initialize(self, text: str) -> GeometricState:
        self.op_count['initialize'] += 1
        return GeometricState(vector=self.embed.encode(text))

    def readout(self, state: GeometricState, corpus: List[str], k=5) -> List[Tuple[str, float]]:
        self.op_count['readout'] += 1
        if not corpus: return []
        cv = self.embed.encode(corpus)
        scores = [(corpus[i], float(np.dot(state.vector, cv[i]) /
                   (np.linalg.norm(state.vector) * np.linalg.norm(cv[i]) + 1e-8)))
                  for i in range(len(corpus))]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    # ---- Geodesics: Core processes sequences ----
    def navigate(self, start: str, end: str, steps: int,
                 corpus: List[str], k=3) -> List[Dict]:
        self.op_count['navigate'] += 1
        self._init_core()
        sv = self.embed.encode(start); ev = self.embed.encode(end)
        path = [{'step': 0, 'text': start}]
        current = sv.copy()
        for step in range(1, steps + 1):
            t = step / (steps + 1)
            interp = (1 - t) * current + t * ev
            seq = np.stack([current, interp])
            result, pc = self._core_process(seq)
            current = result[0, -1, :].numpy()
            nearest = self.readout(GeometricState(vector=current), corpus, k=1)
            path.append({'step': step, 'text': nearest[0][0] if nearest else '...',
                        'phase_coh': pc})
        return path

    def entangle(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        self._init_core()
        seq = np.stack([state1.vector, state2.vector])
        result, _ = self._core_process(seq)
        return GeometricState(vector=result[0].mean(dim=0).numpy())

    def interpolate(self, state1: GeometricState, state2: GeometricState,
                    t: float) -> GeometricState:
        self._init_core()
        interp = (1 - t) * state1.vector + t * state2.vector
        seq = np.stack([state1.vector, interp, state2.vector])
        result, _ = self._core_process(seq)
        return GeometricState(vector=result[0, 1, :].numpy())

    def superpose(self, state1: GeometricState, state2: GeometricState) -> GeometricState:
        """Phase-rich superposition via Core — used by SemanticDiffusion.navigate()."""
        self.op_count['superpose'] += 1
        self._init_core()
        seq = np.stack([state1.vector, state2.vector])
        result, _ = self._core_process(seq)
        return GeometricState(vector=result[0].mean(dim=0).numpy())

    def project(self, state: GeometricState, context: List[GeometricState]) -> GeometricState:
        """Born rule projection via Core — used by SemanticDiffusion.navigate()."""
        self.op_count['project'] += 1
        self._init_core()
        if not context: return state
        vecs = [state.vector] + [c.vector for c in context[:7]]
        seq = np.stack(vecs)
        result, _ = self._core_process(seq)
        return GeometricState(vector=result[0].mean(dim=0).numpy())

    def E_with(self, state1: GeometricState, state2: GeometricState) -> float:
        """Born rule resonance via Core geodesics — replaces cosine similarity.

        Daemon collision: processes both states through the Core as a 2-vector
        sequence. Phase coherence output IS the E measurement.
        The 0.16 threshold gates on geodesic alignment, not surface similarity.
        """
        self.op_count.setdefault('E_with', 0)
        self.op_count['E_with'] += 1
        self._init_core()
        seq = np.stack([state1.vector, state2.vector])
        _, pc = self._core_process(seq)
        return pc  # phase coherence = resonance

    def get_stats(self) -> Dict:
        return {'model': 'NativeEigenReasoner', 'core_d': self.d,
                'core_heads': self.heads, 'core_layers': self.layers,
                'cycles': self.cycles, 'operations': self.op_count}
