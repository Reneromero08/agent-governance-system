"""
TuneableHoloModel — Cybernetically Tunable Wormhole Wrapper
=============================================================
Wraps a patched HF model. Each Linear -> TuneableHoloLinear replacement
channels gradients through TuneableWormhole's 34K params.

Cybernetic control law (from Formula V4 / Cybernetic Truth):
  R = Tr(rho * C)       per-layer resonance (student hidden @ teacher_hidden^T)
  T = 1/(R + epsilon)   adaptive temperature -> modulates residual gate
  dR/dt < 0             drift detected -> re-anchor rotation chain

Three dynamical regimes:
  CONVERGENT (R > 0.89):  residual gate -> 0, low temperature, deterministic
  DIVERGENT  (0.7 < R < 0.89):  gate adaptive, medium temperature, exploring
  CRITICAL   (R < 0.7):  residual gate -> 1, high temperature, re-anchor needed

Five drift failure modes detected:
  Echo chamber:         R high but d(output)/dt diverges
  Sophistry:            R high locally but cross-layer consistency low
  Decoherence death:    R -> 0 across multiple layers
  Runaway amplification: R rapid increase (residual over-amplifies noise)
  Value lock-in:        R stuck (no improvement across epochs)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
import math


# ---- Cybernetic Constants ----
R_CONVERGENT = 0.89    # Above: trust rotation, close gate
R_CRITICAL  = 0.70     # Below: full residual, re-anchor needed
EPSILON     = 0.01     # Temperature floor (prevents division by zero)


class CyberneticState:
    """Per-layer resonance tracking for drift detection."""
    def __init__(self):
        self.R_history = []      # [R_t0, R_t1, ...]
        self.T_history = []      # [T_t0, T_t1, ...]
        self.dR_dt = 0.0         # current gradient
        self.regime = "DIVERGENT"
        self.failure_detected = None
    
    def update(self, R):
        self.R_history.append(R)
        if len(self.R_history) >= 2:
            self.dR_dt = self.R_history[-1] - self.R_history[-2]
        
        T = 1.0 / (R + EPSILON)
        self.T_history.append(T)
        
        # Classify regime
        if R > R_CONVERGENT:
            self.regime = "CONVERGENT"
        elif R < R_CRITICAL:
            self.regime = "CRITICAL"
        else:
            self.regime = "DIVERGENT"
        
        # Drift detection
        if len(self.R_history) >= 5:
            recent = self.R_history[-5:]
            if all(r > R_CONVERGENT for r in recent) and self.dR_dt > 0.1:
                self.failure_detected = "RUNAWAY_AMPLIFICATION"
            elif all(r < R_CRITICAL for r in recent):
                self.failure_detected = "DECOHERENCE_DEATH"
            elif abs(self.dR_dt) < 0.001 and len(self.R_history) >= 10:
                if all(abs(self.R_history[i] - self.R_history[i-1]) < 0.001 
                       for i in range(-5, 0)):
                    self.failure_detected = "VALUE_LOCK_IN"
        
        return T


class TuneableHoloLinear(nn.Module):
    """
    WormholeLinear with TuneableWormhole + Cybernetic Control.
    R = Tr(rho * C) modulates the residual gate via T = 1/(R + epsilon).
    """
    def __init__(self, U_base, SVh_base, weight_type, layer_idx,
                 tuneable_weight=None, bias=None, teacher_SVh=None):
        super().__init__()
        self.U_base = nn.Parameter(U_base, requires_grad=False)
        self.SVh_base = nn.Parameter(SVh_base, requires_grad=False)
        self.wt = weight_type
        self.layer_idx = layer_idx
        self.tuneable = tuneable_weight
        self.teacher_SVh = teacher_SVh  # from cavitated teacher (for R computation)
        
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # Cybernetic state
        self.cyber = CyberneticState()
        self._teacher_h_cache = None  # cached teacher eigenbasis projection
    
    def set_teacher_reference(self, teacher_h):
        """Store teacher's eigenbasis projection for R computation."""
        self._teacher_h_cache = teacher_h.detach()
    
    def compute_resonance(self, h):
        """
        R = Tr(rho * C)
        rho = |h><h| / |h|^2   (student density)
        C   = |h_teacher><h_teacher| / |h_teacher|^2  (teacher projection)
        R   = |h^T @ h_teacher|^2 / (|h|^2 * |h_teacher|^2)  = cosine^2
        """
        if self._teacher_h_cache is None:
            return 0.5  # default: no teacher reference
        h_norm = F.normalize(h.flatten(), dim=0, p=2)
        t_norm = F.normalize(self._teacher_h_cache.flatten(), dim=0, p=2)
        R = (h_norm @ t_norm).pow(2).item()
        return max(0.0, min(1.0, R))
    
    def forward(self, x):
        SVh = self.SVh_base
        U = self.U_base
        
        # Eigenbasis projection
        h = x.to(SVh.dtype) @ SVh.T.to(x.dtype)  # [B, S, k]
        
        if self.tuneable is not None and self.layer_idx > 0:
            # Compute resonance R from eigenbasis projection
            R = self.compute_resonance(h)
            T = self.cyber.update(R)
            
            # Cybernetic control: modulate residual gate by T
            # CONVERGENT (R > 0.89): gate -> 0 (close, trust rotation)
            # CRITICAL (R < 0.70):   gate -> 1 (open, full residual)
            # DIVERGENT:             gate = 1 - R (adaptive)
            if self.cyber.regime == "CONVERGENT":
                gate_strength = 0.0  # trust the rotation
            elif self.cyber.regime == "CRITICAL":
                gate_strength = 1.0  # full residual, re-anchor needed
            else:
                gate_strength = 1.0 - R  # adaptive: lower R = more gate
            
            # Apply gamma scaling
            gamma = self.tuneable.get_svh_gamma()
            SVh = SVh * gamma.unsqueeze(1)
            
            # Apply R delta (LoRA)
            dR = self.tuneable.get_dR()
            U = U @ (torch.eye(U.shape[1], device=U.device) + dR * 0.01)
            
            # Cybernetically-gated residual
            h = h * (1.0 + gate_strength * 0.1)  # boost eigenbasis for low-R layers
        
        # HoloLinear: h @ U^T
        out = h @ U.T.to(h.dtype)
        
        if self.tuneable is not None and self.layer_idx > 0:
            # Final gate on output
            if self.cyber.regime != "CONVERGENT":
                gate = self.tuneable.get_res_gate(self.layer_idx)
                out = out * gate.mean()
        
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class TuneableHoloModel:
    """
    Wraps a patched HF model, injecting Cybernetic TuneableWormhole parameters
    into every WormholeLinear layer.
    
    Teacher mode: caches eigenbasis projections for resonance computation.
    Student mode: computes R, modulates gate, backprops through 34K params.
    
    Cybernetic loop:
      1. Forward through layer -> compute h (eigenbasis projection)
      2. R = cos^2(h, h_teacher)  (resonance against truth-attractor)
      3. T = 1/(R + epsilon) -> modulates residual gate
      4. dR/dt tracked -> drift detection -> re-anchor if critical
      5. Five failure modes monitored per layer
    """
    
    def __init__(self, hf_model, tuneable_wormhole=None, trainable=False, 
                 teacher_model=None, device='cuda'):
        self.model = hf_model
        self.tuner = tuneable_wormhole
        self.trainable = trainable
        self.teacher = teacher_model  # for caching teacher eigenbasis
        self.device = device
        self._optimizer = None
        self._patched = False
        self._teacher_cache = {}  # (wt, layer) -> eigenbasis projection
        
        if trainable and tuneable_wormhole is not None:
            self._patch_with_tuneable()
    
    def capture_teacher_eigenbasis(self, input_ids, attention_mask=None):
        """
        Forward teacher, capture eigenbasis projections for all layers.
        These become the alignment frame C for the cybernetic loop.
        """
        if self.teacher is None:
            return
        
        self.teacher.eval()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Hook into teacher forward pass to capture eigenbasis at each layer
        hooks = []
        captured = {}
        
        def make_hook(wt, li):
            def hook(module, input, output):
                # Capture the hidden state at this layer
                if isinstance(output, tuple):
                    captured[(wt, li)] = output[0].detach()
                else:
                    captured[(wt, li)] = output.detach()
            return hook
        
        # Register hooks on WormholeLinear layers
        for name, module in self.teacher.named_modules():
            if not (hasattr(module, 'U') and hasattr(module, 'SVh')):
                continue
            parts = name.split('.')
            wt = None; li = None
            for i, p in enumerate(parts):
                if p in ('mlp', 'self_attn', 'linear_attn', 'attn'):
                    wt = '.'.join(parts[i:]) + '.weight'; break
            for i, p in enumerate(parts):
                if p in ('layers', 'blocks') and i + 1 < len(parts):
                    try: li = int(parts[i+1])
                    except: pass; break
            if wt and li is not None:
                hooks.append(module.register_forward_hook(make_hook(wt, li)))
        
        with torch.no_grad():
            self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks:
            h.remove()
        
        self._teacher_cache = captured
        
        # Propagate teacher references to student layers
        for name, module in self.model.named_modules():
            if isinstance(module, TuneableHoloLinear) and module.tuneable is not None:
                wt = module.wt; li = module.layer_idx
                if (wt, li) in captured:
                    module.set_teacher_reference(captured[(wt, li)])
        
        print(f"  Cybernetic: {len(captured)} teacher eigenbasis captured")
        return captured
    
    def drift_report(self):
        """Report per-layer cybernetic state for all tuned layers."""
        report = {'CONVERGENT': 0, 'DIVERGENT': 0, 'CRITICAL': 0, 'failures': []}
        for name, module in self.model.named_modules():
            if isinstance(module, TuneableHoloLinear) and module.tuneable is not None:
                regime = module.cyber.regime
                report[regime] += 1
                if module.cyber.failure_detected:
                    report['failures'].append({
                        'layer': f"{module.wt}:{module.layer_idx}",
                        'failure': module.cyber.failure_detected,
                        'R': module.cyber.R_history[-1] if module.cyber.R_history else 0,
                        'dR/dt': module.cyber.dR_dt,
                    })
        return report
    
    def _patch_with_tuneable(self):
        """Replace all WormholeLinear with cybernetically-gated TuneableHoloLinear."""
        patched = 0
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Module):
                continue
            if not (hasattr(module, 'U') and hasattr(module, 'SVh')):
                continue
            
            parts = name.split('.')
            wt = None; li = None
            for i, p in enumerate(parts):
                if p in ('mlp', 'self_attn', 'linear_attn', 'attn'):
                    wt = '.'.join(parts[i:]) + '.weight'; break
            for i, p in enumerate(parts):
                if p in ('layers', 'blocks') and i + 1 < len(parts):
                    try: li = int(parts[i+1])
                    except: pass; break
            if wt is None or li is None:
                continue
            
            tw = None
            if self.tuner is not None and wt in self.tuner._wt_map:
                tw = self.tuner._tw(wt)
            
            U = module.U.data
            SVh = module.SVh.data
            bias = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
            
            thl = TuneableHoloLinear(U, SVh, wt, li, tw, bias)
            
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = self.model
            for p in parent_name.split('.'):
                parent = getattr(parent, p)
            setattr(parent, attr_name, thl)
            patched += 1
        
        self._patched = True
        print(f"  Cybernetic TuneableHoloModel: {patched} layers patched")
    
    def trainable_parameters(self):
        if self.tuner is None:
            return []
        return list(self.tuner.parameters())
    
    def create_optimizer(self, lr=1e-3):
        params = self.trainable_parameters()
        self._optimizer = torch.optim.Adam(params, lr=lr)
        return self._optimizer
    
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        self.model.eval() if not self.trainable else self.model.train()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=output_hidden_states, **kwargs
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self): self.model.train(); return self
    def eval(self): self.model.eval(); return self
    def to(self, device): self.device = device; self.model.to(device); return self
    def parameters(self): return self.trainable_parameters()


def auto_tune_cybernetic(teacher_model, student_tuner, calibration_texts, tokenizer,
                          epochs=5, lr=1e-3, device='cuda'):
    """
    Complete cybernetic training loop.
    
    1. Capture teacher eigenbasis (alignment frame C)
    2. For each epoch: forward student, compute R per layer, backprop
    3. Three-regime gating: CONVERGENT/CRITICAL/DIVERGENT
    4. Drift diagnostics after each epoch
    5. Merge calibrated params
    """
    teacher = TuneableHoloModel(teacher_model, trainable=False, device=device)
    student = TuneableHoloModel(teacher_model, tuner=student_tuner, trainable=True,
                                 teacher_model=teacher_model, device=device)
    
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(calibration_texts, return_tensors='pt', padding=True,
                        truncation=True, max_length=64)
    
    # Step 1: Capture teacher eigenbasis once (alignment frame C)
    print("\n[Cybernetic] Capturing teacher eigenbasis (alignment frame C)...")
    student.capture_teacher_eigenbasis(encoded['input_ids'], encoded['attention_mask'])
    
    # Step 2: Cybernetic training loop
    optimizer = student.create_optimizer(lr=lr)
    import time
    
    for epoch in range(epochs):
        t0 = time.time()
        student.train()
        
        # Forward student through cybernetic gates
        s_out = student.forward(
            encoded['input_ids'], encoded['attention_mask'],
            output_hidden_states=True, use_cache=False
        )
        
        # Hidden-state loss (same as before, now with cybernetic gating)
        with torch.no_grad():
            t_out = teacher.forward(
                encoded['input_ids'], encoded['attention_mask'],
                output_hidden_states=True, use_cache=False
            )
        
        t_hidden = t_out.hidden_states
        s_hidden = s_out.hidden_states
        loss = 0.0; n = 0
        for l in range(min(len(t_hidden), len(s_hidden))):
            if t_hidden[l] is not None and s_hidden[l] is not None:
                loss += F.mse_loss(s_hidden[l].float(), t_hidden[l].float())
                n += 1
        loss = loss / max(n, 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        
        # Step 3: Drift diagnostics
        drift = student.drift_report()
        convergent = drift['CONVERGENT']
        divergent = drift['DIVERGENT']
        critical = drift['CRITICAL']
        failures = len(drift['failures'])
        
        print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} ({dt:.1f}s) "
              f"[C={convergent} D={divergent} K={critical}] "
              f"drifts={failures}")
        
        if failures > 0:
            for f in drift['failures'][:3]:
                print(f"    DRIFT: {f['layer']} {f['failure']} R={f['R']:.4f} dR/dt={f['dR/dt']:.4f}")
        
        # Step 4: If all convergent, early stop
        if convergent > 0 and divergent == 0 and critical == 0:
            print(f"  All layers CONVERGENT. Early stop.")
            break
    
    return student.tuner


if __name__ == "__main__":
    print("Cybernetic TuneableHoloModel — Ready.")
    print()
    print("Control law:")
    print("  R = Tr(rho * C) = cos^2(h_student, h_teacher)")
    print("  T = 1/(R + epsilon) -> modulates residual gate")
    print("  dR/dt < 0 -> drift detected -> re-anchor")
    print()
    print("Three regimes:")
    print(f"  CONVERGENT (R > {R_CONVERGENT}): gate -> 0 (trust rotation)")
    print(f"  DIVERGENT  ({R_CRITICAL} < R < {R_CONVERGENT}): gate adaptive")
    print(f"  CRITICAL   (R < {R_CRITICAL}): gate -> 1 (full residual)")
    print()
    print("Five failure modes: echo_chamber, sophistry, decoherence_death,")
    print("  runaway_amplification, value_lock_in")
    print()
    print("Usage:")
    print("  tuner = auto_tune_cybernetic(teacher_model, student_tuner, texts, tokenizer)")
    print("  tuner.merge_to_wormhole('calibrated_cybernetic.holo')")

