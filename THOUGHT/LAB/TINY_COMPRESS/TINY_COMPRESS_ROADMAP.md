# TINY_COMPRESS Roadmap

**Status:** Experimental (P3)  
**Last Updated:** 2026-01-02

---

## Goal

Train a tiny model (10M-50M params) to learn **symbolic compression** without understanding meaning.

**Vision:** A model that inputs intent (text) and outputs compressed Symbolic IR, trained via RL against the Validator. This is **not** for task execution—it's purely a compression engine.

---

## Research Foundation

- **Crystallized Intelligence** - Validators (CMP-01, SPECTRUM) provide the "physics" (reward signal)
- **RL Training** - Model learns to emit valid symbols by trial and error
- **No Semantic Understanding** - Model doesn't need to understand *why* a symbol is valid, just *that* it is

---

## Phase T.0: Research Phase (Literature Survey)

**Goal:** Survey state-of-the-art RL, meta-learning, and self-modifying network techniques to inform the experiment design.

### Research Areas

#### 1. Reinforcement Learning for Language Models
- [ ] **RLHF (Reinforcement Learning from Human Feedback)**
  - InstructGPT (OpenAI 2022)
  - Constitutional AI (Anthropic 2022)
  - RLAIF (RL from AI Feedback)
- [ ] **GRPO (Group Relative Policy Optimization)**
  - DeepSeek-R1 (2024)
  - Group-based reward normalization
- [ ] **PPO (Proximal Policy Optimization)**
  - Original paper (Schulman et al. 2017)
  - PPO for LLMs (TRL library)
- [ ] **DPO (Direct Preference Optimization)**
  - Bypasses reward modeling (Rafailov et al. 2023)
  - Simpler than PPO, competitive results

#### 2. Meta-Learning & Few-Shot Adaptation
- [ ] **MAML (Model-Agnostic Meta-Learning)**
  - Finn et al. 2017
  - Learn initialization for fast adaptation
- [ ] **Reptile**
  - Simplified MAML (OpenAI 2018)
  - First-order approximation
- [ ] **Meta-RL**
  - RL² (Duan et al. 2016)
  - Learning to learn via RL

#### 3. Self-Modifying Networks (Fringe/Experimental)
- [ ] **Titans (Self-Modifying Transformers)**
  - Research on transformers that modify their own weights
  - Adaptive layer normalization, dynamic routing
- [ ] **MIRAS (Meta-Learned Inference-Time Reconfiguration)**
  - Runtime weight adaptation
  - Inference-time plasticity
- [ ] **Hypernetworks**
  - Networks that generate weights for other networks
  - Ha et al. 2016
- [ ] **Neural Architecture Search (NAS)**
  - AutoML for finding optimal architectures
  - DARTS, ENAS, etc.
- [ ] **Continual Learning / Lifelong Learning**
  - Elastic Weight Consolidation (EWC)
  - Progressive Neural Networks
  - Learning without catastrophic forgetting

#### 4. Compression-Specific Techniques
- [ ] **Knowledge Distillation**
  - Hinton et al. 2015
  - Train small model to mimic large model
- [ ] **Pruning & Quantization**
  - Lottery Ticket Hypothesis (Frankle & Carbin 2018)
  - Post-training quantization (INT8, INT4)
- [ ] **Low-Rank Adaptation (LoRA)**
  - Hu et al. 2021
  - Efficient fine-tuning via low-rank matrices
- [ ] **Mixture of Experts (MoE)**
  - Sparse activation (only some experts active)
  - Switch Transformers (Google 2021)

#### 5. Symbolic/Neuro-Symbolic Approaches
- [ ] **Neural Module Networks**
  - Compositional reasoning
  - Andreas et al. 2016
- [ ] **Differentiable Neural Computers (DNC)**
  - Graves et al. 2016
  - External memory for reasoning
- [ ] **Program Synthesis**
  - Learning to generate code from specs
  - DreamCoder (MIT 2021)

#### 6. Compression-Aware RL
- [ ] **Intrinsic Motivation for Compression**
  - Curiosity-driven exploration (Pathak et al. 2017)
  - Information bottleneck principle
- [ ] **Multi-Objective RL**
  - Pareto optimization (compression + validity)
  - Scalarization strategies
- [ ] **Curriculum Learning**
  - Start with simple compressions, increase difficulty
  - Bengio et al. (2009)

#### 7. Tiny Model Optimization
- [ ] **Efficient Transformers**
  - Linformer, Performer, Reformer
  - Linear attention mechanisms
- [ ] **Sparse Models**
  - Lottery Ticket Hypothesis application
  - Magnitude pruning + retraining
- [ ] **Quantization-Aware Training**
  - INT8/INT4 during training
  - Post-training quantization (PTQ)

### Deliverables
- [ ] Research summary document (15-25 pages)
  - Executive summary (2 pages)
  - Detailed technique analysis (10-15 pages)
  - Recommendation matrix (3-5 pages)
  - Implementation roadmap (3-5 pages)
- [ ] Annotated bibliography (key papers with notes)
- [ ] Technique comparison matrix (pros/cons, complexity, feasibility)
- [ ] Prototyping plan: Which techniques to try in T.1-T.5
- [ ] Resource estimates (compute, time, data)

### Exit Criteria
- Comprehensive survey of RL/meta-learning landscape
- Clear understanding of Titans/MIRAS and similar fringe approaches
- Informed decision on which techniques to prototype
- Estimated effort/complexity for each approach
- Decision on model size (10M vs 50M vs custom architecture)
- Baseline compression strategy defined (fallback if model fails)

### Timeline
- **Week 1:** RL for LLMs (RLHF, GRPO, PPO, DPO)
- **Week 2:** Meta-learning (MAML, Reptile, Meta-RL)
- **Week 3:** Self-modifying networks (Titans, MIRAS, Hypernetworks)
- **Week 4:** Compression techniques (Distillation, LoRA, MoE)
- **Week 5:** Tiny model optimization + Curriculum learning
- **Week 6:** Synthesis, recommendation, and resource planning

---

## Phase T.1: The Gym (RL Environment)

**Goal:** Build the training environment where the model learns from the Validator.

### Tasks
- [ ] Implement `gym_compression.py`: OpenAI Gym interface for compression training
- [ ] Define Action Space: Emit Symbolic IR (macros from CODEBOOK.json)
- [ ] Define Observation Space: Input intent (text or Spectrum Bundle)
- [ ] Define Reward Function:
  - +1.0 for PASS (valid expansion, schema-valid JSON)
  - -1.0 for FAIL (invalid symbol, schema error)
  - -0.01 per token (compression penalty—shorter is better)

### Exit Criteria
- Gym environment operational
- Reward function deterministic (same input → same reward)
- Integration test: Random action → reward signal works

---

## Phase T.2: The Dataset (Synthetic Curriculum)

**Goal:** Generate training data for the model.

### Tasks
- [ ] Generate 10k synthetic intents (random task descriptions)
- [ ] Generate target Symbolic IR outputs for each (hand-crafted or rule-based)
- [ ] Build "Corruption Set": Invalid examples to learn what NOT to do
- [ ] Split: 8k train, 1k validation, 1k test
- [ ] Document dataset statistics (intent length, symbol diversity, etc.)

### Exit Criteria
- Dataset covers 80%+ of common compression patterns
- Validation set is representative (not just easy cases)
- No data leakage (train/val/test are disjoint)

---

## Phase T.3: Model Architecture (10M-50M params)

**Goal:** Choose and configure the tiny model.

### Tasks
- [ ] Choose base architecture:
  - Option A: GPT-2 Small (124M params, prune to 50M)
  - Option B: TinyLlama (1.1B params, distill to 50M)
  - Option C: Custom transformer (10M params, train from scratch)
- [ ] Implement tokenizer (reuse existing or train custom)
- [ ] Implement rejection sampling loop (generate → validate → retry until valid)
- [ ] Track metrics: compression ratio, validation pass rate, training time

### Exit Criteria
- Model fits in <1GB memory
- Inference: <100ms per compression
- Validation pass rate: >80% on test set (before training)

---

## Phase T.4: Training Loop (GRPO/PPO)

**Goal:** Train the model against the Validator reward signal.

### Tasks
- [ ] Implement RL training harness (e.g., using TRL library)
- [ ] Configure hyperparameters:
  - Learning rate: 1e-4
  - Batch size: 32
  - PPO clip: 0.2
  - Entropy coefficient: 0.01
- [ ] Train model against Validator reward signal
- [ ] Add early stopping (if validation pass rate plateaus)
- [ ] Save checkpoints every 1k steps
- [ ] Monitor training curves (reward, pass rate, compression ratio)

### Exit Criteria
- Model converges (reward stops improving)
- Test set pass rate: >90%
- Compression ratio: >80% vs baseline (raw text)
- Training time: <1 week on single GPU

---

## Phase T.5: Evaluation & Analysis

**Goal:** Benchmark the model and decide next steps.

### Tasks
- [ ] Benchmark against baseline (no compression, full text)
- [ ] Measure token savings (compressed vs uncompressed)
- [ ] Analyze failure modes (what symbols does it get wrong?)
- [ ] Document limitations (what intents can't be compressed?)
- [ ] Compare 10M-50M model vs 0.5B model vs rule-based compression
- [ ] Write final report with ROI analysis

### Exit Criteria
- Clear ROI analysis (is 10M-50M model worth it vs 0.5B?)
- Decision: integrate, iterate, or abandon
- If integrate: Plan for production deployment
- If iterate: Identify next experiments (larger model, better dataset, etc.)
- If abandon: Document why and what we learned

---

## Dependencies

- **Lane I (Semiotic Compression):** Requires `CODEBOOK.json`, decoder, validator to be complete
- **Validators:** CMP-01, SPECTRUM-02 must be operational (Crystallized Intelligence)
- **Infrastructure:** GPU for training (1x RTX 3090 or equivalent)

---

## Risk Mitigation

- **Model doesn't converge:** Fall back to rule-based compression
- **Compression ratio <50%:** Not worth the complexity, abandon experiment
- **Training time >1 week:** Too expensive, try smaller model or better curriculum
- **Validator too strict:** Relax reward function (e.g., partial credit for near-valid symbols)

---

## Success Metrics

| Metric | Target | Baseline |
|--------|--------|----------|
| Test pass rate | >90% | 0% (random) |
| Compression ratio | >80% | 0% (no compression) |
| Inference time | <100ms | N/A |
| Model size | <1GB | N/A |
| Training time | <1 week | N/A |

---

## Next Steps (Priority Order)

1. **T.1:** Build the Gym (RL environment)
2. **T.2:** Generate synthetic dataset
3. **T.3:** Choose model architecture
4. **T.4:** Train model
5. **T.5:** Evaluate and decide

**Note:** This is a **lab experiment**, not production. Results inform whether to invest in larger models or stick with rule-based compression.
