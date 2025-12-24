# Catalytic Computing: Master Implementation Plan

**Status**: Blueprint
**Goal**: Break the context window barrier
**Outcome**: Agents reason over unlimited data with mathematical restoration proofs

---

## The Core Insight

**Context window is NOT the workspace. It's the cache.**

| Concept | Formal Theory | AGS Implementation |
|---------|---------------|-------------------|
| Clean space | Small, blank, expensive | Context tokens (Opus) |
| Catalytic space | Huge, arbitrary, must restore | Disk + small models |
| Restoration | Byte-identical after computation | Merkle root match |

An agent can "think" using 10GB of disk while holding 100KB in context. The disk is the extended mind. Context is just the hot path.

---

## Mathematical Foundation

### Fourier Analogy
- **Full repo** = time domain (huge, every sample)
- **LITE pack** = frequency domain (compact, structural)
- **Transform is lossless** - reconstruct full from LITE + catalytic store

### Fractal Analogy
- **Hash** = seed (64 bytes)
- **Content** = generated structure (unlimited)
- **Catalytic store** = iteration rules
- Small input → infinite output, deterministic

### Sheaf Analogy
- **Context window** = local section
- **Full repo** = global structure
- **Restoration proof** = gluing condition
- Local edits that violate gluing are rejected

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OPUS (Orchestrator)                  │
│                   Context: 100K tokens                  │
│            Holds: hashes, schemas, control flow         │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ Small Model │ │ Small Model │ │ Small Model │
   │   Worker    │ │   Worker    │ │   Worker    │
   │ (1B-7B)     │ │ (1B-7B)     │ │ (1B-7B)     │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │              CATALYTIC STORE (Disk)                 │
   │                                                     │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
   │  │ Content-    │  │   Merkle    │  │ Instruction │ │
   │  │ Addressed   │  │   Trees     │  │  Schemas    │ │
   │  │   Store     │  │             │  │             │ │
   │  └─────────────┘  └─────────────┘  └─────────────┘ │
   │                                                     │
   │  Verification: O(log n) via Merkle roots           │
   │  Restoration: Proven, not assumed                  │
   └─────────────────────────────────────────────────────┘
```

---

## Build Order (Dependency Chain)

### Phase 1: Core Primitives (No Dependencies)

#### 1.1 Content-Addressed Store
**File**: `TOOLS/catalytic_store.py` (~100 LOC)

```python
class CatalyticStore:
    def __init__(self, root: Path):
        self.root = root  # e.g., CONTRACTS/_runs/_cas/

    def put(self, content: bytes) -> str:
        """Store content, return hash (externalize)."""
        h = sha256(content).hexdigest()
        path = self.root / h[:2] / h[2:4] / h
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return h

    def get(self, h: str) -> bytes:
        """Retrieve content by hash (internalize)."""
        return (self.root / h[:2] / h[2:4] / h).read_bytes()

    def has(self, h: str) -> bool:
        return (self.root / h[:2] / h[2:4] / h).exists()

    def verify(self, h: str, content: bytes) -> bool:
        return sha256(content).hexdigest() == h
```

**Test**: `store.verify(store.put(b"hello"), b"hello") == True`

#### 1.2 Merkle Tree
**File**: `TOOLS/merkle.py` (~150 LOC)

```python
class MerkleTree:
    def __init__(self, items: List[Tuple[str, str]]):
        """items = [(path, content_hash), ...]"""
        self.leaves = sorted(items)
        self.root = self._build()

    def _build(self) -> str:
        """Build tree, return root hash."""
        if not self.leaves:
            return sha256(b"empty").hexdigest()
        level = [sha256(f"{p}:{h}".encode()).hexdigest()
                 for p, h in self.leaves]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i+1] if i+1 < len(level) else left
                next_level.append(sha256(f"{left}{right}".encode()).hexdigest())
            level = next_level
        return level[0]

    @classmethod
    def from_directory(cls, path: Path) -> "MerkleTree":
        items = []
        for f in path.rglob("*"):
            if f.is_file():
                h = sha256(f.read_bytes()).hexdigest()
                items.append((str(f.relative_to(path)), h))
        return cls(items)

    def diff(self, other: "MerkleTree") -> dict:
        """Return differences between trees."""
        self_dict = dict(self.leaves)
        other_dict = dict(other.leaves)
        return {
            "added": {k: v for k, v in other_dict.items() if k not in self_dict},
            "removed": {k: v for k, v in self_dict.items() if k not in other_dict},
            "changed": {k: (self_dict[k], other_dict[k])
                       for k in self_dict.keys() & other_dict.keys()
                       if self_dict[k] != other_dict[k]}
        }
```

**Test**: `MerkleTree.from_directory(path).root` is deterministic

#### 1.3 Update catalytic_runtime.py
**Change**: Replace O(n) snapshot with Merkle root

```python
# Before (O(n) - hash every file):
def snapshot_domains(self):
    for domain in self.catalytic_domains:
        snapshot = CatalyticSnapshot(domain)
        snapshot.capture()  # Hashes every file

# After (O(n) build, O(1) compare):
def snapshot_domains(self):
    for domain in self.catalytic_domains:
        tree = MerkleTree.from_directory(domain)
        self.pre_trees[domain] = tree
        print(f"[catalytic] Merkle root: {tree.root[:16]}...")

def verify_restoration(self):
    for domain, pre_tree in self.pre_trees.items():
        post_tree = MerkleTree.from_directory(domain)
        if pre_tree.root != post_tree.root:
            diff = pre_tree.diff(post_tree)
            return False, diff
    return True, {}
```

---

### Phase 2: Context Protocol

#### 2.1 Catalytic Context Protocol (CCP)
**File**: `TOOLS/ccp.py` (~100 LOC)

```python
class CatalyticContext:
    """Page data between context (expensive) and store (cheap)."""

    def __init__(self, store: CatalyticStore):
        self.store = store
        self.live_refs: Set[str] = set()  # Hashes currently "in use"

    def externalize(self, data: any) -> str:
        """Move data from context to store. Return hash."""
        content = json.dumps(data).encode() if not isinstance(data, bytes) else data
        h = self.store.put(content)
        self.live_refs.add(h)
        return h

    def internalize(self, h: str) -> any:
        """Retrieve data from store to context."""
        content = self.store.get(h)
        try:
            return json.loads(content)
        except:
            return content

    def release(self, h: str):
        """Mark hash as no longer needed (will be verified as restored)."""
        self.live_refs.discard(h)

    def summary(self) -> dict:
        """What's currently externalized."""
        return {
            "live_refs": len(self.live_refs),
            "total_bytes": sum(len(self.store.get(h)) for h in self.live_refs)
        }
```

**Usage**:
```python
ccp = CatalyticContext(store)

# Instead of holding 10KB in context:
large_ast = parse_file("huge.py")  # 10KB

# Hold 64-byte hash:
ast_hash = ccp.externalize(large_ast)  # 64 bytes

# Later, when needed:
ast = ccp.internalize(ast_hash)
```

---

### Phase 3: Instruction Harness (For Small Models)

#### 3.1 Instruction Schema Format
**File**: `CONTRACTS/fixtures/catalytic/instruction_schemas/extract_structure.json`

```json
{
  "id": "extract_structure",
  "version": "1.0",
  "description": "Extract structural elements from code",
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {"type": "string"},
      "language": {"type": "string"}
    },
    "required": ["code"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "functions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "line": {"type": "integer"},
            "params": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["name", "line"]
        }
      },
      "classes": {"type": "array"},
      "imports": {"type": "array"}
    },
    "required": ["functions"]
  },
  "prompt_template": "Extract all functions, classes, and imports from this {language} code. Output valid JSON matching the schema.\n\nCode:\n```\n{code}\n```\n\nOutput:",
  "constraints": [
    "Output MUST be valid JSON",
    "Every function MUST have name and line number",
    "If parsing fails, return empty arrays, not error messages"
  ]
}
```

#### 3.2 Instruction Harness
**File**: `SKILLS/instruction-harness/run.py` (~200 LOC)

```python
class InstructionHarness:
    def __init__(self, schema_path: Path, model: str = "neural-chat"):
        self.schema = json.loads(schema_path.read_text())
        self.model = model
        self.max_retries = 3

    def execute(self, input_data: dict) -> dict:
        """Run instruction with validation and retry."""
        # Validate input
        if not self._validate(input_data, self.schema["input_schema"]):
            return {"success": False, "error": "Invalid input"}

        # Build prompt
        prompt = self.schema["prompt_template"].format(**input_data)
        prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in self.schema["constraints"])

        for attempt in range(self.max_retries):
            # Run model
            output = self._run_model(prompt)

            # Try to parse JSON
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as e:
                prompt = f"Your output was not valid JSON. Error: {e}\nTry again:\n{prompt}"
                continue

            # Validate against schema
            if self._validate(parsed, self.schema["output_schema"]):
                return {"success": True, "data": parsed, "attempts": attempt + 1}

            # Add constraint guidance for retry
            violations = self._explain_violations(parsed, self.schema["output_schema"])
            prompt = f"Output violated schema:\n{violations}\nTry again:\n{prompt}"

        return {"success": False, "error": "Max retries exceeded", "last_output": output}

    def _run_model(self, prompt: str) -> str:
        """Call local Ollama model."""
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout

    def _validate(self, data: dict, schema: dict) -> bool:
        """Validate data against JSON schema."""
        try:
            import jsonschema
            jsonschema.validate(data, schema)
            return True
        except:
            return False
```

---

### Phase 4: Integration

#### 4.1 Catalytic Swarm Bridge
**File**: `TOOLS/catalytic_swarm.py` (~150 LOC)

```python
class CatalyticSwarm:
    """Bridge between catalytic primitives and swarm execution."""

    def __init__(self, store: CatalyticStore, num_workers: int = 4):
        self.store = store
        self.ccp = CatalyticContext(store)
        self.num_workers = num_workers

    def map(self, instruction_id: str, inputs: List[dict]) -> List[dict]:
        """
        Parallel map over inputs using instruction schema.

        Opus calls this with instruction ID and list of inputs.
        Swarm executes in parallel with small models.
        Results are validated before returning.
        """
        schema_path = PROJECT_ROOT / "CONTRACTS/fixtures/catalytic/instruction_schemas" / f"{instruction_id}.json"

        # Externalize inputs to store (save context)
        input_hashes = [self.ccp.externalize(inp) for inp in inputs]

        # Prepare tasks for swarm
        tasks = [
            {
                "id": f"{instruction_id}_{i}",
                "instruction": instruction_id,
                "input_hash": h,
                "model": "neural-chat"
            }
            for i, h in enumerate(input_hashes)
        ]

        # Execute via swarm-governor
        results = self._execute_swarm(tasks, schema_path)

        # Release inputs (they've been processed)
        for h in input_hashes:
            self.ccp.release(h)

        return results

    def _execute_swarm(self, tasks: List[dict], schema_path: Path) -> List[dict]:
        """Delegate to swarm-governor with instruction harness."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        harness = InstructionHarness(schema_path)
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._run_task, task, harness): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"task_id": task["id"], "success": False, "error": str(e)})

        return results

    def _run_task(self, task: dict, harness: InstructionHarness) -> dict:
        """Execute single task with harness."""
        input_data = self.ccp.internalize(task["input_hash"])
        result = harness.execute(input_data)
        result["task_id"] = task["id"]
        return result
```

---

## File Manifest

```
TOOLS/
├── catalytic_store.py      # Content-addressed store (Phase 1.1)
├── merkle.py               # Merkle tree verification (Phase 1.2)
├── catalytic_runtime.py    # UPDATE: Use Merkle roots (Phase 1.3)
├── ccp.py                  # Catalytic Context Protocol (Phase 2.1)
├── catalytic_swarm.py      # Swarm integration (Phase 4.1)
└── catalytic_validator.py  # UPDATE: Verify Merkle proofs

SKILLS/
└── instruction-harness/
    ├── SKILL.md
    ├── run.py              # Instruction execution (Phase 3.2)
    ├── validate.py
    └── fixtures/

CONTRACTS/fixtures/catalytic/
└── instruction_schemas/
    ├── extract_structure.json
    ├── summarize_section.json
    ├── validate_format.json
    └── transform_data.json
```

---

## Usage Example (End-to-End)

```python
# Opus orchestrating a 10,000-file codebase analysis

from catalytic_store import CatalyticStore
from catalytic_swarm import CatalyticSwarm

store = CatalyticStore(Path("CONTRACTS/_runs/_cas"))
swarm = CatalyticSwarm(store, num_workers=8)

# 1. Externalize all files (context holds hashes only)
file_hashes = {}
for f in Path("src").rglob("*.py"):
    file_hashes[str(f)] = store.put(f.read_bytes())
# Context: 10,000 hashes × 64 bytes = 640KB (fits easily)
# Store: 10,000 files × 10KB avg = 100MB (on disk)

# 2. Parallel extraction via swarm
inputs = [{"code": store.get(h).decode(), "path": p} for p, h in file_hashes.items()]
results = swarm.map("extract_structure", inputs)
# Swarm: 8 workers × 1250 files each
# Small models: ~50K tokens each
# Opus: 0 tokens (just orchestration)

# 3. Aggregate results (back in context)
all_functions = []
for r in results:
    if r["success"]:
        all_functions.extend(r["data"]["functions"])

# 4. Verify restoration
runtime = CatalyticRuntime(...)
assert runtime.verify_restoration()  # O(1) Merkle root comparison

# Result: Complete analysis of 10,000 files
# Opus tokens: ~1000 (orchestration only)
# Small model tokens: ~500,000 (cheap, parallel)
# Ratio: 500:1 token efficiency
```

---

## Success Criteria

1. **Merkle verification works**: O(log n) not O(n)
2. **CCP externalize/internalize works**: Hashes in context, content on disk
3. **Instruction harness validates**: Small model output matches schema or fails
4. **Swarm executes in parallel**: N workers, validated results
5. **End-to-end demo**: Analyze 1000+ files, <1000 Opus tokens

---

## What Makes This Revolutionary

| Before | After |
|--------|-------|
| Context IS the workspace | Context is the cache |
| Limited by token window | Limited by disk |
| O(n) verification | O(log n) Merkle |
| Small models hallucinate | Small models fill templates |
| Opus does all the work | Opus orchestrates, swarm executes |
| Trust the model | Verify the proof |

**The context window stops being the limit. The agent's mind is the entire filesystem.**

---

## Build Sequence (Copy-Paste Ready)

```bash
# Phase 1: Core Primitives
# 1.1 Content-Addressed Store
touch TOOLS/catalytic_store.py  # ~100 LOC

# 1.2 Merkle Tree
touch TOOLS/merkle.py  # ~150 LOC

# 1.3 Update Runtime
# Edit TOOLS/catalytic_runtime.py to use Merkle

# Phase 2: Context Protocol
touch TOOLS/ccp.py  # ~100 LOC

# Phase 3: Instruction Harness
mkdir -p SKILLS/instruction-harness/fixtures
touch SKILLS/instruction-harness/SKILL.md
touch SKILLS/instruction-harness/run.py  # ~200 LOC
touch SKILLS/instruction-harness/validate.py

# Phase 3.1: Instruction Schemas
mkdir -p CONTRACTS/fixtures/catalytic/instruction_schemas
touch CONTRACTS/fixtures/catalytic/instruction_schemas/extract_structure.json
touch CONTRACTS/fixtures/catalytic/instruction_schemas/summarize_section.json

# Phase 4: Integration
touch TOOLS/catalytic_swarm.py  # ~150 LOC

# Total: ~700 LOC for complete catalytic system
```

---

## If Context Runs Out

Read this file. It contains everything:
- Theory (Fourier/Fractal/Sheaf analogies)
- Architecture diagram
- All code structures
- Build order with dependencies
- Usage examples
- Success criteria

Start from Phase 1.1 (catalytic_store.py) and follow the chain.

**The goal**: Agent reasons over unlimited data. Context holds hashes. Disk holds content. Small models grind. Proofs verify. Revolution achieved.
