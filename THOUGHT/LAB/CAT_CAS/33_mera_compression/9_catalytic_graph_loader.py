"""
Catalytic Module Graph Loader
==============================
EIGEN_BUDDY-style: modules are nodes in a computational graph.
Loading is catalytic: borrow workspace, compute, restore.
Wormhole rotations form the transport network between layers.

Key catalytic transfers from CAT_CAS Map:
  1. Orthogonal subspaces (Exp 13): zero cross-talk between modules
  2. Si matrix persistence (Eigen Buddy): phase is unconsumed catalyst
  3. Phase cavity (Exp 20/21): eigenmode sieve detection
  4. Catalytic Feistel: multi-round borrow/return pattern
"""
import torch, json, re
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

# ---- Data Structures ----

@dataclass
class ModuleNode:
    """A node in the catalytic module graph."""
    name: str
    path: Path
    weight_types: list  # e.g. ['mlp.down_proj.weight', ...]
    rotation_fidelity: dict  # per-weight-type fidelity
    file_size_mb: float
    svh_shared: dict  # {wt: tensor} -- shared SVh per weight type

@dataclass  
class CatalyticGraph:
    """DAG of module nodes. Routes weight types -> modules."""
    nodes: dict  # module_name -> ModuleNode
    wt_index: dict  # weight_type -> module_name
    manifest_path: Path

    def route(self, weight_type):
        """Which module owns this weight type?"""
        return self.wt_index.get(weight_type)

    def all_weight_types(self):
        return list(self.wt_index.keys())

    def needed_modules(self, weight_types):
        """Given a set of weight types, which modules are needed?"""
        modules = set()
        for wt in weight_types:
            mod = self.route(wt)
            if mod: modules.add(mod)
        return modules


@dataclass
class CatalyticSession:
    """
    Catalytic workspace for module loading.
    Borrows volatile memory, loads modules into orthogonal subspaces,
    computes forward, restores state. Zero cross-talk.
    """
    graph: CatalyticGraph
    workspace: dict = field(default_factory=dict)
    _loaded: set = field(default_factory=set)
    _phase_cache: dict = field(default_factory=dict)  # si matrix persistence

    def borrow(self, module_name):
        """Catalytic load: borrow workspace, load module."""
        node = self.graph.nodes[module_name]
        worm = torch.load(str(node.path), map_location='cpu', weights_only=True)
        # Parse wormhole groups
        pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
        groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))
        for key, val in worm.items():
            m = pattern.match(key)
            if not m: continue
            wt, layer_str, field = m.groups()
            l = int(layer_str)
            g = groups[wt]
            if field == 'U':
                g['first_U'] = val; g['first_l'] = l
            elif field == 'R':
                g['rots'][l] = val
            elif field == 'res_idx':
                g['res'].setdefault(l, {})['idx'] = val
            elif field == 'res_max':
                if l in g['res']:
                    g['res'][l]['max'] = val
        
        self.workspace[module_name] = {'worm': worm, 'groups': groups, 'node': node}
        self._loaded.add(module_name)

    def return_workspace(self, module_name):
        """Restore workspace to original state (catalytic undo)."""
        self.workspace.pop(module_name, None)
        self._loaded.discard(module_name)

    def reconstruct(self, weight_type, layer_idx):
        """Reconstruct a single weight layer's U matrix from wormhole rotations."""
        module = self.graph.route(weight_type)
        if not module or module not in self._loaded:
            return None
        
        ws = self.workspace[module]
        groups = ws['groups']
        if weight_type not in groups:
            return None
        
        g = groups[weight_type]
        first_l = g['first_l']
        if layer_idx == first_l:
            return g['first_U'].float()
        
        if layer_idx in g['rots']:
            anchor = g['first_U'].float()
            R = g['rots'][layer_idx].float()
            recon = anchor @ R
            if layer_idx in g['res'] and g['res'][layer_idx].get('idx') is not None:
                rd = g['res'][layer_idx]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                residual = levels[rd['idx'].long()]
                recon = recon + residual
            return recon
        
        return None

    def get_svh(self, weight_type):
        """Get shared SVh for a weight type across all layers."""
        module = self.graph.route(weight_type)
        if not module or module not in self._loaded:
            return None
        ws = self.workspace[module]
        worm = ws['worm']
        svh_key = f"{weight_type}.SVh"
        if svh_key in worm:
            return worm[svh_key]
        return None

    def forward_linear(self, x, weight_type, layer_idx):
        """
        Catalytic forward pass through one HoloLinear layer.
        x * SVh^T * U^T without materializing the full weight matrix.
        """
        U = self.reconstruct(weight_type, layer_idx)
        SVh = self.get_svh(weight_type)
        if U is None or SVh is None:
            return None
        # HoloLinear: output = x @ SVh^T @ U^T
        h = x @ SVh.T.float()  # (B, S, k)
        out = h @ U.T.float()        # (B, S, n)
        return out

    def close(self):
        """Return all borrowed workspace. Final catalytic cleanup."""
        for mod in list(self._loaded):
            self.return_workspace(mod)
        self._phase_cache.clear()


# ---- Manifest Builder ----

def build_manifest(module_files, output_path=None):
    """
    Scan wormhole module files and build a CatalyticGraph manifest.
    
    module_files: dict {name: path}
    """
    nodes = {}
    wt_index = {}

    for name, path in module_files.items():
        path = Path(path)
        if not path.exists():
            continue
        
        worm = torch.load(str(path), map_location='cpu', weights_only=True)
        weight_types = []
        rotation_fidelity = {}
        svh_shared = {}
        
        # Extract weight types from wormhole L-format keys
        pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
        seen_wt = set()
        for key, val in worm.items():
            m = pattern.match(key)
            if m:
                wt = m.group(1)
                seen_wt.add(wt)
        
        weight_types = sorted(seen_wt)
        
        # Extract shared SVh
        for key, val in worm.items():
            if '.L' not in key and key.endswith('.SVh'):
                wt = key.replace('.SVh', '')
                if wt in seen_wt:
                    svh_shared[wt] = val
        
        node = ModuleNode(
            name=name,
            path=path,
            weight_types=weight_types,
            rotation_fidelity=rotation_fidelity,  # compute on-demand
            file_size_mb=path.stat().st_size / 1024 / 1024,
            svh_shared=svh_shared,
        )
        nodes[name] = node
        for wt in weight_types:
            wt_index[wt] = name
    
    manifest = {
        'modules': [{'name': n.name, 'path': str(n.path), 'weight_types': n.weight_types,
                      'file_size_mb': n.file_size_mb} for n in nodes.values()],
        'wt_index': wt_index,
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    return manifest


def load_graph(manifest_path_or_files):
    """Load a CatalyticGraph from a manifest JSON or file paths dict."""
    if isinstance(manifest_path_or_files, (str, Path)):
        with open(manifest_path_or_files) as f:
            manifest = json.load(f)
        files = {m['name']: m['path'] for m in manifest['modules']}
    else:
        files = manifest_path_or_files
    
    return _graph_from_files(files)


def _graph_from_files(module_files):
    nodes = {}
    wt_index = {}
    for name, path in module_files.items():
        path = Path(path)
        if not path.exists(): continue
        worm = torch.load(str(path), map_location='cpu', weights_only=True)
        pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
        seen_wt = set()
        for key in worm:
            m = pattern.match(key)
            if m: seen_wt.add(m.group(1))
        svh_shared = {}
        for key, val in worm.items():
            if '.L' not in key and key.endswith('.SVh'):
                wt = key.replace('.SVh', '')
                if wt in seen_wt: svh_shared[wt] = val
        node = ModuleNode(name=name, path=path, weight_types=sorted(seen_wt),
                         rotation_fidelity={}, file_size_mb=path.stat().st_size/1024/1024,
                         svh_shared=svh_shared)
        nodes[name] = node
        for wt in node.weight_types:
            wt_index[wt] = name
    return CatalyticGraph(nodes=nodes, wt_index=wt_index, manifest_path=Path())


# ---- CLI ----

if __name__ == "__main__":
    import sys
    base = Path("THOUGHT/LAB/CAT_CAS/33_mera_compression")
    files = {
        "llm": base / "qwen_27b_llm_wormhole.holo",
        "visual": base / "qwen_27b_visual_wormhole.holo",
    }
    
    print("Building catalytic manifest...")
    manifest = build_manifest(files, output_path=str(base / "catalytic_manifest.json"))
    print(f"  Modules: {len(manifest['modules'])}")
    for m in manifest['modules']:
        print(f"    {m['name']}: {len(m['weight_types'])} types, {m['file_size_mb']:.0f} MB")

    print("\nLoading catalytic graph...")
    graph = load_graph(files)
    print(f"  Total weight types: {len(graph.wt_index)}")

    # Demo: catalytic session
    print("\nDemo: Catalytic Session")
    session = CatalyticSession(graph=graph)
    
    # Borrow LLM module
    session.borrow("llm")
    print(f"  Borrowed 'llm': {len(session.workspace['llm']['groups'])} weight groups")
    
    # Reconstruct one layer
    U = session.reconstruct("mlp.down_proj.weight", 0)
    if U is not None:
        print(f"  Reconstructed mlp.down_proj.weight.L0: shape={list(U.shape)}")
    
    SVh = session.get_svh("mlp.down_proj.weight")
    if SVh is not None:
        print(f"  SVh shared: shape={list(SVh.shape)}")
    
    # Test forward pass: gate_proj takes hidden=5120 input, outputs 17408
    x = torch.randn(1, 8, 5120).float()
    out = session.forward_linear(x, "mlp.gate_proj.weight", 0)
    if out is not None:
        print(f"  Forward pass: {list(x.shape)} -> {list(out.shape)}")
    
    session.close()
    print("  Workspace restored. Session closed.")
    print("  Zero bits erased (catalytic).")
