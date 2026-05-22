"""Corrected Wormhole Inference — Hayden-Preskill unscrambling."""
import sys,io,torch,torch.nn as nn,time,json,gc,importlib.util
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from safetensors import safe_open
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')

MD=r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B';dev=torch.device('cuda')
sys.path.insert(0,'THOUGHT/LAB/CAT_CAS/33_mera_compression')
import _paths
spec=importlib.util.spec_from_file_location('p','THOUGHT/LAB/CAT_CAS/33_mera_compression/13_patch_model.py')
patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)

# Load correction tape
tape_path=str(_paths.HOLO_MODELS/'qwen_27b_correction_tape.pt')
tape=torch.load(tape_path,map_location='cpu',weights_only=True)
print(f'Loaded correction tape: {len(tape)} entries',flush=True)

# Build corrected WormholeLinear
class CorrectedWormholeLinear(nn.Module):
    def __init__(self,U,SVh,corr_u,corr_v,corr_s,bias=None):
        super().__init__()
        self.U=nn.Parameter(U,requires_grad=False)
        self.SVh=nn.Parameter(SVh,requires_grad=False)
        self.corr_u=nn.Parameter(corr_u,requires_grad=False)
        self.corr_v=nn.Parameter(corr_v,requires_grad=False)
        self.corr_s=corr_s
        self.bias=nn.Parameter(bias,requires_grad=False) if bias is not None else None
    
    def forward(self,x):
        # Base HoloLinear: x @ SVh^T @ U^T
        h=torch.matmul(x,self.SVh.t())  # (B,S,k)
        out=torch.matmul(h,self.U.t())  # (B,S,out_dim)
        # STEP 5: Hayden-Preskill correction
        # delta_U = corr_u * corr_s @ corr_v^T  -> applied through SVh
        # correction = x @ SVh^T @ delta_U^T = h @ corr_v * corr_s * corr_u^T
        proj=torch.matmul(h,self.corr_v.unsqueeze(-1)).squeeze(-1)  # (B,S,)
        delta=proj.unsqueeze(-1)*self.corr_u.unsqueeze(0)*self.corr_s  # (B,S,out_dim)
        out=out+delta
        if self.bias is not None:out+=self.bias
        return out

# Build model + patch with correction
cf=AutoConfig.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
with torch.device('meta'):m=AutoModelForCausalLM.from_config(cf,trust_remote_code=True)
ag={};asvh={}
for mn,path in _paths.MODULE_PATHS.items():
    if not path.exists():continue
    g,svh=patcher.parse_wormhole(path);ag.update(g);asvh.update(svh)

# Patch with CorrectedWormholeLinear
patched=0
for name,mod in list(m.named_modules()):
    if not isinstance(mod,nn.Linear):continue
    parts=name.split('.')
    layer_idx=None
    for i,p in enumerate(parts):
        if p in ('layers','blocks') and i+1<len(parts):
            try:layer_idx=int(parts[i+1])
            except:pass
            break
    if layer_idx is None:continue
    wt_parts=[]
    for p in ('mlp','self_attn','linear_attn','attn'):
        if p in parts:idx=parts.index(p);wt_parts=parts[idx:];break
    if not wt_parts:continue
    wt_name='.'.join(wt_parts)+'.weight'
    
    # Get base wormhole reconstruction
    wl=patcher.WormholeLinear.from_wormhole_rotations(wt_name,layer_idx,ag,asvh,device=dev)
    if wl is None:continue
    
    # Add correction tape
    tape_key=f'{wt_name}.L{layer_idx}'
    corr=tape.get(tape_key)
    if corr is not None:
        cw=CorrectedWormholeLinear(
            wl.U.data,wl.SVh.data,
            corr['u'].to(dev,dtype=torch.bfloat16),
            corr['v'].to(dev,dtype=torch.bfloat16),
            corr['s'],
            wl.bias.data if wl.bias is not None else None
        )
    else:
        cw=wl
    
    pname=name.rsplit('.',1)
    if len(pname)==2:
        setattr(m.get_submodule(pname[0]),pname[1],cw)
        patched+=1

m.to_empty(device=dev)
# Load norms
with open(f'{MD}/model.safetensors.index.json')as f:idx=json.load(f)
wm=idx['weight_map'];sf=defaultdict(list)
hlmods=set()
for n,mod in m.named_modules():
    if isinstance(mod,(patcher.WormholeLinear,CorrectedWormholeLinear)):hlmods.add(n)
for name,param in m.named_parameters():
    if param.device.type!='meta':continue
    mod_name=name.rsplit('.',1)[0]
    if mod_name in hlmods:continue
    sfk=name.replace('model.','model.language_model.',1)
    if sfk in wm:sf[wm[sfk]].append((name,sfk))
for shard,keys in sf.items():
    with safe_open(f'{MD}/{shard}',framework='pt',device='cpu')as sf2:
        for name,sfk in keys:
            val=sf2.get_tensor(sfk)
            try:
                parts=name.rsplit('.',1)
                if len(parts)==2:
                    parent=m.get_submodule(parts[0]);mm=getattr(parent,parts[1])
                    if isinstance(mm,nn.Parameter):mm.data=val.to(dev,dtype=mm.dtype)
            except:pass
m.eval();gc.collect();torch.cuda.empty_cache()
print(f'Patched: {patched} | GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB',flush=True)

# Test
tok=AutoTokenizer.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
prompts=['The meaning of life is','Artificial intelligence will','The capital of France is']
for p in prompts:
    ids=tok(p,return_tensors='pt').to(dev)
    torch.cuda.synchronize();t0=time.perf_counter()
    with torch.no_grad():out=m.generate(**ids,max_new_tokens=25,do_sample=True,temperature=0.7)
    torch.cuda.synchronize();dt=time.perf_counter()-t0
    text=tok.decode(out[0],skip_special_tokens=True)
    try:text=text.encode('ascii',errors='replace').decode('ascii')
    except:pass
    print(f'  {25/dt:.1f} tok/s | {repr(text[:120])}',flush=True)
