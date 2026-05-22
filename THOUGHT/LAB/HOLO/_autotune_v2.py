"""Autotune: Teacher forward -> save -> free -> Student train."""
import sys,io,torch,torch.nn as nn,time,json,gc,importlib.util,os
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from safetensors import safe_open
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')

MD=r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
dev=torch.device('cuda')
sys.path.insert(0,'THOUGHT/LAB/CAT_CAS/33_mera_compression')
import _paths
spec=importlib.util.spec_from_file_location('p','THOUGHT/LAB/CAT_CAS/33_mera_compression/13_patch_model.py')
patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
spec2=importlib.util.spec_from_file_location('md','THOUGHT/LAB/CAT_CAS/33_mera_compression/8_modular_decoder.py')
moddec=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(moddec)

class HL(nn.Module):
    def __init__(self,U,SVh):super().__init__();self.U=nn.Parameter(U,requires_grad=False);self.SVh=nn.Parameter(SVh,requires_grad=False)
    def forward(self,x):return x@self.SVh.t()@self.U.t()

def load_norms(model):
    with open(f'{MD}/model.safetensors.index.json')as f:idx=json.load(f)
    wm=idx['weight_map'];sf=defaultdict(list)
    hlmods=set()
    for n,mod in model.named_modules():
        if isinstance(mod,HL) or isinstance(mod,patcher.WormholeLinear):hlmods.add(n)
    for name,param in model.named_parameters():
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
                        parent=model.get_submodule(parts[0]);mm=getattr(parent,parts[1])
                        if isinstance(mm,nn.Parameter):mm.data=val.to(dev,dtype=mm.dtype)
                except:pass
    model.eval();gc.collect();torch.cuda.empty_cache()

tok=AutoTokenizer.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
tok.pad_token=tok.eos_token
texts=[
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming the world",
    "In the beginning the Universe was created",
    "The meaning of life is to explore and discover",
    "Mathematics is the language of nature",
    "The future belongs to those who believe in their dreams",
    "Science is a way of thinking much more than a body of knowledge",
    "Every great dream begins with a dreamer",
]
batch=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=48)
input_ids=batch['input_ids'].to(dev);attn=batch.get('attention_mask')
if attn is not None:attn=attn.to(dev)

# ===== PHASE 1: Teacher forward -> save hidden states -> free =====
print('=== PHASE 1: Teacher ===',flush=True)
cf=AutoConfig.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
with torch.device('meta'):teacher=AutoModelForCausalLM.from_config(cf,trust_remote_code=True)
sd_t=moddec.load_modules(['llm','visual'],module_dir=str(_paths.HOLO_MODELS))
remap_t={k.replace('model.language_model.','model.'):v for k,v in sd_t.items()}
tp=0
for n,m in teacher.named_modules():
    if isinstance(m,nn.Linear):
        wk=n+'.weight';uk=wk+'.U';sk=wk+'.SVh'
        if uk in remap_t and sk in remap_t:
            h=HL(remap_t[uk].to(dev,dtype=torch.bfloat16),remap_t[sk].to(dev,dtype=torch.bfloat16))
            p=n.rsplit('.',1)
            if len(p)==2:setattr(teacher.get_submodule(p[0]),p[1],h);tp+=1
teacher.to_empty(device=dev);load_norms(teacher)
print(f'  Patched: {tp}, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB',flush=True)

teacher.eval()
with torch.no_grad():
    t_out=teacher(input_ids,attention_mask=attn,output_hidden_states=True)
t_hidden=[h.float().cpu() if h is not None else None for h in t_out.hidden_states]
print(f'  Saved {len(t_hidden)} teacher hidden states to CPU',flush=True)

# Free teacher
for attr in list(teacher.__dict__.keys()):
    try:delattr(teacher,attr)
    except:pass
del teacher;del t_out;del sd_t;del remap_t
gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
print(f'  GPU after free: {torch.cuda.memory_allocated()/1e9:.1f} GB',flush=True)

# ===== PHASE 2: Student train against saved teacher states =====
print('=== PHASE 2: Student Training ===',flush=True)
with torch.device('meta'):student=AutoModelForCausalLM.from_config(cf,trust_remote_code=True)
ag={};asvh={}
for mn,path in _paths.MODULE_PATHS.items():
    if not path.exists():continue
    g,svh=patcher.parse_wormhole(path);ag.update(g);asvh.update(svh)
student,stats=patcher.patch_model_with_wormhole(student,ag,asvh,device='meta')
student.to_empty(device=dev);load_norms(student)
print(f'  Patched: {stats["patched"]}, GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB',flush=True)

# Baseline
student.eval()
with torch.no_grad():
    s_out=student(input_ids,attention_mask=attn,output_hidden_states=True)
bl=0.0;nl=min(len(t_hidden),len(s_out.hidden_states))
for l in range(nl):
    th=t_hidden[l];sh=s_out.hidden_states[l]
    if th is not None and sh is not None:
        bl+=nn.functional.mse_loss(sh.float().cpu(),th).item()
bl/=max(nl,1)
print(f'  Baseline MSE: {bl:.6f}',flush=True)

# Train SVh only (shared per type, ~163M params)
for n,p in student.named_parameters():p.requires_grad=('SVh' in n)
n_train=sum(p.numel() for p in student.parameters() if p.requires_grad)
opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5)
print(f'  Trainable: {n_train:,} SVh params',flush=True)

best=bl
for epoch in range(5):
    t0=time.perf_counter()
    student.train()
    s_out=student(input_ids,attention_mask=attn,output_hidden_states=True)
    loss=0.0
    for l in range(nl):
        th=t_hidden[l];sh=s_out.hidden_states[l]
        if th is not None and sh is not None:
            loss+=nn.functional.mse_loss(sh.float(),th.to(dev).float())
    loss/=max(nl,1)
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],0.1)
    opt.step()
    dt=time.perf_counter()-t0
    improved='*' if loss.item()<best else' '
    best=min(best,loss.item())
    print(f'  Epoch {epoch+1}: loss={loss.item():.6f} {improved} [{dt:.1f}s]',flush=True)

pct=(bl-best)/max(bl,1e-9)*100
print(f'\n  Baseline: {bl:.6f} -> Best: {best:.6f} ({pct:.1f}% improvement)')
print(f'  GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# Save calibrated
student.eval()
out_path=str(_paths.HOLO_MODELS/'qwen_27b_llm_autotuned_v2.holo')
torch.save({k:v.cpu() for k,v in student.state_dict().items() if 'SVh' in k},out_path)
print(f'  Saved SVh weights to {out_path}')
