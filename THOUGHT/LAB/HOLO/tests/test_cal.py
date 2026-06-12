import sys,io,torch,time,json,gc,importlib.util
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from safetensors import safe_open
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sys.path.insert(0,'THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression')
import _paths

spec=importlib.util.spec_from_file_location('p','THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/13_patch_model.py')
patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)

MD=r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
dev=torch.device('cuda')
torch.cuda.empty_cache();gc.collect()

paths={'llm':_paths.HOLO_MODELS/'qwen_27b_llm_calibrated.holo','visual':_paths.VISUAL_WORMHOLE,'aux':_paths.AUX_WORMHOLE}

cf=AutoConfig.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
with torch.device('meta'):m=AutoModelForCausalLM.from_config(cf,trust_remote_code=True)
ag={};asvh={}
for mn,path in paths.items():
    if not path.exists():continue
    g,svh=patcher.parse_wormhole(path);ag.update(g);asvh.update(svh)
m,stats=patcher.patch_model_with_wormhole(m,ag,asvh,device='meta')
m.to_empty(device=dev)

with open(f'{MD}/model.safetensors.index.json')as f:idx=json.load(f)
wm=idx['weight_map'];sf=defaultdict(list)
hlmods=set()
for n,mod in m.named_modules():
    if isinstance(mod,patcher.WormholeLinear):hlmods.add(n)
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
                    if isinstance(mm,torch.nn.Parameter):mm.data=val.to(dev,dtype=mm.dtype)
            except:pass
m.eval();gc.collect();torch.cuda.empty_cache()
mem=torch.cuda.memory_allocated()/1e9
meta=sum(1 for p in m.parameters() if p.device.type=='meta')
print(f'GPU: {mem:.1f} GB | {stats["patched"]} layers | {stats["compression"]:.1f}x | Meta: {meta}',flush=True)

tok=AutoTokenizer.from_pretrained(MD,local_files_only=True,trust_remote_code=True)
prompts=['The meaning of life is','Artificial intelligence will','The capital of France is']
total_tps=0
for p in prompts:
    ids=tok(p,return_tensors='pt').to(dev)
    torch.cuda.synchronize();t0=time.perf_counter()
    with torch.no_grad():out=m.generate(**ids,max_new_tokens=25,do_sample=True,temperature=0.7)
    torch.cuda.synchronize();dt=time.perf_counter()-t0
    newt=out.shape[1]-ids['input_ids'].shape[1];tps=newt/dt;total_tps+=tps
    text=tok.decode(out[0],skip_special_tokens=True)
    try:text=text.encode('ascii',errors='replace').decode('ascii')
    except:pass
    print(f'  {tps:.1f} tok/s | {repr(text[:100])}',flush=True)
avg=total_tps/len(prompts)
print(f'\nAvg: {avg:.1f} tok/s | GPU: {mem:.1f} GB | {stats["compression"]:.1f}x')
