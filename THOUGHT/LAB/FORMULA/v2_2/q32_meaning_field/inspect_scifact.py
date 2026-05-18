import os; os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"
from datasets import load_dataset
c = load_dataset("allenai/scifact","claims",split="train",trust_remote_code=True)
print("Keys:", list(c[0].keys()))
for k in c[0].keys():
    v = c[0][k]
    print(f"  {k}: {v if not isinstance(v,list) else f'list[{len(v)}]'}")
corpus = load_dataset("allenai/scifact","corpus",split="train",trust_remote_code=True)
print("\nCorpus keys:", list(corpus[0].keys()))
for k in corpus[0].keys():
    v = corpus[0][k]
    print(f"  {k}: {v if not isinstance(v,list) else f'list[{len(v)}]'}")
