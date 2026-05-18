"""Download NLI cross-encoder for Q32 gap closure."""
from sentence_transformers import CrossEncoder
print("Downloading cross-encoder/nli-MiniLM2-L6-H768...")
m = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
print("Done.")
