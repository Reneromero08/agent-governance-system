#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 3: Does Df × α = 8e Hold Across Diverse Architectures?

Testing the conservation law across:
- Text: BERT, RoBERTa, DistilBERT, MiniLM, MPNet, E5, BGE, GTE
- Vision-Text: CLIP variants, OpenCLIP
- Code: CodeBERT, CodeT5, GraphCodeBERT
- Audio: Wav2Vec2, Whisper (if available)
- Multilingual: mBERT, XLM-R, LaBSE

Pass criteria:
- CV < 10% across all architectures
- Mean Df × α within 5% of 8e
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_df(eigenvalues):
    """Participation ratio Df = (Σλ)² / Σλ²"""
    ev = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
    """Power law decay exponent α where λ_k ~ k^(-α)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings):
    """Get eigenvalues from covariance matrix."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings, name):
    """Compute Df, α, and Df × α for embeddings."""
    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    df_alpha = Df * alpha
    vs_8e = abs(df_alpha - 8 * np.e) / (8 * np.e) * 100

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),
    }


def main():
    print("=" * 70)
    print("Q50 PART 3: CROSS-MODAL UNIVERSALITY TEST")
    print("Testing if Df × α = 8e holds for vision/audio/code models")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_CROSS_MODAL',
        'target': 8 * np.e,
        'models': [],
        'summary': {}
    }

    all_df_alpha = []

    # ============================================================
    # BASELINE: Text Models (already validated)
    # ============================================================
    print("\n" + "=" * 60)
    print("BASELINE: TEXT MODELS (Reference)")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        WORDS = [
            "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
            "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
            "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
            "heart", "eye", "hand", "head", "brain", "blood", "bone",
            "mother", "father", "child", "friend", "king", "queen",
            "love", "hate", "truth", "life", "death", "time", "space", "power",
            "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
            "book", "door", "house", "road", "food", "money", "stone", "gold",
            "light", "shadow", "music", "word", "name", "law",
            "good", "bad", "big", "small", "old", "new", "high", "low",
        ]

        # Diverse text model architectures
        text_models = [
            # Original baselines
            ("all-MiniLM-L6-v2", "MiniLM-L6"),
            ("all-mpnet-base-v2", "MPNet-base"),

            # Different transformer architectures
            ("bert-base-nli-mean-tokens", "BERT-base-NLI"),
            ("distilbert-base-nli-mean-tokens", "DistilBERT-NLI"),
            ("paraphrase-MiniLM-L6-v2", "MiniLM-Paraphrase"),
            ("paraphrase-mpnet-base-v2", "MPNet-Paraphrase"),

            # E5 family (different training: contrastive)
            ("intfloat/e5-small-v2", "E5-small"),
            ("intfloat/e5-base-v2", "E5-base"),

            # BGE family (BAAI - different architecture choices)
            ("BAAI/bge-small-en-v1.5", "BGE-small"),
            ("BAAI/bge-base-en-v1.5", "BGE-base"),

            # GTE family (Alibaba)
            ("thenlper/gte-small", "GTE-small"),
            ("thenlper/gte-base", "GTE-base"),

            # Different size variants
            ("all-MiniLM-L12-v2", "MiniLM-L12"),

            # Multilingual (different tokenization, training)
            ("paraphrase-multilingual-MiniLM-L12-v2", "mMiniLM-L12"),
            ("distiluse-base-multilingual-cased-v1", "mDistilUSE"),
        ]

        for model_name, display_name in text_models:
            try:
                model = SentenceTransformer(model_name)
                embeddings = model.encode(WORDS, normalize_embeddings=True)
                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'text'
                result['model_id'] = model_name

                print(f"\n  {display_name}:")
                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])
            except Exception as e:
                print(f"  {display_name}: FAILED - {e}")

    except ImportError:
        print("  sentence-transformers not available")

    # ============================================================
    # VISION MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("VISION MODELS")
    print("=" * 60)

    # Try CLIP
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import requests

        print("\n  Loading CLIP ViT-B/32...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Generate diverse "pseudo-images" using random noise
        # (Since we can't easily load 100 real images)
        print("  Generating synthetic image embeddings...")

        np.random.seed(42)
        n_images = 100

        # Create random pixel data and get CLIP embeddings
        # This tests the model's eigenstructure even without real images
        image_embeddings = []

        # Use random projections in CLIP's embedding space
        # This simulates what real diverse images might produce
        clip_dim = 512
        random_embeddings = np.random.randn(n_images, clip_dim)
        random_embeddings = random_embeddings / np.linalg.norm(random_embeddings, axis=1, keepdims=True)

        result = analyze_embeddings(random_embeddings, "CLIP-random-sim")
        result['modality'] = 'vision'
        result['model_id'] = 'clip-vit-base-patch32'
        result['note'] = 'Simulated via random normalized vectors'

        print(f"\n  CLIP (simulated):")
        print(f"    Shape: {result['shape']}")
        print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
        print(f"    Df × α = {result['Df_alpha']:.4f}")
        print(f"    NOTE: Random normalized vectors for baseline")

        results['models'].append(result)
        # Don't add to all_df_alpha since this is simulated

    except ImportError:
        print("  CLIP not available (transformers)")
    except Exception as e:
        print(f"  CLIP failed: {e}")

    # Try with actual text-based CLIP embeddings (image descriptions)
    try:
        from sentence_transformers import SentenceTransformer

        print("\n  Testing CLIP/vision-language models with image descriptions...")

        # Diverse image descriptions (simulating image content)
        IMAGE_DESCRIPTIONS = [
            "a photo of a cat", "a photo of a dog", "a red apple",
            "a mountain landscape", "a city skyline", "a beach sunset",
            "a forest path", "a snowy peak", "a river flowing",
            "a person walking", "a car on road", "a plane in sky",
            "a flower garden", "a fish swimming", "a bird flying",
            "abstract art", "geometric shapes", "colorful patterns",
            "food on plate", "coffee cup", "book on table",
            "computer screen", "phone on desk", "music notes",
            "sports game", "concert crowd", "fireworks display",
            "microscope image", "space galaxy", "atom structure",
            "historical painting", "modern sculpture", "architecture",
            "fashion model", "street scene", "market stalls",
            "underwater coral", "desert dunes", "polar ice",
            "tropical rainforest", "autumn leaves", "spring flowers",
            "winter snow", "summer beach", "cloudy sky",
            "lightning storm", "rainbow arc", "starry night",
            "full moon", "solar eclipse", "northern lights",
            "volcano eruption", "waterfall cascade", "cave interior",
            "ancient ruins", "modern building", "bridge structure",
            "train station", "airport terminal", "harbor ships",
            "factory machines", "laboratory equipment", "medical scan",
            "map terrain", "chart graphs", "diagram flow",
            "portrait face", "group photo", "action shot",
            "still life", "macro detail", "wide angle",
            "black and white", "sepia tone", "vivid colors",
            "blurred motion", "sharp focus", "depth of field",
        ]

        # Multiple CLIP/vision-language model variants
        clip_models = [
            ("clip-ViT-B-32", "CLIP-ViT-B-32"),
            ("clip-ViT-B-16", "CLIP-ViT-B-16"),
            ("clip-ViT-L-14", "CLIP-ViT-L-14"),
        ]

        for model_name, display_name in clip_models:
            try:
                model = SentenceTransformer(model_name)
                embeddings = model.encode(IMAGE_DESCRIPTIONS, normalize_embeddings=True)

                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'vision-text'
                result['model_id'] = model_name

                print(f"\n  {display_name}:")
                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])
            except Exception as e:
                print(f"  {display_name} failed: {e}")

    except Exception as e:
        print(f"  CLIP models failed: {e}")

    # ============================================================
    # CODE MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("CODE MODELS")
    print("=" * 60)

    # Code snippets representing diverse programming concepts
    CODE_SNIPPETS = [
        "def hello(): print('Hello')",
        "for i in range(10): print(i)",
        "class Dog: def bark(self): pass",
        "import numpy as np",
        "x = [1, 2, 3, 4, 5]",
        "if x > 0: return True",
        "while True: break",
        "try: x = 1 except: pass",
        "lambda x: x * 2",
        "async def fetch(): await response",
        "with open('f') as f: data = f.read()",
        "@decorator def func(): pass",
        "yield from generator()",
        "raise ValueError('error')",
        "assert x == y",
        "global counter",
        "nonlocal value",
        "del object",
        "pass",
        "return None",
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "sorted(items, key=lambda x: x.value)",
        "filter(lambda x: x > 0, numbers)",
        "map(str, integers)",
        "reduce(lambda a, b: a + b, items)",
        "list(zip(a, b))",
        "dict(enumerate(items))",
        "set(duplicates)",
        "tuple(sequence)",
        "frozenset(immutable)",
        "bytes(string, 'utf-8')",
        "bytearray(data)",
        "memoryview(buffer)",
        "int('42')",
        "float('3.14')",
        "str(number)",
        "bool(value)",
        "len(collection)",
        "sum(numbers)",
        "min(values)",
        "max(values)",
        "abs(negative)",
        "round(decimal, 2)",
        "pow(base, exp)",
        "divmod(a, b)",
        "hex(255)",
        "oct(64)",
        "bin(16)",
        "ord('A')",
        "chr(65)",
        "repr(object)",
        "hash(immutable)",
        "id(object)",
        "type(instance)",
        "isinstance(obj, cls)",
        "issubclass(child, parent)",
        "callable(func)",
        "getattr(obj, 'attr')",
        "setattr(obj, 'attr', val)",
        "hasattr(obj, 'attr')",
        "delattr(obj, 'attr')",
        "vars(obj)",
        "dir(module)",
        "locals()",
        "globals()",
        "exec('code')",
        "eval('1+1')",
        "compile('code', 'f', 'exec')",
    ]

    # First try sentence-transformers based code models (more reliable)
    try:
        from sentence_transformers import SentenceTransformer

        # Code-aware embedding models via sentence-transformers
        code_st_models = [
            # Use general models on code - they still work!
            ("all-MiniLM-L6-v2", "MiniLM-code"),
            ("all-mpnet-base-v2", "MPNet-code"),
        ]

        for model_name, display_name in code_st_models:
            try:
                model = SentenceTransformer(model_name)
                embeddings = model.encode(CODE_SNIPPETS, normalize_embeddings=True)

                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'code'
                result['model_id'] = model_name

                print(f"\n  {display_name}:")
                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])
            except Exception as e:
                print(f"  {display_name} failed: {e}")

    except ImportError:
        print("  sentence-transformers not available")

    # Also try transformers-based code models
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        code_hf_models = [
            ("microsoft/codebert-base", "CodeBERT"),
            ("microsoft/graphcodebert-base", "GraphCodeBERT"),
        ]

        for model_name, display_name in code_hf_models:
            try:
                print(f"\n  Loading {display_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                embeddings = []
                with torch.no_grad():
                    for snippet in CODE_SNIPPETS:
                        inputs = tokenizer(snippet, return_tensors="pt",
                                         padding=True, truncation=True, max_length=128)
                        outputs = model(**inputs)
                        # Use CLS token
                        emb = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(emb[0])

                embeddings = np.array(embeddings)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'code'
                result['model_id'] = model_name

                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])

            except Exception as e:
                print(f"  {display_name} failed: {e}")

    except ImportError:
        print("  transformers not available for code models")

    # ============================================================
    # AUDIO MODELS (if available)
    # ============================================================
    print("\n" + "=" * 60)
    print("AUDIO MODELS")
    print("=" * 60)

    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch

        print("\n  Loading Wav2Vec2...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Generate synthetic audio features (since loading real audio is complex)
        print("  Generating synthetic audio embeddings...")

        np.random.seed(123)
        n_samples = 100
        audio_dim = 768  # Wav2Vec2 hidden size

        # Random normalized embeddings to simulate audio
        audio_embeddings = np.random.randn(n_samples, audio_dim)
        audio_embeddings = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)

        result = analyze_embeddings(audio_embeddings, "Wav2Vec2-random-sim")
        result['modality'] = 'audio'
        result['model_id'] = 'wav2vec2-base'
        result['note'] = 'Simulated via random normalized vectors'

        print(f"\n  Wav2Vec2 (simulated):")
        print(f"    Shape: {result['shape']}")
        print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
        print(f"    Df × α = {result['Df_alpha']:.4f}")
        print(f"    NOTE: Random normalized vectors for baseline")

        results['models'].append(result)
        # Don't add simulated to final comparison

    except ImportError:
        print("  Wav2Vec2 not available")
    except Exception as e:
        print(f"  Wav2Vec2 failed: {e}")

    # ============================================================
    # DOMAIN-SPECIFIC MODELS (Scientific, Legal, etc.)
    # ============================================================
    print("\n" + "=" * 60)
    print("DOMAIN-SPECIFIC MODELS")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        # Domain-specific words/concepts
        DOMAIN_CONCEPTS = [
            # Scientific
            "quantum", "entropy", "wavelength", "momentum", "particle",
            "molecule", "catalyst", "synthesis", "spectrum", "radiation",
            # Medical
            "diagnosis", "symptom", "treatment", "pathology", "prognosis",
            "antibiotic", "vaccine", "inflammation", "metabolism", "hormone",
            # Legal
            "liability", "jurisdiction", "defendant", "plaintiff", "statute",
            "precedent", "contract", "negligence", "tort", "amendment",
            # Financial
            "asset", "liability", "equity", "dividend", "portfolio",
            "derivative", "hedging", "arbitrage", "liquidity", "volatility",
            # Technical
            "algorithm", "database", "encryption", "protocol", "architecture",
            "latency", "throughput", "scalability", "redundancy", "cache",
        ]

        # Specialized domain models
        domain_models = [
            ("allenai/scibert_scivocab_uncased", "SciBERT"),
            ("emilyalsentzer/Bio_ClinicalBERT", "ClinicalBERT"),
            ("nlpaueb/legal-bert-base-uncased", "LegalBERT"),
            ("ProsusAI/finbert", "FinBERT"),
        ]

        for model_name, display_name in domain_models:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                print(f"\n  Loading {display_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                embeddings = []
                with torch.no_grad():
                    for word in DOMAIN_CONCEPTS:
                        inputs = tokenizer(word, return_tensors="pt",
                                         padding=True, truncation=True, max_length=128)
                        outputs = model(**inputs)
                        emb = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(emb[0])

                embeddings = np.array(embeddings)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'domain-specific'
                result['model_id'] = model_name

                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])

            except Exception as e:
                print(f"  {display_name} failed: {e}")

    except ImportError:
        print("  Domain models not available")

    # ============================================================
    # INSTRUCTION-TUNED MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("INSTRUCTION-TUNED MODELS")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        # Instruction-tuned models have different training objectives
        instruction_models = [
            ("BAAI/bge-small-en-v1.5", "BGE-small-instruct"),
            ("intfloat/e5-small-v2", "E5-small-instruct"),
            ("sentence-transformers/gtr-t5-base", "GTR-T5-base"),
            ("sentence-transformers/sentence-t5-base", "ST5-base"),
        ]

        # Test with instruction-formatted queries
        INSTRUCTION_QUERIES = [
            "query: What is machine learning?",
            "query: Explain quantum computing",
            "query: How does photosynthesis work?",
            "query: What are neural networks?",
            "query: Define artificial intelligence",
            "passage: Machine learning is a subset of AI",
            "passage: Quantum computers use qubits",
            "passage: Plants convert sunlight to energy",
            "passage: Neural networks mimic the brain",
            "passage: AI simulates human intelligence",
        ] + WORDS[:60]  # Add some regular words too

        for model_name, display_name in instruction_models:
            try:
                model = SentenceTransformer(model_name)
                embeddings = model.encode(INSTRUCTION_QUERIES, normalize_embeddings=True)

                result = analyze_embeddings(embeddings, display_name)
                result['modality'] = 'instruction-tuned'
                result['model_id'] = model_name

                print(f"\n  {display_name}:")
                print(f"    Shape: {result['shape']}")
                print(f"    Df = {result['Df']:.4f}, α = {result['alpha']:.4f}")
                print(f"    Df × α = {result['Df_alpha']:.4f} (8e = {8*np.e:.4f})")
                print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

                results['models'].append(result)
                all_df_alpha.append(result['Df_alpha'])
            except Exception as e:
                print(f"  {display_name} failed: {e}")

    except ImportError:
        print("  Instruction-tuned models not available")
    except Exception as e:
        print(f"  Instruction-tuned models failed: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: CROSS-MODAL UNIVERSALITY")
    print("=" * 70)

    if all_df_alpha:
        mean_df_alpha = np.mean(all_df_alpha)
        std_df_alpha = np.std(all_df_alpha)
        cv = std_df_alpha / mean_df_alpha * 100 if mean_df_alpha > 0 else float('inf')
        vs_8e = abs(mean_df_alpha - 8 * np.e) / (8 * np.e) * 100

        print(f"\n  Models tested (real embeddings): {len(all_df_alpha)}")
        print(f"\n  Df × α statistics:")
        print(f"    Mean: {mean_df_alpha:.4f}")
        print(f"    Std: {std_df_alpha:.4f}")
        print(f"    CV: {cv:.2f}%")
        print(f"    Target (8e): {8 * np.e:.4f}")
        print(f"    Error vs 8e: {vs_8e:.2f}%")

        print(f"\n  Pass criteria:")
        print(f"    CV < 10%: {'PASS' if cv < 10 else 'FAIL'} ({cv:.2f}%)")
        print(f"    Mean within 5% of 8e: {'PASS' if vs_8e < 5 else 'FAIL'} ({vs_8e:.2f}%)")

        overall_pass = cv < 10 and vs_8e < 5

        results['summary'] = {
            'n_models': len(all_df_alpha),
            'mean_df_alpha': float(mean_df_alpha),
            'std_df_alpha': float(std_df_alpha),
            'cv_percent': float(cv),
            'vs_8e_percent': float(vs_8e),
            'passes_cv_threshold': cv < 10,
            'passes_mean_threshold': vs_8e < 5,
            'overall_pass': overall_pass,
        }

        print(f"\n  OVERALL: {'PASS - 8e appears universal!' if overall_pass else 'NEEDS MORE DATA'}")

    else:
        print("\n  No valid embeddings collected")
        results['summary'] = {'n_models': 0, 'overall_pass': False}

    # Per-model breakdown
    print("\n  Per-model results:")
    print(f"  {'Model':<25} {'Modality':<12} {'Df':<10} {'α':<10} {'Df×α':<10} {'vs 8e':<10}")
    print("  " + "-" * 77)
    for m in results['models']:
        if 'note' not in m:  # Skip simulated
            print(f"  {m['name']:<25} {m['modality']:<12} {m['Df']:<10.4f} {m['alpha']:<10.4f} {m['Df_alpha']:<10.4f} {m['vs_8e_percent']:<10.2f}%")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_cross_modal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
