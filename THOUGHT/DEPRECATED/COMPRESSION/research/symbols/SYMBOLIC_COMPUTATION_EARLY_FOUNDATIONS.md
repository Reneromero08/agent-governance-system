# Symbolic Computation and Early Foundations

Early AI (“Good Old-Fashioned AI”) was built on symbolic rule‐based systems and logic. Pioneering ideas include Newell & Simon’s Physical Symbol System hypothesis (1976), Minsky’s frames, semantic networks (e.g. Quillian), and Hofstadter’s Gödel, Escher, Bach (1979) exploring self‐reference and formal rules in cognition. These works emphasize compositional, discrete representations of meaning. However, pure symbolic systems struggled with brittleness and scalability, spurring the rise of connectionist models. This legacy underpins modern work on integrating symbols with sub-symbolic methods[1].

## Neuro–Symbolic Integration

Neuro-symbolic AI combines neural networks with symbolic reasoning to gain both flexibility and interpretability. Recent reviews survey logic‐tensor networks, differentiable logic programming, neural theorem provers, etc. For example, Nawaz et al. (2025) review frameworks (LogicTensorNetwork, DeepProbLog, Scallop, DomiKnowS) that merge deep learning with logic or probabilistic reasoning[2][3]. Sinha et al. (2025) compare NeSy frameworks and note their focus on explainable integration of “neural representations and learning with symbolic representations and reasoning”[4]. IBM’s Neuro-Vector-Symbolic Architecture (NeuroVSA) is a notable project: it uses high-dimensional vector operations to represent symbolic facts and enables analogical reasoning. As IBM describes, NeuroVSA “combines strengths of symbolic AI and neural networks” by encoding data in hypervectors and applying algebraic operations for inference[5]. These efforts aim to let LLMs or other networks call on explicit logic or knowledge modules for compositional problem-solving.

## Semantic Compression and Context Extension

“Semantic compression” refers to lossy data compression that preserves meaning. Can (2025) formulates semantic compression in a continuous embedding space and shows there are phase transitions between lossless and lossy regimes[6][7]. In practice, researchers use LLMs themselves to compress inputs. Gilbert et al. (2023) evaluate GPT-3.5/4 on semantic text compression: they introduce metrics (Exact and Semantic Reconstruction Effectiveness) and find GPT-4 can compress text and decompress it with most meaning intact, effectively extending the input by ~5×[8]. Similarly, Fei et al. (ACL 2024) propose a semantic coder that “reduces the semantic redundancy” of long inputs so an LLM can process text 6–8× longer than normally possible[9][10]. These methods draw on information‐theoretic source coding ideas, using one model to pre-process (“telegraphic compression”) before a main LLM. Overall, recent work demonstrates that by focusing on preserving meaning rather than exact text, one can greatly expand effective context windows and offload details into compressed latent form[8][9].

## Logographic and Compositional Writing Systems

Researchers also study how symbolic information is compressed in human writing. For instance, Jiang et al. (2024, CogSci) develop a “library learning” model that discovers compositional patterns in Chinese characters. Their system learns reusable sub-programs (like radicals) that combine to form logographic characters, effectively compressing the writing system into an efficient, hierarchical representation[11][12]. They show the model recovers known orthographic structures and predicts how Chinese evolves under pressure for simplicity[13]. A follow-up (Jiang et al., 2025) adds multi-modal semantics: by jointly compressing the written form and spoken sounds of characters, the model uncovers systematic form–sound correspondences (akin to phonetic radicals) and links character parts to meanings[12]. In summary, these works treat written symbols as programs: they compress characters into reusable parts so that the meaning of unseen characters can be inferred compositionally[12][13]. (Hofstadter’s GEB similarly emphasizes self-referential systems and compositional structure, inspiring this line of work.)

## Hyperdimensional Computing and Vector Symbolic Architectures (VSA)

Hyperdimensional Computing (also “Vector Symbolic Architecture”) encodes symbols as random high-dimensional vectors (hypervectors) and manipulates them algebraically. Kanerva (2009) pioneered HDC for cognitive tasks; recent surveys summarize its evolution. For example, Liu et al. (IEEE 2022) review how VSAs support all data structures relevant to modern computing – numbers, symbols, bindings – via operations like superposition, binding, and permutation[14]. These models yield memory‐efficient, robust representations. A notable VSA-based project is IBM’s NeuroVSA, mentioned above. Another is Poduval et al. (2022): their GrapHD framework encodes entire graphs into a single hypervector for brain-like reasoning[15]. GrapHD’s graph encoder bundles node and edge information holistically so that one can later reconstruct node neighborhoods, match subgraphs, or find shortest paths directly in vector space[15]. The overview is shown in Figure 1 below:

Figure: GrapHD overview – a graph (A) is encoded into one high-dimensional vector, enabling cognitive operations like graph retrieval and matching (B,C)[16][15].

These hyperdimensional methods provide inspiration for symbolic vector databases and concept compression: they show how complex structured data (graphs, symbols) can be compressed into fixed-size vectors while still supporting logical queries or similarities[16][15].

## Graph-Based Symbolic Representations

Structured knowledge is often represented as graphs. Knowledge graphs (KGs) explicitly encode entities and relations (triples) and have a long tradition (WordNet, ConceptNet, Freebase, etc.). Recent work integrates KGs with neural models for reasoning. For instance, Liu et al. (2024) survey neural-symbolic reasoning over knowledge graphs, noting that KGs are “discrete symbolic representations” that are naturally combined with neural nets to handle noise, while symbolic inference on them provides interpretability[17]. In practical applications, engineers build graph-based symbolic layers. Follmann et al. (2023) introduce the nuScenes Knowledge Graph: a detailed ontological representation of a driving scene (lanes, lights, vehicles, trajectories) as a scene graph[18][19]. They show that this “graph-based, symbolic representation” encodes spatial and semantic relations that can then be consumed by graph neural networks for trajectory prediction[18][19]. In general, graph-based symbolic databases (triple stores, RDF, scene graphs) complement vector databases by capturing explicit relations. Hybrid neuro-symbolic systems often query such graphs to retrieve concept-level facts or perform logical inference in service of LLM-based reasoning[17][18].

## Token-Level Compression and Concept Models

Recent work aims to make language models more token-efficient by decomposing or compressing their vocabulary. Kavin et al. (2025) propose Aggregate Semantic Grouping (ASG): they apply product quantization to split each token embedding into sub-vectors (concept vectors) shared across words[20]. ASG represents each word as a sequence of concept‐IDs. This compresses the embedding matrix by ~99.5% while retaining about 95% of the original performance[20]. The key insight is that words share semantic “building blocks” (concepts), so representing them compositionally allows dramatic compression with little loss[20].

Likewise, Meta’s Large Concept Models (LCM) (Duquenne et al., 2024) build on high-level embeddings. They use SONAR sentence embeddings (200+ languages) as concepts, then train a Transformer to predict one sentence embedding from previous ones, rather than tokens[21][22]. In effect, the model reasons at the sentence-concept level. Early results show LCMs can zero-shot generalize summaries and other tasks better than same-sized token-based LMs[21][22].

Finally, many works on context extension and retrieval touch on concept compression. By offloading details into compressed representations (e.g. RAG with an external symbolic memory), models effectively achieve concept-level caching. Overall, these lines of research aim to let models think in concept-space rather than token-space – echoing older visions of a high-level language of thought – which could underpin future symbolic vector databases and compositional reasoning engines.

Sources: Key references include the IEEE/PMC survey on Vector Symbolic Architectures[14], reviews of neuro-symbolic AI[2][3], graph-reasoning surveys[17], and recent ML papers on semantic compression and concept embeddings[6][8][9][21][20]. Each cited work provides detailed methods and evaluations.

[1] [3] [4] Neuro-Symbolic Frameworks: Conceptual Characterization and Empirical Comparative Analysis

https://arxiv.org/pdf/2509.07122

[2] A review of neuro-symbolic AI integrating reasoning and learning for advanced cognitive systems - ScienceDirect

https://www.sciencedirect.com/science/article/pii/S2667305325000675

[5] Neuro-Vector-Symbolic Architecture - IBM Research

https://research.ibm.com/projects/neuro-vector-symbolic-architecture

[6] [7] Statistical Mechanics of Semantic Compression

https://arxiv.org/html/2503.00612v1

[8] [2304.12512] Semantic Compression With Large Language Models

https://arxiv.org/abs/2304.12512

[9] [10] Extending Context Window of Large Language Models via Semantic Compression - ACL Anthology

https://aclanthology.org/2024.findings-acl.306/

[11] [13] [2405.06906] Finding structure in logographic writing with library learning

https://arxiv.org/abs/2405.06906

[12] jiang.gy

https://jiang.gy/assets/pdf/jiang2025grapheme.pdf

[14]  Vector Symbolic Architectures as a Computing Framework for Emerging Hardware - PMC

https://pmc.ncbi.nlm.nih.gov/articles/PMC10588678/

[15] [16] Frontiers | GrapHD: Graph-Based Hyperdimensional Memorization for Brain-Like Cognitive Learning

https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.757125/full

[17] Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective

https://arxiv.org/html/2412.10390v1

[18] [19] nuScenes Knowledge Graph - A comprehensive semantic representation of traffic scenes for trajectory prediction

https://arxiv.org/html/2312.09676v1

[20] Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics

https://arxiv.org/html/2509.17737v2

[21] [22] [2412.08821] Large Concept Models: Language Modeling in a Sentence Representation Space

https://arxiv.org/abs/2412.08821