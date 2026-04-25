# Datasets

This file lists preferred external datasets for the ML program.

## Wave 1: Practical Text Datasets

1. `glue/sst2`
   - binary sentiment classification
   - good for pure-vs-mixed cluster tests
2. `ag_news`
   - 4-class topic classification
   - stronger semantic cluster separation
3. `snli`
   - 3-way natural language inference
   - useful for semantic purity and contradiction structure
4. `glue/stsb`
   - semantic similarity regression
   - useful for pair-level geometry checks

## Wave 2: Representation Transfer

- MTEB subsets
- BEIR subsets
- FEVER / climate-fever for factuality-adjacent structure

## Wave 3: Non-Text

- CIFAR-10 / CIFAR-100 via pretrained encoders
- ImageNet subsets via CLIP / DINOv2
- ESC-50 or similar for audio embeddings

## Rule

Claims require public external data. Synthetic data is allowed only for
development, debugging, and sanity checks.
