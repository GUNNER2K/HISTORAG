# HISTORAG: Retrieval-Augmented Histopathology Atlas

## Overview

HISTORAG is a retrieval-based histopathology image analysis project that explores how foundation model embeddings can be used to build a Retrieval-Augmented Generation (RAG)-inspired system for Whole Slide Images (WSIs).

Instead of retrieving text documents, HISTORAG retrieves visually similar tissue regions from large pathology slides. The project investigates whether modern pathology foundation models can learn meaningful representations of tissue morphology that enable efficient and biologically relevant image retrieval.

The project was developed as part of the BIMAP research project.

---

## Project Goals

The main objective of HISTORAG is to:

* Extract image patches from Whole Slide Images (WSIs)
* Encode patches into feature embeddings using pathology foundation models
* Store embeddings in a searchable vector space
* Retrieve visually similar tissue regions given a query patch
* Compare retrieval methods and foundation models
* Evaluate retrieval quality using quantitative metrics

---

## Motivation

Whole Slide Images are gigapixel-scale pathology scans that cannot be processed directly due to their enormous size.

Recent pathology foundation models such as UNI, Virchow, and CONCH provide powerful feature representations that capture tissue morphology and cellular structure.

Inspired by Retrieval-Augmented Generation (RAG) systems in Natural Language Processing, HISTORAG investigates whether similar retrieval-based techniques can be applied to histopathology images.

Potential applications include:

* Histopathology atlas creation
* Similar case retrieval
* Exploratory tissue analysis
* Content-based image search
* Foundation model evaluation

---

## Project Pipeline

1. Whole Slide Image (WSI) loading
2. Tissue segmentation
3. Patch extraction
4. Feature embedding generation
5. Vector database construction
6. Similarity search
7. Retrieval evaluation
8. Visualization and analysis

** The first 4 are not included for simplicity purposes
---

### Repository Structure

```text
HISTORAG/
│
├── demo_data/                 # Lightweight reproducible demo dataset
├── results/                   # Generated experiment outputs
├── mvp.py                      # Baseline MVP
├── phase_1.py              # Hypothesis 1: Retrieval latency comparison
├── phase_1_h2.py              # Hypothesis 2: Annotation-aware retrieval evaluation
├── phase_1_h3.py              # Hypothesis 3: Foundation model comparison
├── embeddings_2D_3D.py              # Hypothesis 4: Embedding-space visualization
├── alignment_verification.py  # Coordinate alignment verification
├── requirements.txt
└── README.md
```
---


## Dataset

The experiments were conducted on Whole Slide Images (WSIs) and associated annotations.

The repository does not contain the raw datasets because:

* WSIs are extremely large
* Embedding files can exceed GitHub storage limits
* Dataset redistribution may be restricted

For reproducibility purposes, lightweight demo data has been created and stored in [`demo_data/`](demo_data/).

* Original WSI (Primary tumor slides (CUP)): https://www.hancock.research.fau.eu/download
* Primary tumor annotations (slide number HE_484): https://www.hancock.research.fau.eu/download

---


## Foundation Models Evaluated

Links, installation guides, and usage details for all models can be found at:  

https://github.com/mahmoodlab/Trident?tab=readme-ov-file

The project evaluates multiple pathology foundation models for patch-level feature extraction and retrieval-based representation learning on Whole Slide Images (WSIs). All embeddings were extracted at 20x magnification using 256×256 tissue patches with 128 px overlap for dense patches.

---

### UNI and UNI-2

UNI and UNI-2 were used as the primary baseline pathology foundation models throughout the project.

* Vision Transformer (ViT)-based pathology encoders
* Used for:
  * patch feature extraction
  * retrieval benchmarking
  * annotation-aware retrieval evaluation
  * embedding-space visualization
* Generated high-dimensional semantic embeddings for each tissue patch
* Embedding dimensions:
  * UNI: 1024-dimensional feature vectors
  * UNI-2: 1536-dimensional feature vectors

UNI was used for the initial MVP retrieval system and for developing the retrieval evaluation pipeline.

---

### Virchow

Virchow is a large-scale pathology foundation model trained on extensive histopathology data.

* Evaluated for embedding quality and feature separability
* Used in comparative experiments against UNI and CONCH
* Strong semantic clustering observed in embedding-space visualization
* Generated:
  * 2560-dimensional feature embeddings

Virchow embeddings were evaluated using:
* brute-force retrieval
* FAISS-based approximate nearest neighbor retrieval
* t-SNE embedding-space visualization

---

### CONCH

CONCH is a pathology vision-language foundation model designed for multimodal histopathology representation learning.

* Evaluated for retrieval robustness and embedding separability
* Used in:
  * retrieval experiments
  * annotation-aware evaluation
  * embedding-space comparison
* Generated:
  * 512-dimensional feature embeddings

CONCH demonstrated competitive retrieval performance while maintaining a significantly smaller embedding dimensionality compared to larger pathology encoders.

---

## MVP Implementation 

* Code: [`mvp.py`](mvp.py).
* Output: [`results/mvp`](results/mvp).

The initial MVP consists of:

* Patch extraction using Trident
* UNI feature extraction
* Brute-force similarity search
* Top-k patch retrieval
* Visual inspection of retrieval results

This MVP serves as the foundation for all subsequent experiments.

---

# Experimental Hypotheses

## Hypothesis 1: Better Retrieval Methods Improve Similarity Search

Approximate Nearest Neighbor (ANN) retrieval methods such as FAISS-HNSW can significantly reduce retrieval latency compared to brute-force similarity search while maintaining retrieval quality.

* Code: [`phase_1.py`](phase_1.py).
* Output: [`results/h1`](results/h1).

### Experiments

* Brute-force retrieval
* FAISS Flat
* FAISS IVF
* FAISS HNSW

### Evaluation Metrics

* Retrieval latency

---

## Hypothesis 2: Retrieval Quality Can Be Quantitatively Measured

* Code: [`phase_1_h2.py`](phase_1_h2.py).
* Output: [`results/h2`](results/h2).

The quality of histopathology patch retrieval can be objectively evaluated using quantitative retrieval metrics.

### Experiments

* Annotation processing
* Patch-to-polygon mapping
* Patch labeling
* Metric computation

### Evaluation Metrics

* Precision@5
* Precision@10
* Mean Average Precision (mAP)

---

## Hypothesis 3 and 4: Foundation Models Capture Biological Structure Differently

Different pathology foundation models may produce embeddings with varying retrieval performance and representation quality.

* Code: [`phase_1_h3.py`](phase_1_h3.py) and [`embeddings_2D_3D`](embeddings_2D_3D)
* Output: [`results/h3`](results/h3) and [`results/h4`](results/h4)

### Experiments

* UNI retrieval evaluation
* Virchow retrieval evaluation
* CONCH retrieval evaluation
* Embedding space visualization

### Evaluation Metrics

* Precision@k
* mAP
* Retrieval latency
* PCA visualization
* t-SNE visualization
* UMAP visualization

---

## Retrieval Evaluation Metrics

### Precision@k

Measures the fraction of retrieved patches that are relevant among the top-k retrieved results.

Example:

If 8 of the top 10 retrieved patches belong to the same class:

Precision@10 = 0.8

### Recall@k

Measures how many relevant patches are retrieved relative to all relevant patches available in the dataset.

### Average Precision (AP)

Computes precision across all ranks where relevant items are retrieved.

### Mean Average Precision (mAP)

Average AP across all query patches.

This is the primary metric used to compare retrieval quality.

---


## Installation

### Clone Repository

```bash
git clone https://github.com/GUNNER2K/HISTORAG.git
cd HISTORAG
````

---

### Create Conda Environment

```bash
conda create -n historag python=3.10 -y
conda activate historag
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Install OpenSlide System Libraries

#### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install openslide-tools libopenslide-dev
```

#### Conda Alternative

```bash
conda install -c conda-forge openslide
```

---

### Verify Installation

```bash
python -c "import openslide; print('OpenSlide installed successfully')"
```

---


### Run Example Experiments

#### Hypothesis 1 — Retrieval Latency Comparison

```bash
python phase_1.py
```

---

#### Hypothesis 2 — Annotation-Aware Retrieval Evaluation

```bash
python phase_1_h2.py
```

---

#### Hypothesis 3 — Foundation Model Comparison

```bash
python phase_1_h3.py
```

---

#### Hypothesis 4 — Embedding Space Visualization

```bash
python embeddings_2D_3D.py
```

---

### Output Directory

All generated figures, plots, retrieval visualizations, and embedding-space analyses are stored inside:

```text
results/
```

---

## Author

Satyaki Bhattacharjee

Master's (M.Sc.) Student in Data Science

Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

---
