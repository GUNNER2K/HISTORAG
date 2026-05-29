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

---

## Repository Structure

```text
historag/
│
├── src/
│   ├── retrieval/
│   ├── evaluation/
│   ├── visualization/
│   ├── preprocessing/
│   └── utilities/
│
├── scripts/
│   ├── extraction/
│   ├── retrieval/
│   └── evaluation/
│
├── configs/
│   └── experiment yaml files
│
├── experiment_logs/
│   └── experiments.csv
│
├── figures/
│   ├── retrieval_examples/
│   ├── latency_plots/
│   └── embedding_visualizations/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The experiments were conducted on Whole Slide Images (WSIs) and associated annotations.

The repository does not contain the raw datasets because:

* WSIs are extremely large
* Embedding files can exceed GitHub storage limits
* Dataset redistribution may be restricted

Users should download datasets separately and update the corresponding paths in the configuration files.

---

## Foundation Models Evaluated

### UNI

Pathology foundation model used as the initial MVP encoder.

### Virchow

Large-scale pathology foundation model evaluated for embedding quality.

### CONCH

Pathology vision-language foundation model evaluated for retrieval performance.

---

## MVP Implementation

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

### Experiments

* Brute-force retrieval
* FAISS Flat
* FAISS IVF
* FAISS HNSW

### Evaluation Metrics

* Retrieval latency
* Precision@k
* Mean Average Precision (mAP)

---

## Hypothesis 2: Retrieval Quality Can Be Quantitatively Measured

The quality of histopathology patch retrieval can be objectively evaluated using quantitative retrieval metrics.

### Experiments

* Annotation processing
* Patch-to-polygon mapping
* Patch labeling
* Metric computation

### Evaluation Metrics

* Precision@5
* Precision@10
* Recall@5
* Recall@10
* Average Precision (AP)
* Mean Average Precision (mAP)

---

## Hypothesis 3: Foundation Models Capture Biological Structure Differently

Different pathology foundation models may produce embeddings with varying retrieval performance and representation quality.

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
git clone https://github.com/<username>/historag.git
cd historag
```

### Create Environment

```bash
conda create -n historag python=3.10
conda activate historag
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Feature Extraction

Example:

```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /path/to/wsi.svs \
    --job_dir /path/to/output \
    --patch_encoder uni_v1
```

For other models:

```bash
--patch_encoder virchow
```

```bash
--patch_encoder conch_v15
```

---

## Running Retrieval Experiments

Example:

```bash
python retrieval_experiment.py
```

Outputs:

* Retrieval visualizations
* Retrieval metrics
* Latency statistics
* Evaluation reports

---

## Reproducing Experiments

1. Download dataset
2. Extract image patches
3. Generate embeddings
4. Build retrieval index
5. Run retrieval evaluation
6. Compute metrics
7. Generate plots

---

## Hardware

Experiments were conducted using:

* NVIDIA A100 GPUs
* NVIDIA V100 GPUs
* TinyGPU Cluster
* Woody Storage Cluster

---

## Future Work

Potential future improvements include:

* Cross-slide retrieval
* Multi-slide retrieval benchmarking
* Vector database integration
* Interactive pathology atlas
* Multi-modal retrieval
* Retrieval-augmented pathology foundation models
* Clinical decision support applications

---

## Author

Satyaki Bhattacharjee

Master's Student in Data Science

Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

---

## Acknowledgements

* BIMAP Project
* Pattern Recognition Lab (FAU)
* Trident Framework
* UNI Foundation Model
* Virchow Foundation Model
* CONCH Foundation Model
* FAU TinyGPU Infrastructure
* FAU Woody Storage Infrastructure
