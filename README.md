# Extending plASgraph2 with GRU-Enhanced Edge-Gated GGNNs for Plasmid Contig Classification

[cite_start]This repository contains the PyTorch Geometric implementation for the thesis: "Extending plASgraph2 with GRU-Enhanced Edge-Gated GGNNs for Plasmid Contig Classification"[cite: 1].

[cite_start]This project reimplements and extends the original plASgraph2 to solve its core "information spill-over" problem[cite: 13]. [cite_start]It introduces a GGNNModel (Gated Graph Neural Network) architecture that uses learned, feature-aware edge gates and GRU-based node updates to more accurately distinguish plasmid and chromosomal contigs in assembly graphs[cite: 14].

## Core Features

* [cite_start]**Advanced GNN Model**: Implements a `GGNNModel` (in `models.py`) alongside the baseline `GCNModel` for comparison[cite: 413]. [cite_start]The GGNN uses edge gates and GRU updates to control information flow[cite: 14].
* [cite_start]**Modern Framework**: Migrates the original TensorFlow/Keras project to PyTorch Geometric (PyG) for greater flexibility[cite: 15].
* [cite_start]**Scalable Training**: Utilizes Hugging Face Accelerate for distributed multi-GPU training and PyG's NeighborLoader for scalable mini-batch training on large graphs[cite: 15].
* [cite_start]**Rich Features**: Integrates DNABERT-2 to generate sequence embeddings as edge features (see `data.py`)[cite: 143].
* **Full Experiment Pipeline**: Provides a complete workflow:
    * [cite_start]`tune_hpo.py`: Hyperparameter optimization using Optuna[cite: 446].
    * [cite_start]`train.py`: Final k-fold ensemble model training[cite: 487].
    * [cite_start]`evaluate.py`: Evaluation of the trained ensemble on a test set[cite: 409].

## Datasets

[cite_start]The scripts are configured to run against the original plasgraph2-datasets (which must be cloned separately)[cite: 169].

[cite_start]This repository also contains `plasgraph2-datasets_new`, which includes samples from the original dataset that were reproduced using the custom data pipeline described in the thesis (Section 2.2)[cite: 164]. [cite_start]This new dataset format supports additional edge features, such as read support counts[cite: 165].

## Installation and Usage

All information on setting up the conda environment, installing dependencies, and running the complete HPO, training, and evaluation pipelines is detailed in: `commands.txt`.