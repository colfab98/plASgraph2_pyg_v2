# Extending plASgraph2 with GRU-Enhanced Edge-Gated GGNNs for Plasmid Contig Classification

This repository contains the PyTorch Geometric implementation for the thesis: "Extending plASgraph2 with GRU-Enhanced Edge-Gated GGNNs for Plasmid Contig Classification".

This project reimplements and extends the original plASgraph2 to solve its core "information spill-over" problem. It introduces a GGNNModel (Gated Graph Neural Network) architecture that uses learned, feature-aware edge gates and GRU-based node updates to more accurately distinguish plasmid and chromosomal contigs in assembly graphs.

## Core Features

* **Advanced GNN Model**: Implements a `GGNNModel` (in `models.py`) alongside the baseline `GCNModel` for comparison. The GGNN uses edge gates and GRU updates to control information flow.
* **Modern Framework**: Migrates the original TensorFlow/Keras project to PyTorch Geometric (PyG) for greater flexibility.
* **Scalable Training**: Utilizes Hugging Face Accelerate for distributed multi-GPU training and PyG's NeighborLoader for scalable mini-batch training on large graphs.
* **Rich Features**: Integrates DNABERT-2 to generate sequence embeddings as edge features (see `data.py`).
* **Full Experiment Pipeline**: Provides a complete workflow:
    * `tune_hpo.py`: Hyperparameter optimization using Optuna.
    * `train.py`: Final k-fold ensemble model training.
    * `evaluate.py`: Evaluation of the trained ensemble on a test set.

## Datasets

The scripts are configured to run against the original plasgraph2-datasets (which must be cloned separately).

This repository also contains `plasgraph2-datasets_new`, which includes samples from the original dataset that were reproduced using the custom data pipeline described in the thesis (Section 2.2). This new dataset format supports additional edge features, such as read support counts.

## Installation and Usage

All information on setting up the conda environment, installing dependencies, and running the complete HPO, training, and evaluation pipelines is detailed in: `commands.txt`.