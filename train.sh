#!/bin/bash
#SBATCH --job-name=train-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # A single task
#SBATCH --gpus-per-task=1    # Request one GPU for the task
#SBATCH --time=04:00:00

PYTHONUNBUFFERED=1 python -m scripts.train \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    output/ESKAPEE_hpo_study/best_arch_params.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_final_model/ \
    > output/train.log 2> output/train.err