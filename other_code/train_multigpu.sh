#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --time=04:00:00

PYTHONUNBUFFERED=1 accelerate launch --num_processes=2 --mixed_precision=fp16 -m scripts.train \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    output/ESKAPEE_hpo_study/best_arch_params.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_final_model/ \
    > output/train.log 2> output/train.err