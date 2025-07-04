#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=04:00:00

PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 --mixed_precision=fp16 -m scripts.train \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    output/ESKAPEE_hpo_study/custom_params.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_final_model/ \
    > output/train.log 2> output/train.err
