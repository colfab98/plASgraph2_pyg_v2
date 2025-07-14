#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=04:00:00

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
# export TOKENIZERS_PARALLELISM=false && \

export TOKENIZERS_PARALLELISM=false


PYTHONUNBUFFERED=1 accelerate launch --num_processes=4 --mixed_precision=fp16 -m scripts.tune_hpo \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_hpo_study \
    > output/hpo.log 2> output/hpo.err
