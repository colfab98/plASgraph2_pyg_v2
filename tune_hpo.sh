#!/bin/bash
#SBATCH --job-name=tune_hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --time=12:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate plasgrpah

export TOKENIZERS_PARALLELISM=false

PYTHONUNBUFFERED=1 accelerate launch --num_processes=2 --mixed_precision=fp16 -m scripts.tune_hpo \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train_small.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_hpo_study \
    > output/hpo.log 2> output/hpo.err
