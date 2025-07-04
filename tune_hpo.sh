#!/bin/bash
#SBATCH --job-name=tune_hpo
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
python -u -m scripts.tune_hpo \
    --data_cache_dir cache/eskapee-train/ \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    output/ESKAPEE_hpo_study \
    > output/hpo.log 2> output/hpo.err
