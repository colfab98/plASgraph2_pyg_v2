#!/bin/bash
#SBATCH --job-name=tune_hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate plasgraph

export TOKENIZERS_PARALLELISM=false

export RUN_NAME="eskapee_v1"

mkdir -p runs/${RUN_NAME}

PYTHONUNBUFFERED=1 accelerate launch --num_processes=2 --mixed_precision=fp16 -m scripts.tune_hpo \
    --run_name ${RUN_NAME} \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    > runs/${RUN_NAME}/hpo_study/hpo.log 2> runs/${RUN_NAME}/hpo_study/hpo.err
