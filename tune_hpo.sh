#!/bin/bash
#SBATCH --job-name=tune_hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate plasgraph

export TOKENIZERS_PARALLELISM=false

export RUN_NAME="eskapee_v3"

mkdir -p runs/${RUN_NAME}

mkdir -p runs/${RUN_NAME}/hpo_study/

PYTHONUNBUFFERED=1 accelerate launch --num_processes=2 --mixed_precision=fp16 -m scripts.tune_hpo \
    --run_name ${RUN_NAME} \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    > runs/${RUN_NAME}/hpo_study/hpo.log 2> runs/${RUN_NAME}/hpo_study/hpo.err
