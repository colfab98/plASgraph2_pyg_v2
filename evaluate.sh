#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export TOKENIZERS_PARALLELISM=false

export RUN_NAME="test_new_dt"

mkdir -p runs/${RUN_NAME}/evaluation_results

PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 --mixed_precision=fp16 -m scripts.evaluate \
    --run_name ${RUN_NAME} \
    plasgraph2-datasets_new/eskapee-test.csv \
    plasgraph2-datasets_new/ \
    > runs/${RUN_NAME}/evaluation_results/eval.log 2> runs/${RUN_NAME}/evaluation_results/eval.err
