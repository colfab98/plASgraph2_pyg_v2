#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export TOKENIZERS_PARALLELISM=false

export RUN_NAME="run_18"

mkdir -p runs/${RUN_NAME}/evaluation_results

# Original plasgraph2 dataset
PYTHONUNBUFFERED=1 accelerate launch --num_processes=4 --mixed_precision=fp16 -m scripts.evaluate \
    --run_name ${RUN_NAME} \
    plasgraph2-datasets/eskapee-test.csv \
    plasgraph2-datasets/ \
    > runs/${RUN_NAME}/evaluation_results/eval.log 2> runs/${RUN_NAME}/evaluation_results/eval.err

# Filtered dataset
# PYTHONUNBUFFERED=1 accelerate launch --num_processes=4 --mixed_precision=fp16 -m scripts.evaluate \
#     --run_name ${RUN_NAME} \
#     plasgraph2-datasets/eskapee-test_filtered.csv \
#     plasgraph2-datasets/ \
#     > runs/${RUN_NAME}/evaluation_results/eval.log 2> runs/${RUN_NAME}/evaluation_results/eval.err

# My new dataset
# PYTHONUNBUFFERED=1 accelerate launch --num_processes=4 --mixed_precision=fp16 -m scripts.evaluate \
#     --run_name ${RUN_NAME} \
#     plasgraph2-datasets_new/eskapee-test.csv \
#     plasgraph2-datasets_new/ \
#     > runs/${RUN_NAME}/evaluation_results/eval.log 2> runs/${RUN_NAME}/evaluation_results/eval.err
