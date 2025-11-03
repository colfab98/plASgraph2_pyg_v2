#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1    
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

export RUN_NAME="run_13" 

mkdir -p runs/${RUN_NAME}
mkdir -p runs/${RUN_NAME}/final_model


PYTHONUNBUFFERED=1 accelerate launch --num_processes=4 --mixed_precision=fp16 -m scripts.train \
    --run_name ${RUN_NAME} \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    > runs/${RUN_NAME}/final_model/train.log 2> runs/${RUN_NAME}/final_model/train.err


# PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 --mixed_precision=fp16 -m scripts.train \
#     --run_name ${RUN_NAME} \
#     plasgraph_config.yaml \
#     plasgraph2-datasets_new/eskapee-train.csv \
#     plasgraph2-datasets_new/ \
#     > runs/${RUN_NAME}/final_model/train.log 2> runs/${RUN_NAME}/final_model/train.err