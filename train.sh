#!/bin/bash
#SBATCH --job-name=train-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-task=1    
#SBATCH --time=06:00:00

export RUN_NAME="eskapee_v1" 

mkdir -p runs/${RUN_NAME}

PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 --mixed_precision=fp16 -m scripts.train \
    --run_name ${RUN_NAME} \
    plasgraph_config.yaml \
    plasgraph2-datasets/eskapee-train.csv \
    plasgraph2-datasets/ \
    --training_mode k-fold \
    > runs/${RUN_NAME}/train.log 2> runs/${RUN_NAME}/train.err
