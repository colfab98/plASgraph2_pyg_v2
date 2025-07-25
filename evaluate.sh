#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

export TOKENIZERS_PARALLELISM=false

PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 --mixed_precision=fp16 -m scripts.evaluate \
    output/ESKAPEE_final_model/ \
    plasgraph2-datasets/eskapee-test.csv \
    plasgraph2-datasets/ \
    output/evaluation_results/ \
    > output/evaluation_results/eval.log 2> output/evaluation_results/eval.err