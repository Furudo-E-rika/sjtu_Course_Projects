#!/bin/bash

#SBATCH --job-name=512
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log

source activate /lustre/home/acct-stu/stu168/miniconda3/envs/py37


python run.py train_evaluate configs/hidden_256.yaml data/eval/feature.csv data/eval/label.csv

