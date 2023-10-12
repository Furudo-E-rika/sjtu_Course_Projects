#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

source activate /lustre/home/acct-stu/stu168/miniconda3/envs/py37

# training
python train.py --config_file configs/audio_only.yaml

# evaluation
python evaluate.py --experiment_path experiments/audio_only
