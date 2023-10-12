#!/bin/bash

#SBATCH --job-name=VAE_training
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=./output/%j.log
#SBATCH --error=./error/%j.log

source activate /lustre/home/acct-stu/stu170/env

cd /lustre/home/acct-stu/stu146/Variational_AutoEncoders || exit

python train.py
python visualization.py