#!/bin/bash

#SBATCH --job-name=LanguageModel
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=./output/RNN/%j_out.log
#SBATCH --error=./output/RNN/%j_err.log

source activate /lustre/home/acct-stu/stu170/env

cd /lustre/home/acct-stu/stu146/languagemodel || exit


python main.py --cuda --epochs 50 --model LSTM --emsize 1000 --nhid 1000 --lr 20 --tied

