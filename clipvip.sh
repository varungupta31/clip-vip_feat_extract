#!/bin/bash
#SBATCH -A varungupta
#SBATCH -c 9 
#SBATCH --gres=gpu:1
#SBATCH -w gnode037
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --mail-user gupta.varun@research.iiit.ac.in
#SBATCH --mail-type END

conda init --all
source activate clipvip
python feat_extract_msrvtt.py
