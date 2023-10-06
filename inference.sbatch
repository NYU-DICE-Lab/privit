#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=privit
#SBATCH --output=jobname%j.out
module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate compression
python inference.py \
    --dataset dataset_name \
    --checkpoint /path/to/privit/checkpoint