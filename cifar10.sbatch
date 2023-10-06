#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:00
#SBATCH --mem=32GB
#SBATCH --job-name=privit
#SBATCH --output=jobname%j.out
module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate compression
python train.py \
    /path/to/output/directory \
    --dataset "cifar10" \
    --finetune_epochs 50 \
    --alpha_softmax 0.00003 \
    --alpha_gelu 0.00003 \
    --epochs 2000 \
    --batch 64 \
    --wd 0.0001 \
    --softmax_budget $1 \
    --gelu_budget $2 \
    --model_type 'WinKawaks/vit-tiny-patch16-224'