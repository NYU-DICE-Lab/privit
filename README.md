# PriViT: Vision Transformers for Fast Private Inference

This repository contains the code of our paper PriViT (arxiv link). Here we provide all the necessary files and instructions to replicate our training, testing, and benchmarking processes.

## Table of Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Benchmarking](#benchmarking)
- [Ablation Studies](#ablation-studies)
- [Citation](#citation)


## Getting Started

This section will guide you through the basic steps to get the code running on your local machine for development and testing purposes. For detailed instructions about each script, please refer to the specific subsections below.

### Installation
You would need to create two separate environments one for training privit models, and another for benchmarking the trained models on secretflow using SEMI2K.
We have exported the environment so you can simply create one by loading the yml files provided in the src. privit_training_environmnet.yml has packages for training environment and spu-jax.yml has packages for the benchmarking environment.

```sh
git clone [repository]
cd privit
cd src
conda env create -n compression --file privit_training_environmnet.yml

```

## Training
### Primary Training Scripts
You can find the training script in the slurm script files. Ensure that correct datapath is updated in the dataset.py folder.
- tinyimagenet.sbatch: This is the slurm script to train privit model on Tiny Imagenet dataset
- cifar100.sbatch: This is the slurm script to train privit model on Cifar 100 dataset.
- cifar10.sbatch: This is the slurm script to train privit model on Cifar 10 dataset.
- train.py: This script has the primary training logic of PriViT

We have also released model checkpoints [here](https://drive.google.com/drive/folders/16cn2JwPNSwy5j-FWm9tdwoZzYmc7SDyG?usp=sharing).


## Inference
You can find the inference script in inference.sbatch file, primary inference logic is in inference.py.

## Benchmarking
The folder benchmark/ contains all flax code used for benchmarking the performance of these PyTorch models using secretflow framework on SEMI2k protocol. For detailed instructions on how to setup a benchmarking setup using secretflow, refer their [documentation](https://github.com/secretflow/spu/tree/atc23_ae#usenix-atc-23-artifact-evaluation).
2pc.json is the configuration file, update the IP address of the two nodes in this file.
Start the server on two nodes using this:
### Server 1
```sh
conda activate spu-jax
python nodectl.py -c 2pc.json start --node_id node:0 &> node0.log &
```
### Server 2
```sh
conda activate spu-jax
python nodectl.py -c 2pc.json start --node_id node:1 &> node1.log &
```

To benchmark the privit model run on node 0
```sh
python privit_secretflow.py --config 2pc.json --checkpoint "/path/to/checkpoint" --dataset tiny_imagenet (or cifar10 or cifar100)
```
To benchmark the mpcvit model run on node 0
```sh
python mpcvit_secretflow.py --config 2pc.json --checkpoint "/path/to/checkpoint" --dataset tiny_imagenet (or cifar10 or cifar100)
```
These scripts load the pytorch checkpoints of privit/mpcvit and converts them to be compatible with flax.

## Ablation studies
Ablation studies are performed using this script
- train_without_kd.py: Training without knowledge distillation (kd).
- train_without_pretrain.py: Training without using pretrained checkpoints.

## Citation
If you find our work helpful to your research, please cite our paper:
