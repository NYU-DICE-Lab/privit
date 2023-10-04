# PriViT: Vision Transformers for Fast Private Inference

This repository contains the code of our paper PriViT (arxiv link). Here we provide all the necessary files and instructions to replicate our training, testing, and benchmarking processes.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Ablation Studies](#ablation-studies)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Getting Started

This section will guide you through the basic steps to get the code running on your local machine for development and testing purposes. For detailed instructions about each script, please refer to the specific subsections below.

### Prerequisites

- Python 3.8+
- PyTorch
- Flax (for benchmarking)

Additional dependencies are listed in `requirements.txt`.

### Installation

```sh
git clone [your-repo-link]
cd [your-repo-name]
pip install -r requirements.txt
```

## Training
### Primary Training Scripts
- train.py: Use this script to train the model using our proposed method, PRIVIT.

```sh
python train.py --args
```
### Ablation Training Scripts
- train_without_kd.py: Training without knowledge distillation (kd).
- train_without_pretrain.py: Training without using pretrained checkpoints.

## Testing
- inference.py: Run this script to perform inference using a trained model.
```sh
python inference.py --model-path [path-to-trained-model] --test-data [path-to-test-data]
```

## Benchmarking
The folder benchmark/ contains all flax code used for benchmarking the performance of these PyTorch models.

- Navigate to the benchmark folder and follow additional instructions as needed:



## Contributing
If you would like to contribute to this repository, please follow these guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use our code in your research, please cite our paper: