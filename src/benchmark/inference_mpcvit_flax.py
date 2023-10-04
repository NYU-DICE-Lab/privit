import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import torch
import jax.numpy as jnp
from dataloader_tinyimagenet import TinyImageNet
from flax_mpcvit_model import CustomFlaxViTForImageClassification
from flax_utils import print_named_params, map_qkv_weights_to_flax, pytorch_to_flax_mpcvit, get_config, get_infer_cipher_mpc_vit
from dataset import get_dataset

parser = argparse.ArgumentParser(description='distributed driver.')

parser.add_argument(
    "-d", "--dataset", default="tiny_imagenet"
)
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')
args = parser.parse_args()

config = get_config(args.dataset)
checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
alpha_infer, beta_infer = get_infer_cipher_mpc_vit(checkpoint['state_dict'])

config.alpha_infer = alpha_infer
arr = jnp.sum(jnp.array(alpha_infer),axis=(-3,-2,-1))
config.alpha_sizes = arr.astype(int).tolist()
if len(beta_infer) == 0:
    config.beta_infer = None
else:
    config.beta_infer = beta_infer

model = CustomFlaxViTForImageClassification(config)
model = pytorch_to_flax_mpcvit(checkpoint['state_dict'], model)


def accuracy(predictions, labels):
    if predictions.shape != labels.shape:
        raise ValueError("Predictions")

    correct_predictions = jnp.sum(predictions == labels)
    total_predictions = predictions.size

    return correct_predictions / total_predictions, correct_predictions, total_predictions




def run_on_cpu(inputs, targets):

    outputs = model(pixel_values=inputs, params= model.params)["logits"].block_until_ready()

    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    return acc

if __name__ == '__main__':
    print('\n------\nRun on CPU')
    torch.manual_seed(0)
    resize = 64 if args.dataset == 'tiny_imagenet' else 32
    test_dataset = get_dataset(args.dataset,'test', False, resize)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=1)
    correct_tot = 0
    total_total = 0
    for i, (inputs, targets) in enumerate(testloader):
        acc_cpu, correct, total = run_on_cpu(inputs.numpy(), targets)
        correct_tot+=correct
        total_total+=total
        if i%64 ==0:
            print(f"minibatch {i} Accuracy: {correct_tot/total_total} ")
    print(f"Final Accuracy is {correct_tot/total_total}")