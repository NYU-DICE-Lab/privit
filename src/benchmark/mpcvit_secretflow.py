import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import json
import torch
from torch.utils.data import DataLoader
from flax_mpcvit_model import CustomFlaxViTForImageClassification
import flax
import jax.numpy as jnp
import numpy as np
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2
from flax_utils import print_named_params,accuracy, get_config, pytorch_to_flax_mpcvit, get_infer_cipher_mpc_vit
from dataset import get_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--config", default="2pc.json"
)
parser.add_argument(
    "-d", "--dataset", default="tiny_imagenet"
)
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')
args = parser.parse_args()
with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

config = get_config(args.dataset)


checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

alpha_infer, beta_infer = get_infer_cipher_mpc_vit(checkpoint['state_dict'])
# mpcvit will have an alpha parameter to control for the 
config.alpha_infer = alpha_infer
arr = jnp.sum(jnp.array(alpha_infer),axis=(-3,-2,-1))
config.alpha_sizes = arr.astype(int).tolist()

# mpcvit plus will have extra beta parameters
if len(beta_infer) == 0:
    config.beta_infer = None
else:
    config.beta_infer = beta_infer

model = CustomFlaxViTForImageClassification(config)

model = pytorch_to_flax_mpcvit(checkpoint['state_dict'], model)

model_test = CustomFlaxViTForImageClassification(config)
model_cipher = CustomFlaxViTForImageClassification(config=config)

def infer_cipher(inputs, params):
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs

def run_on_spu(inputs, targets):
    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P2")(lambda x: x)(model.params)
    outputs = ppd.device("SPU")(infer_cipher)(inputs, params)
    outputs = ppd.get(outputs)
    outputs = outputs['logits']
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc

def run_on_cpu(inputs, targets):
    outputs = model_test(pixel_values=inputs, params= model.params)["logits"].block_until_ready()
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc

if __name__ == '__main__':
    torch.manual_seed(0)
    resize = 64 if args.dataset == 'tiny_imagenet' else 32
    test_dataset = get_dataset(args.dataset,'test', False, resize)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    inputs, targets = next(iter(testloader))
    print('\n------\nRun on CPU')
    acc_cpu = run_on_cpu(inputs.numpy(), targets)
    print('\n------\nRun on SPU')
    acc_spu = run_on_spu(inputs.numpy(),targets)