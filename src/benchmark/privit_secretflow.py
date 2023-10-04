import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
from flax_privit_model import CustomFlaxViTForImageClassification
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2
from flax_utils import accuracy, pytorch_to_flax, get_infer_cipher, print_named_params


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="2pc.json"
)
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dpath', default=None, type=str, help='Path to the dataset')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')
parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=1, type=int, metavar='N', help='batchsize (default: 64)')
args = parser.parse_args()
with open(args.config, 'r') as file:
    conf = json.load(file)
ppd.init(conf["nodes"], conf["devices"])

CKPT = args.checkpoint
checkpoint = torch.load(CKPT,map_location='cpu')
test_dataset = get_dataset(args.dataset, 'test',False,resize=224, dpath=args.dpath)
num_classes = len(test_dataset.classes)    
config = AutoConfig.from_pretrained('WinKawaks/vit-tiny-patch16-224', num_labels=num_classes)
alpha_infer, beta_infer = get_infer_cipher(checkpoint['state_dict'])

config.infer_alpha = alpha_infer
config.infer_beta = beta_infer

arr = jnp.sum(jnp.array(alpha_infer),axis=(-2,-1))
config.alpha_sizes = arr.astype(int).tolist()
arr1 = jnp.sum(jnp.array(beta_infer),axis=(-3,-2,-1))
config.beta_sizes = arr1.astype(int).tolist()
model = CustomFlaxViTForImageClassification(config)

print("model defined")
model = pytorch_to_flax(checkpoint['state_dict'], model)
model_test = CustomFlaxViTForImageClassification(config)
model_cipher = CustomFlaxViTForImageClassification(config=config)
print("Model Loaded")



def infer_cipher(inputs, params):
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs

def run_on_cpu(inputs, targets):
    outputs = model_test(pixel_values=inputs, params= model.params)["logits"]
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc


def run_on_spu(inputs, targets):
    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P2")(lambda x: x)(model.params)
    outputs = ppd.device("SPU")(infer_cipher)(inputs, params)
    outputs = ppd.get(outputs)
    outputs = outputs['logits']
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    print(acc)
    return acc

if __name__ == '__main__':
    print('\n------\nRun on CPU')
    torch.manual_seed(0)
    
    print(num_classes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    inputs, targets = next(iter(test_loader))
    acc_cpu = run_on_cpu(inputs.numpy(), targets)
    print('\n------\nRun on SPU')
    acc_spu = run_on_spu(inputs.numpy(),targets)