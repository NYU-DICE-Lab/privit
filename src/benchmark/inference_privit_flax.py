import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
from transformers import AutoConfig
from flax_utils import print_named_params, pytorch_to_flax, get_infer_cipher, accuracy
from flax_privit_model import CustomFlaxViTForImageClassification
import jax.numpy as jnp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dpath', default=None, type=str, help='Path to the dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=8, type=int, metavar='N', help='batchsize (default: 64)')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint')
args = parser.parse_args()

test_dataset = get_dataset(args.dataset, 'test', args.dpath)
num_classes = len(test_dataset.classes)
CKPT = args.checkpoint
checkpoint = torch.load(CKPT,map_location='cpu')    
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

print("Model Loaded")


def infer_cipher(inputs, params):
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs

def run_on_cpu(inputs, targets):  
    outputs = model(pixel_values=inputs, params= model.params)["logits"]
    acc = accuracy(jnp.argmax(outputs,axis=1),targets.numpy())
    return acc


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    torch.manual_seed(0)
    
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    correct = 0
    count = 0
    for i , (inputs, targets) in enumerate(test_loader):
        acc_cpu = run_on_cpu(inputs.numpy(), targets)
        correct+=acc_cpu*targets.shape[-1]
        count+=targets.shape[-1]
        if i%64 ==0:
            print(f"minibatch {i} Accuracy: {correct/count} ")
    print(f"Final Accuracy is {correct/count}")
