import argparse
import torch
import torch.nn as nn
from utils import gelu_counting, softmax_counting
from parametric_masks import LearnableAlpha, LearnableBeta
from dataset import get_dataset
from train_utils import test
from privit_model import CustomViTForImageClassification
import transformers
from transformers import AutoConfig
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="VIT SNL Training")
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--checkpoint', type=str, help='checkpoint to test')
parser.add_argument('--threshold', default=1e-3, type=float)
args = parser.parse_args()

def main():
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    test_dataset = get_dataset(args.dataset, 'test', False)

    num_classes = len(test_dataset.classes)
    print(num_classes)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=8, num_workers=4, pin_memory=True)

    checkpoint = torch.load(args.checkpoint)
    config = AutoConfig.from_pretrained('WinKawaks/vit-tiny-patch16-224', num_labels=num_classes)

    model = CustomViTForImageClassification(config)
    for i in range(len(model.vit.encoder.layer)):
        model.vit.encoder.layer[i].intermediate.intermediate_act_fn = LearnableAlpha(config)
        model.vit.encoder.layer[i].attention.attention.betas = LearnableBeta(config)
    model.load_state_dict(checkpoint['state_dict'])
    gelu_count = gelu_counting(model,args)
    softmax_count = softmax_counting(model,args)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    test_loss, test_top1, test_top5 = test(test_loader, model, criterion, device, 100, display=False)
    print(f"Model has {gelu_count} Gelus and {softmax_count} Softmax operations")
    print(f"Accuracy over {args.dataset} is {test_top1}")

if __name__ == "__main__":
    main()