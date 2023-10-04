# Applying SNL on Vision Transformer.

import argparse
from calendar import c
import os
import time
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log, AverageMeter, accuracy, gelu_counting, softmax_counting
from parametric_masks import LearnableAlpha, LearnableBeta
from dataset import get_dataset
from train_utils import train, test, mask_train_kd
# from transformers import ViTForImageClassification
from privit_model import CustomViTForImageClassification
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision import transforms, datasets
from torchvision.models import resnet18


parser = argparse.ArgumentParser(description="VIT SNL Training")
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch', default=64, type=int, metavar='N',
                    help='batchsize (default: 64)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate',default=1e-4, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--finetune_lr', default=1e-4,type=float, 
                    help='fintune learning rate parameter')
parser.add_argument('--alpha_softmax', default=1e-5, type=float,
                    help='Lasso coefficient for Softmax')
parser.add_argument('--alpha_gelu', default=1e-5, type=float,
                    help='Lasso coefficient for Gelus')
parser.add_argument('--threshold', default=1e-3, type=float)
parser.add_argument('--gelu_budget', default=0, type=int)
parser.add_argument('--softmax_budget', default=0, type=int)

parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--skip', action='store_true', help='debug mode')
parser.add_argument('--skip_softmax', action='store_true', help='debug mode')
parser.add_argument('--skip_gelu', action='store_true', help='debug mode')
parser.add_argument('--model_type', '-mt', help='ViT model to use', default='google/vit-base-patch16-224-in21k', choices=['google/vit-base-patch16-224-in21k', 'google/vit-large-patch16-224-in21k','WinKawaks/vit-small-patch16-224','WinKawaks/vit-tiny-patch16-224','mpc_vit'])
parser.add_argument('--model_student', help='ViT model to use')
parser.add_argument('--soft_interval', type=int, default=200,
                    help='How much desired reduction in softmax each epoch.')
parser.add_argument('--gelu_interval', type=int, default=2,
                    help='How much desired reduction in gelus each epoch.')
parser.add_argument('--save_every', type=int, default=32,
                    help='Frequency to save NAS model checkpoints')
# Aug params
parser.add_argument('--aa', action='store_true', default=False, help='Enable RandAugment')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of applying CutMix aug strategy')



args = parser.parse_args()

def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    logfilename = os.path.join(args.outdir, args.logname)

    # Printing the hyperparameters here.
    log(logfilename, "Hyperparameter List")
    log(logfilename, "Finetune Epochs: {:}".format(args.finetune_epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha Softmax: {:}".format(args.alpha_softmax))
    log(logfilename, "Alpha Gelu: {:}".format(args.alpha_gelu))
    log(logfilename, "GeLU Budget: {:}".format(args.gelu_budget))
    log(logfilename, "Softmax Budget: {:}".format(args.softmax_budget))
    log(logfilename, "Performance longer finetune and aggresive penalty with early binarize")

    print(args)

    train_dataset = get_dataset(args.dataset, 'train', args.aa,resize=224)
    test_dataset = get_dataset(args.dataset, 'test', False,resize=224)

    num_classes = len(train_dataset.classes)
    print(num_classes)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=True)

    # defining the teacher (fintuning from the imagenet pre-trained model)

    model = CustomViTForImageClassification.from_pretrained(args.model_type,num_labels=num_classes,ignore_mismatched_sizes=True)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    log(logfilename, "Finetuning the ImageNet pretrained models to {} dataset (TEACHER)".format(args.dataset))
    best_top1 = 0

    finetune_epoch = 1 if args.debug else 10
    if not args.skip:
    # Fintuning the teacher.
        for epoch in range(finetune_epoch):
            train_loss, train_top1, train_top5 = train(train_loader, model, None, optimizer, criterion, epoch, device, args.cutmix_prob)
            lr_scheduler.step()

            test_loss, test_top1, test_top5 = test(test_loader, model, criterion, device, display=False)

            if best_top1 < test_top1:
                best_top1 = test_top1
                is_best = True
            else:
                is_best = False
            
            if is_best:
                torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, 'img_finetune_best_checkpoint.pth.tar'))
    
        log(logfilename, "ImageNet Finetune best test Prec@1 = {}%".format(best_top1))
    if args.model_student == 'mpc_vit':
        model_config = AutoConfig.from_pretrained(args.model_type)
        model_config.num_hidden_layers = 9
        model_config.num_attention_heads = 12
        model_config.intermediate_size = 384
        model_config.hidden_size = 192
        model_config.image_size = 64
        model_config.patch_size = 4
        model_config.num_labels = 200 # tiny imagenet
    elif args.model_student == 'cifar100':
        model_config = AutoConfig.from_pretrained(args.model_type)
        model_config.num_hidden_layers = 7
        model_config.num_attention_heads = 4
        model_config.intermediate_size = 512
        model_config.hidden_size = 256
        model_config.image_size = 32
        model_config.patch_size = 4
        model_config.num_labels = 100 
    elif args.model_student == 'cifar10':
        model_config = AutoConfig.from_pretrained(args.model_type)
        model_config.num_hidden_layers = 7
        model_config.num_attention_heads = 4
        model_config.intermediate_size = 512
        model_config.hidden_size = 256
        model_config.image_size = 32
        model_config.patch_size = 4
        model_config.num_labels = 10 
    else:
        model_config = AutoConfig.from_pretrained(args.model_type)
        model_config.num_labels = 200 # tiny imagenet
    train_dataset_1 = get_dataset(args.dataset, 'train', args.aa,resize=64)
    test_dataset_1 = get_dataset(args.dataset, 'test', False,resize=64)

    num_classes = len(train_dataset.classes)
    print(num_classes)

    train_loader = DataLoader(train_dataset_1, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset_1, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=True)
    net = CustomViTForImageClassification(model_config)
    layers = len(net.vit.encoder.layer)
    print(layers)
    for i in range(layers):
        if not args.skip_gelu:
            net.vit.encoder.layer[i].intermediate.intermediate_act_fn = LearnableAlpha(model_config)
        if not args.skip_softmax:
            net.vit.encoder.layer[i].attention.attention.betas = LearnableBeta(model_config)
    # Create a new model, init weights from scratch, switch on the mask, use kd to converge.
    # net = copy.deepcopy(model)
    # layers = len(net.vit.encoder.layer)
    # model_config = AutoConfig.from_pretrained(args.model_type)
    # for i in range(layers):
    #     if not args.skip_gelu:
    #         net.vit.encoder.layer[i].intermediate.intermediate_act_fn = LearnableAlpha(model_config)
    #     if not args.skip_softmax:
    #         net.vit.encoder.layer[i].attention.attention.betas = LearnableBeta(model_config)
    
    net = net.to(device)

    optimizer = Adam(net.parameters(), lr=args.lr)
    total_softmax = softmax_counting(net, args)
    total_relu = gelu_counting(net, args)

    lowest_softmax_count, softmax_count = total_softmax, total_softmax
    lowest_relu_count, relu_count = total_relu, total_relu
    nonlinear_count = softmax_count + relu_count
    frozen_softmax = False
    frozen_gelu = False
    # corresponds to line 4-9
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            print("Current epochs: ", epoch)
            print("Nonlinear Count: {:}".format(nonlinear_count))
            print("Softmax Count: {:}".format(softmax_counting(net, args)))
            print("GELU Count: {:}".format(gelu_counting(net, args)))
        
        train_loss = mask_train_kd(train_loader, net, model, criterion, optimizer, epoch, device, args.alpha_gelu, args.alpha_softmax,args.cutmix_prob)
        log(logfilename, "Epoch {:}, Mask Update Train Loss: {train_loss:.4f}".format(epoch, train_loss = train_loss))
        _, test_acc, _ = test(test_loader, net, criterion, device, display=False)
        log(logfilename, "Epoch {:}, Mask Update Test Acc: {:.5}".format(epoch, test_acc))

        softmax_count = softmax_counting(net, args)
        relu_count = gelu_counting(net, args)
        nonlinear_count = softmax_count + relu_count

        penalty_flag_softmax = lowest_softmax_count - softmax_count <args.soft_interval  and not frozen_softmax # to avoid penalizing once budget has reached
        penalty_flag_gelus = lowest_relu_count - relu_count <args.gelu_interval  and not frozen_gelu
        if penalty_flag_softmax and epoch >= 5:
            args.alpha_softmax *= 1.1
            log(logfilename, "args.alpha_softmax = {}".format(args.alpha_softmax))
        
        if penalty_flag_gelus and epoch >= 5:
            args.alpha_gelu *= 1.1
            log(logfilename, "args.alpha_gelus = {}".format(args.alpha_gelu))

        if softmax_count < lowest_softmax_count:
            lowest_softmax_count = softmax_count
        
        if relu_count < lowest_relu_count:
            lowest_relu_count = relu_count
        
        freeze_gelu = relu_count <=args.gelu_budget
        if freeze_gelu and not frozen_gelu:
            log(logfilename,"Freezing Gelus")
            print("GELU Count: {:}".format(relu_count))
            frozen_gelu = True
            for name, param in net.named_parameters():
                if 'alpha' in name:
                    boolean_list = param.data > args.threshold
                    param.data = boolean_list.float()
                    param.requires_grad = False
        
        freeze_softmax = softmax_count <=args.softmax_budget
        if freeze_softmax and not frozen_softmax:
            frozen_softmax = True
            log(logfilename,"Freezing Softmax")
            print("Softmax Count: {:}".format(softmax_count))
            for name, param in net.named_parameters():
                if 'beta' in name:
                    boolean_list = param.data > args.threshold
                    param.data = boolean_list.float()
                    param.requires_grad = False


        break_flag = frozen_softmax and frozen_gelu # ensures not stuck in infinite loop
        if epoch % args.save_every == 0:
            print(f'saving model at epoch {epoch} with relu count {relu_count} and softmax count {softmax_count}')
            torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, f'snl_vit_nas_epoch_{epoch}_checkpoint.pth.tar'))
        if break_flag:
            print("Current epochs breaking loop at {:}".format(epoch))
            break
    log(logfilename, "After SNL Algorithm, the current Nonlinear count: {}".format(nonlinear_count))
    log(logfilename, "After SNL Algorithm, the current Softmax Count: {}".format(softmax_counting(net, args)))
    log(logfilename, "After SNL Algorithm, the current Gelu Count: {}".format(gelu_counting(net, args)))

    # Binarize the learned mask and freeze them to be extra sure
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            param.data = boolean_list.float()
            param.requires_grad = False
        
        if 'beta' in name:
            boolean_list = param.data > args.threshold
            param.data = boolean_list.float()
            param.requires_grad = False

    # Line 12
    finetune_epoch = 10 if args.debug else args.finetune_epochs
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

    log(logfilename, "Finetuning the model")

    best_top1 = 0
    for epoch in range(finetune_epoch):
        train_loss, train_top1, train_top5 = train(train_loader, net, model, optimizer, criterion, epoch, device,args.cutmix_prob)
        test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
        scheduler.step()

        if best_top1 < test_top1:
            best_top1 = test_top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'snl_vit_best_checkpoint.pth.tar'))

    print("Final best Prec@1 = {}%".format(best_top1))
    log(logfilename, "Final best Prec@1 = {}%".format(best_top1))

    
if __name__ == "__main__":
    main(args)
