import os
import torch
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment

def get_dataset(dataset: str, split, augment,resize=224, dpath: str = None ): 
    if dataset == "imagenet":
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STDDEV = [0.229, 0.224, 0.225]
        if split == "train":
            subdir = os.path.join("/data/datasets/ImageNet", "train")
            transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV),
            transforms.RandomErasing()
        ])
            if augment:
                transform.transforms.insert(0, RandAugment(2, 9))
        elif split == "test":
            subdir = os.path.join("/data/datasets/ImageNet/val", "val")
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)
            ])
        return datasets.ImageFolder(subdir, transform)

    elif dataset == "tiny_imagenet":
        if dpath == None:
            dpath = "/scratch/nhd7682/tiny-imagenet-200/"
        _TINY_MEAN = [0.480, 0.448, 0.398]
        _TINY_STD = [0.277, 0.269, 0.282]   
        if split == "train":
            subdir = os.path.join(dpath, "train")
            transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD),
            transforms.RandomErasing()
        ])
            if augment:
                transform.transforms.insert(0, RandAugment(2, 9))
        elif split == "test":
            subdir = os.path.join(dpath, "val")
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(_TINY_MEAN, _TINY_STD)
            ])
        return datasets.ImageFolder(subdir, transform)

    elif dataset in ["cifar10", "cifar100"]:
        _CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        _CIFAR_STDDEV = [0.2023, 0.1994, 0.2010]

        if split == "train" and dataset == "cifar10":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STDDEV),
                transforms.RandomErasing()
                ])
            if augment:
                transform.transforms.insert(0, RandAugment(2, 9))
            return datasets.CIFAR10("/scratch/nhd7682/dataset_cache", train=True, download=True, transform=transform)
        elif split == "test" and dataset == "cifar10":
            return datasets.CIFAR10("/scratch/nhd7682/dataset_cache", train=False, download=True, transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STDDEV)
                ]))
        if split == "train" and dataset == "cifar100":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                transforms.RandomErasing()
                ])
            if augment:
                transform.transforms.insert(0, RandAugment(2, 9))
            return datasets.CIFAR100("/scratch/nhd7682/dataset_cache", train=True, download=True, transform=transform)
        elif split == "test" and dataset == "cifar100":
            return datasets.CIFAR100("/scratch/nhd7682/dataset_cache", train=False, download=True, transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]))       
    else:
        raise NotImplementedError("Not implemented datasets.")