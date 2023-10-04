import torch
import numpy as np
def log(filename: str, text: str):
    print(text)
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def gelu_counting(net, args):
    gelu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            gelu_count += (boolean_list == 1).sum()
    return gelu_count
def softmax_counting(net, args):
    softmax_count = 0
    for name, param in net.named_parameters():
        if 'beta' in name:
            boolean_list = param.data > args.threshold
            softmax_count += (boolean_list == 1).sum()
    return softmax_count

def cutmix_data(inputs, targets, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    bsize, _, h, w = inputs.shape
    shuffled_indices = torch.randperm(bsize)
    inputs_s, targets_s = inputs[shuffled_indices], targets[shuffled_indices]

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)

    # Uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs_s[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    targets = (targets, targets_s, lam)

    return inputs, targets