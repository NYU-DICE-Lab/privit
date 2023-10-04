import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy
import time
import numpy as np
from utils import cutmix_data

class SoftTarget(nn.Module):
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


def train(train_loader, models_student, models_teacher, optimizer, criterion, epoch, device,cutmix_prob=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    kd_flag = False
    if models_teacher is not None:
        kd_flag = True
        models_teacher.eval()
    models_student.train()

    if kd_flag:
        criterion_kd = SoftTarget(4.0).to(device)

    for i, (inputs, targets) in enumerate(train_loader):
        if kd_flag:
            inputs_kd = F.interpolate(inputs,size =224, mode='bilinear')
            inputs_kd = inputs_kd.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)
        r = np.random.rand(1)
        if r < cutmix_prob:
            # do cut mix training
            inputs, labels = cutmix_data(inputs, targets, alpha=1.0)
            targets, targets_s, lam = labels

            outputs_s = models_student(inputs)["logits"]
        
            if kd_flag:
                outputs_t = models_teacher(inputs_kd)["logits"]

            if kd_flag:
                loss = lam*criterion(outputs_s, targets) + (1-lam)*criterion(outputs_s, targets_s)
                loss_kd = criterion_kd(outputs_s, outputs_t.detach()) * 1.0 # no change to kd loss due to cutmix
                total_loss = loss + loss_kd
            else:
                # import pdb; pdb.set_trace()
                total_loss = lam*criterion(outputs_s, targets) + (1-lam)*criterion(outputs_s, targets_s)
        else:
            # do normal training

            outputs_s = models_student(inputs)["logits"]
        
            if kd_flag:
                outputs_t = models_teacher(inputs_kd)["logits"]

            if kd_flag:
                loss = criterion(outputs_s, targets)
                loss_kd = criterion_kd(outputs_s, outputs_t.detach()) * 1.0
                total_loss = loss + loss_kd
            else:
                total_loss = criterion(outputs_s, targets)

        # print(outputs_s.shape)
        # print(targets.shape)
        acc1, acc5 = accuracy(outputs_s, targets, topk=(1, 5))
        losses.update(total_loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    print(
        'Epoch: [{0}][{1}/{2}]\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5)
    )
    return (losses.avg, top1.avg, top5.avg)

def test(loader: DataLoader, model: torch.nn.Module, criterion, device, print_freq=100, display=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # compute output
            outputs = model(inputs)["logits"]
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and display == True:
                print('Test : [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        print(
            'Test Loss  ({loss.avg:.4f})\t'
            'Test Acc@1 ({top1.avg:.3f})\t'
            'Test Acc@5 ({top5.avg:.3f})'.format(
        loss=losses, top1=top1, top5=top5)
        )

        return (losses.avg, top1.avg, top5.avg)


def mask_train_kd(train_loader, model, model_teacher, criterion, optimizer, epoch, device, alpha_gelu,alpha_softmax,cutmix_prob=0):
    # import pdb; pdb.set_trace()
    losses = AverageMeter()
    kd_flag = False
    if model_teacher is not None:
        kd_flag = True
        model_teacher.eval()

    model.train()

    if kd_flag:
        criterion_kd = SoftTarget(4.0).to(device)

    for i, (inputs, targets) in enumerate(train_loader):
        if kd_flag:
            inputs_kd = F.interpolate(inputs,size=224, mode='bilinear')
            inputs_kd = inputs_kd.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        reg_loss_softmax = 0
        reg_loss_gelu = 0

        for name, param in model.named_parameters():   
            if 'alpha' in name:
                reg_loss_gelu += torch.norm(param, p=1)        
            if 'beta' in name:
                reg_loss_softmax += torch.norm(param, p=1)        
        
        r = np.random.rand(1)
        if r < cutmix_prob:
            inputs, labels = cutmix_data(inputs, targets, alpha=1.0)
            targets, targets_s, lam = labels
            outputs_s = model(inputs)["logits"]
            if kd_flag:
                outputs_t = model_teacher(inputs_kd)["logits"]
            loss_CE = lam*criterion(outputs_s,targets)+(1-lam)*criterion(outputs_s,targets_s)
        else:
            outputs_s = model(inputs)["logits"]
            if kd_flag:
                # inputs_kd.to(device)
                outputs_t = model_teacher(inputs_kd)["logits"]
            loss_CE = criterion(outputs_s, targets)
        
        if kd_flag:
            loss = loss_CE + criterion_kd(outputs_s, outputs_t) + alpha_softmax*reg_loss_softmax + alpha_gelu*reg_loss_gelu
        else:
            loss = loss_CE + alpha_softmax*reg_loss_softmax + alpha_gelu*reg_loss_gelu

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg