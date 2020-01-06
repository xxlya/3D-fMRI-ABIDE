import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy

from opts import parse_opts

opt = parse_opts()
device = torch.device("cuda" if opt.use_cuda else "cpu")

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,log_path,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        #print('input shape', inputs.shape, 'targets shape', targets.shape)
        data_time.update(time.time() - end_time)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs).float()
        targets = Variable(targets).long()
        inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], 1, inputs.shape[3], inputs.shape[4],
                    inputs.shape[5])
        targets = torch.squeeze(targets.view(-1,targets.shape[0]*targets.shape[1]),0)
        #print('input shape', inputs.shape, 'targets shape', targets.shape)
        outputs = model(inputs)

        #print(targets)
        #print(outputs)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(float(loss), inputs.size(0))
        accuracies.update(float(acc), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(log_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            #'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(states, save_file_path)