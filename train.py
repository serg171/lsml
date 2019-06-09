from collections import defaultdict
import torch
from torch.nn import functional as F
from tqdm import tqdm
#import matplotlib.pyplot as plt
import numpy as np

import utils

#plt.switch_backend('agg')


def grad_norm(model):
    grad = 0.0
    count = 0
    for name, tensor in model.named_parameters():
        if tensor.grad is not None:
            grad += torch.sqrt(torch.sum((tensor.grad.data) ** 2))
            count += 1
    return grad.cpu().numpy() / count


class Trainer:
    global_step = 0

    def __init__(self, train_writer=None, eval_writer=None, compute_grads=True, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        self.compute_grads = compute_grads

    def train_epoch(self, model, optimizer, dataloader, scheduler, criterion, log_prefix=""):
        device = self.device
        scheduler.step()

        model = model.to(device)
        model.train()
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
            
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            corrects = np.sum(preds.cpu().numpy() == labels.cpu().numpy()) 

            log_entry = dict(
                acc=corrects,
                loss=loss.item(),
                lr=scheduler.get_lr()[0],
            )
            if self.compute_grads:
                log_entry['grad_norm'] = grad_norm(model)

            for name, value in log_entry.items():
                if log_prefix != '':
                    name = log_prefix + '/' + name
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            self.global_step += 1

    def eval_epoch(self, model, dataloader, criterion, log_prefix=""):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = model.to(device)
        model.eval()
        metrics = defaultdict(list)
        
        for inputs, labels in tqdm(dataloader):
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                corrects = np.sum(preds.cpu().numpy() == labels.cpu().numpy())
                
                metrics['loss'].append(loss.item())
                metrics['acc'].append(corrects)

        metrics = {key: np.mean(values) for key, values in metrics.items()}
        for name, value in metrics.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            self.eval_writer.add_scalar(name, value, global_step=self.global_step)

        return metrics