import time
import json
import os 
import wandb
import logging

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from metrics import compute_pro, trapezoid

_logger = logging.getLogger('train')

class AverageMeter:
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



def training(model, trainloader, validloader, criterion, optimizer, scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], 
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, device: str ='cpu') -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights
    
    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    while train_mode:

        end = time.time()
        for inputs, masks, targets in trainloader:
            # batch
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)

            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss.backward()
            
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())
            
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss':focal_losses_m.val,
                    'train_l1_loss':l1_losses_m.val,
                    'train_loss':losses_m.val
                },
                step=step)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps, 
                            loss       = losses_m, 
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))


            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    model        = model, 
                    dataloader   = validloader, 
                    device       = device
                )
                model.train()

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

    

        
def evaluate(model, dataloader, device: str = 'cpu'):
    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []
    
    # aupro
    integration_limit = 0.3

    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # stack targets and outputs
            image_targets.extend(targets.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())
            
            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            
    # metrics    
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)
    
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )

    aupro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
    aupro /= integration_limit
    
    metrics = {
        'AUROC-image':auroc_image,
        'AUROC-pixel':auroc_pixel,
        'AUPRO-pixel':aupro

    }

    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' % 
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))


    return metrics
