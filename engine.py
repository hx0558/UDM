# TODO: ADD INFO

"""
Train and eval functions used in main.py
"""
import time
import torch.nn.functional as F
import numpy as np

import util
import datetime

from timm.utils import accuracy, AverageMeter
from timm.scheduler.scheduler import Scheduler

import torch
from torch.utils.data.dataloader import DataLoader

from util import false_logger
from sklearn import metrics
import scipy.io as sio
from loss import *



def one_hot(x, class_count):
	return torch.eye(class_count)[x,:]
def ece_hist_binary(p, label, n_bins=15, order=1):
    p=np.array(p)
    p = np.clip(p, 1e-256, 1 - 1e-256)

    N = p.shape[0]
    onehot_label=one_hot(np.array(label),17)
    label_index = np.array([np.where(r == 1)[0][0] for r in onehot_label])
    with torch.no_grad():
        preds_new = torch.from_numpy(p)
        preds_b = torch.zeros(N, 1)
        label_binary = np.zeros((N, 1))
        for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)
        binCount = int(len(x) / n_bins)
        bins = np.zeros(n_bins)
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i + 1) * binCount, x.shape[0] - 1)]

        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order * prop_in_bin
    return ece_avg


def model_forward(model, x_hsi, x_lidar, y):
    x_hsi = x_hsi.cuda()
    x_lidar = x_lidar.cuda()
    y = y.cuda()
    hsi_LiDAR_logits, hsi_logits, LiDAR_logits, hsi_mu, hsi_logvar, LiDAR_mu, LiDAR_logvar, mu, logvar, z, cog_un = model(
        x_hsi, x_lidar)

    conloss = con_loss(hsi_mu, torch.exp(hsi_logvar), LiDAR_mu, torch.exp(LiDAR_logvar))
    loss = totalloss(hsi_LiDAR_logits, hsi_logits, y, LiDAR_logits, hsi_mu, hsi_logvar, LiDAR_mu, LiDAR_logvar, mu,
                     logvar, z)

    loss = loss + 5e-4 * KL_cross(hsi_mu, hsi_logvar, LiDAR_mu, LiDAR_logvar) + conloss * 1e-3

    return loss, hsi_LiDAR_logits, cog_un


def train_one_epoch(
        args,
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Scheduler,
        mixup_fn=None, max_norm: float = 0,
        epoch=None, total_epochs=None,
        logger=false_logger,):

    model.train()
    criterion.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_spectral = AverageMeter()
    PRINT_FREQ = 100
    train_all = []
    start = time.time()
    end = time.time()
    for idx, (samples, samples_specral,targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        samples_specral=samples_specral.cuda(non_blocking=True)
        loss, output, cog_uncertainty_dict=model_forward(model, samples, samples_specral, targets)
        optimizer.zero_grad()
        loss.backward()

        if args.modulation_starts <= epoch <= args.modulation_ends:
            coeff_hsi = args.zeta * cog_uncertainty_dict['l'].mean()
            coeff_lidar = args.zeta * cog_uncertainty_dict['v'].mean()
            for name, parms in model.named_parameters():
                if parms.grad is None: continue
                if any(_ in name for _ in ["HSI_Clf"]):
                    parms.grad = parms.grad * (1 + coeff_lidar)
                if any(_ in name for _ in ["LiDAR_Clf"]):
                    parms.grad = parms.grad * (1 + coeff_hsi)
        else:
            pass

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = util.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_total_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        acc_meter.update(accuracy(output, targets, topk=(1,))[0])  #

        if (idx + 1) % PRINT_FREQ == 0 or (idx + 1) == num_steps:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{total_epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        train_all = np.concatenate([train_all, targets.data.cpu().numpy()])
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter


@torch.no_grad()
def evaluate(
        args,
        seed,
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module ,
        acc, A, k, ECE,exp_time,
        logger=false_logger,
         ):
    model.eval()
    criterion.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # acc_meter_spectral = AverageMeter()
    PRINT_FREQ = 2000

    test_pred_all = []
    test_all = []
    correct = 0
    total = 0
    output_all=[]
    with torch.no_grad():
        for idx, (samples, samples_specral,targets) in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            samples_specral = samples_specral.cuda(non_blocking=True)
            loss, output, cog_uncertainty_dict = model_forward(model, samples, samples_specral, targets)

            output_all.append(output.detach().cpu().numpy())

            _, predicted = torch.max(output.data, 1)
            test_all = np.concatenate([test_all, targets.data.cpu().numpy()])
            test_pred_all = np.concatenate([test_pred_all, predicted.cpu()])
            correct += predicted.eq(targets.data.view_as(predicted)).cpu().sum()

            acc2 = accuracy(output, targets, topk=(1,))[0]
            end = time.time()
            loss_meter.update(loss.item(), targets.size(0))
            acc_meter.update(acc2.item(), targets.size(0))
            batch_time.update(time.time() - end)

            if (idx + 1) % PRINT_FREQ == 0 or (idx + 1) == num_steps:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'EVAL: [{(idx + 1)}/{len(data_loader)}]\t'
                    # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

    outputall=np.concatenate(output_all)
    sio.savemat(f'{args.dataset}_{seed}_UDM.mat', {'output_all': outputall, 'test_pred_all': test_pred_all})
    acc[exp_time-1] = 100. * correct / len(data_loader.dataset)
    OA = acc

    num_classes = A.shape[1]
    C = metrics.confusion_matrix(test_all, test_pred_all, labels=range(num_classes))
    with np.errstate(divide='ignore', invalid='ignore'):
        current_acc = np.diag(C) / np.sum(C, axis=1, dtype=np.cfloat)

    current_acc = np.nan_to_num(current_acc)
    A[exp_time - 1, :] = current_acc

    k[exp_time-1] = metrics.cohen_kappa_score(test_all, test_pred_all)
    ECE[exp_time-1] = ece_hist_binary(outputall, test_all)[0].cpu().numpy()
    print(ECE[exp_time-1])
    logger.info(f'EVAL * Acc@ {acc_meter.avg:.3f}')
    return acc_meter.avg, loss_meter.avg, acc, A, k,ECE

def get_flops(
            model: torch.nn.Module,
            trainset: torch.utils.data.Dataset,
            logger = None,
):
    logger = false_logger() if logger is None else logger
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    sample_x1, sample_x2, _ = trainset[0]
    sample_x1 = sample_x1.unsqueeze(0).cuda()
    sample_x2 = sample_x2.unsqueeze(0).cuda()

    flops = FlopCountAnalysis(model, (sample_x1, sample_x2))

    total_flops = flops.total()

    formatted_flops = "{:.6f}".format(total_flops)

    logger.info(f"Formatted FLOPs: {formatted_flops}")

    table_str = flop_count_table(flops, max_depth=2, activations=None, show_param_shapes=False)
    logger.info(table_str)
    return formatted_flops




