import torch
from tqdm import tqdm
import random
import pdb
from termcolor import cprint
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Lib import MetricLogger, get_metrics


def train(
    net,
    view_mode,
    device,
    dataloader,
    optimizer,
    epoch,
    criterion,
    iter_dis,
    rank=0,
    wandb=None,
    **kwargs,
):
    loss_logger = MetricLogger()
    metrics = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    gt_labels = []
    pred_labels_fuse = []
    net.train()
    name_pred_dict = {}
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        effective_views = data["effective_views"]
        effective_views_tensors = data["effective_views_tensors"].to(device, non_blocking=True)
        gt_labels.append(data["label"])
        pred_logits = net(effective_views,effective_views_tensors)
        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        pred_label_fuse = (
            torch.argmax(F.softmax(pred_logits[0], dim=-1), dim=-1).detach().cpu()
        )
        pred_labels_fuse.append(pred_label_fuse)
        pred_label_fuse = [str(i.item()) for i in list(pred_label_fuse)]
        for name, pred in zip(data["name"], pred_label_fuse):
            name_pred_dict[name] = pred
        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "TRAIN || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )
    gt_labels_flat = list(torch.cat(gt_labels, dim=0).squeeze(dim=-1).numpy())
    pred_labels_fuse_flat = list(torch.cat(pred_labels_fuse).numpy())
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_fuse_flat, index=""))
    return (
        net,
        loss_logger,
        gt_labels_flat,
        pred_labels_fuse_flat,
        metrics,
        name_pred_dict,
    )


@torch.no_grad()
def val(
    net,
    view_mode,
    device,
    dataloader,
    epoch,
    criterion,
    iter_dis,
    savename,
    rank=0,
    **kwargs,
):
    loss_logger = MetricLogger()
    metrics = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    gt_labels = []
    pred_labels_fuse = []
    pred_labels_cor = []
    pred_labels_sag = []
    net.eval()
    name_pred_dict = {}
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        effective_views = data["effective_views"]
        effective_views_tensors = data["effective_views_tensors"].to(device, non_blocking=True)
        gt_labels.append(data["label"])
        pred_logits = net(effective_views,effective_views_tensors)
        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)

        pred_label_fuse = (
            torch.argmax(F.softmax(pred_logits[0], dim=-1), dim=-1).detach().cpu()
        )
        pred_labels_fuse.append(pred_label_fuse)
        pred_label_fuse = [str(i.item()) for i in list(pred_label_fuse)]
        for name, pred in zip(data["name"], pred_label_fuse):
            name_pred_dict[name] = pred
        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "VAL || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )
    gt_labels_flat = list(torch.cat(gt_labels, dim=0).squeeze(dim=-1).numpy())
    pred_labels_fuse_flat = list(torch.cat(pred_labels_fuse).numpy())
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_fuse_flat, index=""))
    return (
        net,
        loss_logger,
        gt_labels_flat,
        pred_labels_fuse_flat,
        metrics,
        name_pred_dict,
    )
