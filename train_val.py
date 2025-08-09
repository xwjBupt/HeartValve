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
    device,
    dataloader,
    optimizer,
    epoch,
    criterion,
    iter_dis,
    rank=0,
    wandb=None,
    num_classes=2,
    **kwargs,
):
    loss_logger = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    all_logits, all_labels = [], []
    net.train()
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        effective_views = data["effective_views"]
        effective_views_tensors = data["effective_views_tensors"]
        video_names = data["video_names"]
        patient_name = data["patient_name"]
        pred_logits = net(effective_views, effective_views_tensors, device)
        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        all_logits.append(pred_logits.cpu().detach())
        all_labels.append(label.cpu().detach())

        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "TRAIN || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    return (
        net,
        loss_logger,
        labels,
        logits,
    )


@torch.no_grad()
def val(
    net,
    device,
    dataloader,
    epoch,
    criterion,
    iter_dis,
    rank=0,
    **kwargs,
):
    loss_logger = MetricLogger()
    metrics = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    all_logits, all_labels = [], []
    net.eval()
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        effective_views = data["effective_views"]
        effective_views_tensors = data["effective_views_tensors"]
        patient_name = data["patient_name"]
        pred_logits = net(effective_views, effective_views_tensors, device)
        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)

        all_logits.append(pred_logits.cpu().detach())
        all_labels.append(label.cpu().detach())
        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "VAL || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()

    return (
        net,
        loss_logger,
        labels,
        logits,
    )
