from __future__ import absolute_import
from __future__ import division
import time
from loguru import logger
import pdb
import shutil
import torch
import os
from collections import OrderedDict
import argparse
import wandb
import numpy as np
import pdb
import setproctitle
import matplotlib.pyplot as plt
import json
from yacs.config import CfgNode as CN
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from Lib import (
    git_commit,
    get_metrics,
    get_learning_rate,
    warmup_lr_changer,
    save_checkpoint,
    write_yaml,
    iter_update_dict,
    update_config,
    MetricLogger,
    nodeTodict,
    iter_wandb_log,
    write_json,
    plot_confusion_matrix,
    calculate_and_save_metrics,
)
from builder import build_loader, build_opt, build_loss, build_scheduler, build_model
from train_val import train, val
from infer import infer
from sync_batchnorm import convert_model
from config import _C
import random
from sklearn.metrics import confusion_matrix


def get_arguments(args):
    cfg = _C.clone()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    if not args.nodebug:
        cfg.METHOD.Name = "DEBUG"
        cfg.BASIC.DEBUG = True
    if cfg.BASIC.Seed == 1:
        cfg.BASIC.Seed = random.randint(2, 25530)
    cfg.freeze()
    return cfg


@logger.catch
def main(cfg):
    rootinroot = os.path.dirname(os.path.realpath(__file__))
    timestamp = time.strftime("%Y-%m%d-%H%M", time.localtime())
    expriment = (
        cfg.DATA.Train.DataPara.name
        + "/"
        + cfg.METHOD.Name
        + "/"
        + cfg.METHOD.Desc.replace("/", "/" + timestamp + "#")
    )
    root = rootinroot + "/output_runs/" + expriment + "/"
    setproctitle.setproctitle(cfg.METHOD.Desc)
    save_model = root + "/" + "Model"
    logger.add(root + "/" + "output.log")
    write_yaml(root + "/" + "config.yaml", nodeTodict(cfg))
    dir_needs = [save_model]
    commit_info = (
        "COMMIT INFO >>> "
        + cfg.METHOD.Name
        + "#"
        + timestamp
        + "#"
        + cfg.METHOD.Desc
        + " <<<"
    )
    wandb_tag = cfg.METHOD.Name + "#" + timestamp + "#" + cfg.METHOD.Desc
    os.environ["WANDB_MODE"] = "dryrun"
    if cfg.BASIC.DEBUG:

        logger.info("In debug, using wandb Offline")
        run = wandb.init(
            config=cfg,
            project=cfg.DATA.Train.DataPara["name"],
            name=wandb_tag,
            sync_tensorboard=False,
            reinit=True,
        )
    else:
        logger.info("Not in debug, also using wandb On line")
        record_commit_info = git_commit(rootinroot, timestamp, commit_info=commit_info)
        logger.info(record_commit_info)
        run = wandb.init(
            config=cfg,
            project=cfg.DATA.Train.DataPara["name"],
            name=wandb_tag,
            sync_tensorboard=False,
            reinit=True,
        )
    for dir_need in dir_needs:
        os.makedirs(dir_need, exist_ok=True)

    if cfg.BASIC.Seed is not None:
        np.random.seed(cfg.BASIC.Seed)
        torch.manual_seed(cfg.BASIC.Seed)
        torch.cuda.manual_seed(cfg.BASIC.Seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.BASIC.Num_gpus
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        device = "cpu"
        logger.info("There is no GPUs, using CPU")
    else:
        logger.info("There is %d GPUs, using GPU" % num_gpus)
        device = "cuda"

    train_dataloader = build_loader(cfg.DATA.Train)
    val_dataloader = build_loader(cfg.DATA.Val)
    loss = build_loss(cfg.LOSS)
    net = build_model(cfg.MODEL)
    optimizer = build_opt(cfg.OPT, net)
    scheduler = build_scheduler(cfg.SCHEDULER, optimizer)

    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("num_parameters of the using net is {}".format(num_parameters))

    if cfg.BASIC.Finetune or cfg.BASIC.Resume:
        if cfg.BASIC.Resume:
            check = torch.load(cfg.BASIC.Resume, map_location=torch.device("cpu"))
            logger.info("Loading weight from:%s" % cfg.BASIC.Resume)
        else:
            check = torch.load(cfg.BASIC.Finetune, map_location=torch.device("cpu"))
            logger.info("Loading weight from:%s" % cfg.BASIC.Finetune)
        pre_net_dict = check["net_dict"]
        raw_net_dict = net.state_dict()
        new_state_dict = OrderedDict()
        unloaded_keys = []
        for k, v in pre_net_dict.items():
            if "module" in k and k[7:] in raw_net_dict.keys():
                new_state_dict[k[7:]] = v
            if k in raw_net_dict.keys():
                new_state_dict[k] = v
            else:
                unloaded_keys.append(k)
        raw_net_dict.update(new_state_dict)
        net.load_state_dict(raw_net_dict)

        logger.info("unloaded keys: ")
        logger.info(unloaded_keys)
        net.to(device)

        if cfg.BASIC.Resume:
            optimizer.load_state_dict(check["optimizer_dict"])
            start_epoch = check["epoch"]
            best_mean_loss = check["best_loss"]
            best_mean_acc = check["best_mean_ACC"]
            best_mean_loss_epoch = -1
            best_mean_acc_epoch = -1
            logger.info(
                "Current start_epoch: %d - best_loss: %.5f - best_ACC: %.5f"
                % (start_epoch, best_mean_loss, best_mean_acc)
            )
    if not cfg.BASIC.Resume:
        start_epoch = 0
        init_lr = cfg.OPT.Para.lr
        best_mean_loss = 10000.0
        best_mean_loss_epoch = -1
        best_mean_acc_epoch = -1
        best_mean_acc = -1.0
        best_mean_acc_smooth_epoch = -1
        best_mean_acc_smooth = -1.0
        best_mean_acc_fuse_epoch = -1
        best_mean_acc_fuse = -1.0
        best_mean_acc_smooth_fuse_epoch = -1
        best_mean_acc_smooth_fuse = -1.0
        best_mean_recall = -1.0
        best_mean_recall_epoch = -1
        best_mean_f1 = -1.0
        best_mean_f1_epoch = -1
        best_mean_auc = -1.0
        best_mean_auc_epoch = -1
        final_stop = cfg.BASIC.Epochs

    logger.info("using loss {}".format(loss))

    if num_gpus > 1:
        net = torch.nn.DataParallel(net)
        logger.info("Using DataParallel with %d GPUS<<<" % num_gpus)
    net.to(device)
    loss.to(device)
    logger.info(net)
    logger.info(">>> START TRAINING <<<")
    wandb.watch(net, loss, log="all", log_freq=5)

    train_metich_logger = MetricLogger()
    val_metich_logger = MetricLogger()
    num_classes = cfg.MODEL.Para.model_num_class
    try:
        for epoch in range(start_epoch + 1, cfg.BASIC.Epochs, 1):
            epoch_start = time.time()
            if epoch <= cfg.BASIC.Warmup_epoch and not cfg.BASIC.Resume:
                newlr = warmup_lr_changer(
                    init_lr, epoch, warmup_epoch=cfg.BASIC.Warmup_epoch
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = newlr
            rightnow_lr = get_learning_rate(optimizer)

            ### TRAIN START ###
            train_start = time.time()
            logger.info(
                ">>> [ %s ][ EPOCH: %d / %d ][ 100*LR:%.6f ] <<<"
                % (
                    expriment + "@" + timestamp,
                    epoch,
                    cfg.BASIC.Epochs,
                    rightnow_lr * 100,
                )
            )

            iter_dis = len(train_dataloader) // (len(train_dataloader) // 10 + 1) + 3
            (
                net,
                train_loss_logger,
                train_labels,
                train_logits,
            ) = train(
                net,
                device,
                train_dataloader,
                optimizer,
                epoch,
                loss,
                iter_dis,
                wandb=wandb,
            )
            pos = (train_labels == 1).sum()
            neg = (train_labels == 0).sum()
            logger.debug(
                "Trainging: Positive: {}, Negative: {}, Ratio (/): {}\n raw train_logits {} \n sigmoid train_logits {}".format(
                    pos, neg, pos / neg, train_logits, torch.sigmoid(train_logits)
                )
            )
            train_metrics, train_cm = calculate_and_save_metrics(
                train_labels,
                train_logits,
                epoch,
                log_dir=save_model,
                phase="train",
                num_classes=num_classes,
                threshold=0.5,
            )

            train_metich_logger.update(**train_metrics)
            train_fps = (time.time() - train_start) / len(train_dataloader)
            train_time = time.time() - train_start
            logger.info(
                ">>> TRAIN EPOCH: %d TIME: %.4f FPS: %.4f LOSS: %s  <<<"
                % (epoch, train_time, train_fps, train_loss_logger.lineout())
            )

            logger.info(
                ">>> TRAIN CLASSIFICATION METRICS || {} || <<< \n train_confusion_matrix \n {} \n".format(
                    train_metich_logger.lineout(), train_cm
                )
            )
            plt.figure(figsize=(8, 4))
            plt.hist(
                train_logits, bins=50, range=(0, 1), color="skyblue", edgecolor="black"
            )
            plt.title("Epoch %03d output logit distribution" % (epoch))
            plt.xlabel("Sigmoid Output (train Predicted Probability)")
            plt.ylabel("Count")
            plt.grid(True)
            plt.savefig(
                save_model + "/train_sigmoid_output_logit_Epoch%04d.png" % (epoch)
            )
            plt.close()

            ### TRAIN STOP ###

            ### VAL START ###
            val_start = time.time()
            iter_dis = len(val_dataloader) // (len(val_dataloader) // 20 + 1) + 3
            (
                net,
                val_loss_logger,
                val_labels,
                val_logits,
            ) = val(
                net,
                device,
                val_dataloader,
                epoch,
                loss,
                iter_dis=iter_dis,
            )
            pos = (val_labels == 1).sum()
            neg = (val_labels == 0).sum()
            logger.debug(
                "VALIDATE: Positive: {}, Negative: {}, Ratio (/): {}\n raw val_logits {} \n sigmoid val_logits {}".format(
                    pos, neg, pos / neg, val_logits, torch.sigmoid(val_logits)
                )
            )

            val_metrics, val_cm = calculate_and_save_metrics(
                val_labels,
                val_logits,
                epoch,
                log_dir=save_model,
                phase="val",
                num_classes=num_classes,
                threshold=0.5,
            )
            val_metich_logger.update(**val_metrics)
            val_fps = (time.time() - val_start) / len(val_dataloader)
            val_time = time.time() - val_start
            logger.info(
                ">>> VAL EPOCH: %d TIME: %.4f FPS: %.4f LOSS: %s<<<"
                % (epoch, val_time, val_fps, val_loss_logger.lineout())
            )
            logger.info(
                ">>> VAL CLASSIFICATION METRICS || {} || <<< \n val_confusion_matrix \n {} \n".format(
                    val_metich_logger.lineout(), val_cm
                )
            )

            plt.figure(figsize=(8, 4))
            plt.hist(
                val_logits, bins=50, range=(0, 1), color="skyblue", edgecolor="black"
            )
            plt.title("Epoch %03d output logit distribution" % (epoch))
            plt.xlabel("Sigmoid Output (val Predicted Probability)")
            plt.ylabel("Count")
            plt.grid(True)
            plt.savefig(
                save_model + "/val_sigmoid_output_logit_Epoch%04d.png" % (epoch)
            )
            plt.close()
            ### VAL STOP ###

            # iter_wandb_log(wandb, train_loss_logger, phase="TRAIN#", index=epoch)
            # iter_wandb_log(wandb, val_loss_logger, phase="VAL#", index=epoch)
            # iter_wandb_log(wandb, train_metrics, phase="TRAIN#", index=epoch)
            # iter_wandb_log(wandb, val_metrics, phase="VAL#", index=epoch)

            if epoch > cfg.BASIC.Lr_decay:
                if cfg.SCHEDULER == "ReduceLROnPlateau":
                    scheduler.step(val_metrics.Acc_Seg.avg)
                elif cfg.SCHEDULER == "LambdaLR":  # todo test ok?
                    scheduler.step(epoch)
                else:
                    scheduler.step()
            net_dict = (
                net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            )

            saved = False
            current_loss = val_loss_logger.loss.avg
            current_acc = val_metrics.get("acc")
            current_recall = val_metrics.get("recall")
            current_f1 = val_metrics.get("f1")
            current_ap = val_metrics.get("ap")
            current_auc = val_metrics.get("auc")

            if current_f1 > best_mean_f1:
                best_mean_f1_epoch = epoch
                best_mean_f1 = current_f1
                logger.info(
                    "Mean_F1 Metric update to >>> Mean_F1: %.6f @ epoch: %d"
                    % (best_mean_f1, best_mean_f1_epoch)
                )
                if not saved:
                    save_dict = {
                        "epoch": epoch,
                        "best_loss": current_loss,
                        "best_acc": current_acc,
                        "best_recall": current_recall,
                        "best_f1": best_mean_f1,
                        "net_dict": net_dict,
                        "optimizer_dict": optimizer.state_dict(),
                        "lr": rightnow_lr,
                        "cfg": cfg,
                    }
                    save_checkpoint(
                        save_model,
                        epoch,
                        save_dict=save_dict,
                        best_f1_flag=True,
                    )

                saved = True

            if current_auc > best_mean_auc:
                best_mean_auc_epoch = epoch
                best_mean_auc = current_auc
                logger.info(
                    "Mean_auc Metric update to >>> Mean_auc: %.6f @ epoch: %d"
                    % (best_mean_auc, best_mean_auc_epoch)
                )
                if not saved:
                    save_dict = {
                        "epoch": epoch,
                        "best_loss": current_loss,
                        "best_acc": current_acc,
                        "best_recall": current_recall,
                        "best_f1": current_f1,
                        "best_auc": best_mean_auc,
                        "net_dict": net_dict,
                        "optimizer_dict": optimizer.state_dict(),
                        "lr": rightnow_lr,
                        "cfg": cfg,
                    }
                    save_checkpoint(
                        save_model,
                        epoch,
                        save_dict=save_dict,
                        best_auc_flag=True,
                    )

                saved = True

            if current_acc > best_mean_acc:
                best_mean_acc_epoch = epoch
                best_mean_acc = current_acc
                logger.info(
                    "Mean_ACC Metric update to >>> Mean_ACC %.5f @ epoch: %d"
                    % (best_mean_acc, best_mean_acc_epoch)
                )
                if not saved:
                    save_dict = {
                        "epoch": epoch,
                        "best_loss": current_loss,
                        "best_acc": best_mean_acc,
                        "best_recall": current_recall,
                        "net_dict": net_dict,
                        "optimizer_dict": optimizer.state_dict(),
                        "lr": rightnow_lr,
                        "cfg": cfg,
                    }
                    save_checkpoint(
                        save_model,
                        epoch,
                        save_dict=save_dict,
                        best_acc_flag=True,
                    )

                saved = True

            if current_loss < best_mean_loss:
                best_mean_loss_epoch = epoch
                best_mean_loss = current_loss
                logger.info(
                    "Loss Metric update to >>> Loss: %.6f @ epoch: %d"
                    % (best_mean_loss, best_mean_loss_epoch)
                )
                if not saved:
                    save_dict = {
                        "epoch": epoch,
                        "best_loss": best_mean_loss,
                        "best_acc": current_acc,
                        "net_dict": net_dict,
                        "optimizer_dict": optimizer.state_dict(),
                        "lr": rightnow_lr,
                        "cfg": cfg,
                    }
                    save_checkpoint(
                        save_model,
                        epoch,
                        save_dict=save_dict,
                        best_loss_flag=True,
                    )

                    saved = True

            if current_recall > best_mean_recall:
                best_mean_recall_epoch = epoch
                best_mean_recall = current_recall
                logger.info(
                    "Mean_Recall Metric update to >>> Mean_Recall: %.6f @ epoch: %d"
                    % (best_mean_recall, best_mean_recall_epoch)
                )
                if not saved:
                    save_dict = {
                        "epoch": epoch,
                        "best_loss": current_loss,
                        "best_acc": current_acc,
                        "best_recall": best_mean_recall,
                        "net_dict": net_dict,
                        "optimizer_dict": optimizer.state_dict(),
                        "lr": rightnow_lr,
                        "cfg": cfg,
                    }
                    save_checkpoint(
                        save_model,
                        epoch,
                        save_dict=save_dict,
                        best_recall_flag=True,
                    )

                saved = True

            else:
                save_dict = {
                    "epoch": epoch,
                    "best_loss": best_mean_loss,
                    "best_acc": best_mean_acc,
                    "best_recall": best_mean_recall,
                    "best_f1": best_mean_f1,
                    "best_auc": best_mean_auc,
                    "net_dict": net_dict,
                    "optimizer_dict": optimizer.state_dict(),
                    "lr": rightnow_lr,
                    "cfg": cfg,
                }
                save_checkpoint(
                    save_model,
                    epoch,
                    save_dict=save_dict,
                )

            epoch_stop = time.time()
            epoch_time = epoch_stop - epoch_start
            logger.info(
                ">>> [ %s ][ epoch: %d / %d time: %.3f] [ Current Best: Loss: %.6f @ epoch: %d - Mean_ACC: %.5f @ epoch: %d Mean_Recall: %.5f @ epoch: %d Mean_F1: %.5f @ epoch: %d Mean_Auc: %.5f @ epoch: %d] <<< \n"
                % (
                    expriment,
                    epoch,
                    cfg.BASIC.Epochs,
                    epoch_time,
                    best_mean_loss,
                    best_mean_loss_epoch,
                    best_mean_acc,
                    best_mean_acc_epoch,
                    best_mean_recall,
                    best_mean_recall_epoch,
                    best_mean_f1,
                    best_mean_f1_epoch,
                    best_mean_auc,
                    best_mean_auc_epoch,
                )
            )
            print("\n\n")
            max_stop_epoch = max(
                best_mean_loss_epoch,
                best_mean_acc_epoch,
                best_mean_recall_epoch,
                best_mean_f1_epoch,
                best_mean_auc_epoch,
            )

            if epoch > cfg.BASIC.Epochs - cfg.BASIC.no_trans_epoch:
                train_dataloader.dataset.use_trans = False
                logger.info(
                    "Turn of trans for the last %d epochs" % (cfg.BASIC.no_trans_epoch)
                )
            if epoch - cfg.BASIC.Early_stop > max_stop_epoch:
                logger.info(
                    "Metric stoped for %d epochs, stop training!!"
                    % cfg.BASIC.Early_stop
                )
                break
    except KeyboardInterrupt:
        logger.info("User stop by keyboard")

    logger.info(">>> Finish Training <<<")
    logger.info(
        "[ Best: Loss: %.6f @ epoch : %d -- Mean_ACC: %.5f @ epoch: %d -- Mean_Recall: %.5f @ epoch: %d -- Mean_F1: %.5f @ epoch: %d -- Mean_Auc: %.5f @ epoch: %d] \n\n"
        % (
            best_mean_loss,
            best_mean_loss_epoch,
            best_mean_acc,
            best_mean_acc_epoch,
            best_mean_recall,
            best_mean_recall_epoch,
            best_mean_f1,
            best_mean_f1_epoch,
            best_mean_auc,
            best_mean_auc_epoch,
        )
    )
    logger.info(">>> Start Infer <<<")
    infer_results = infer(infer_path=root, logger=logger, epochs=-1)
    logger.info(">>> Finish Infer <<<")
    run.finish()
    return best_mean_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update custom params")
    parser.add_argument(
        "--nodebug",
        action="store_true",
        help="weather in debug mode, given = no debug, not given  = in debug",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--cfg_file",
        help="Path to the config file",
        default=None,
    )
    args = parser.parse_args()
    cfg = get_arguments(args)
    main(cfg)
