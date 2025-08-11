from yacs.config import CfgNode as CN
from collections import OrderedDict
import os
import matplotlib
import pdb
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
from loguru import logger
import numpy as np
import torch
import csv
import glob
from operator import itemgetter
import json
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# matplotlib.use('TkAgg')
# personal package
from builder import build_loader, build_model
from Lib import (
    MetricLogger,
    get_metrics,
    write_to_csv,
    write_json,
    calculate_and_save_metrics,
    plot_confusion_matrix,
)


@logger.catch()
def infer(
    infer_path, logger, state="test", device="cuda", epochs=-1, args=None, **kwargs
):
    model_paths = []
    if args is None:
        args = "NONE"
    if epochs == -1:
        model_paths = glob.glob(os.path.join(infer_path, "Model/Best_*_Epoch*.pth"))
    else:
        model_paths.append(
            glob.glob(os.path.join(infer_path, "Model/Best_*_Epoch_%04d.pth" % epochs))[
                0
            ]
        )
    logger.info(">>> Going to infer with args as {}".format(args))
    logger.info(model_paths)
    metrics = None
    name_pred_dict = None
    for mp in model_paths:
        if "Loss" in mp:
            phase_index = "BEST_LOSS"
        elif "Acc" in mp:
            if "Fuse" in mp:
                phase_index = "BEST_ACC_FUSE"
            else:
                phase_index = "BEST_ACC"
        elif "Recall" in mp:
            phase_index = "BEST_Recall"
        elif "F1" in mp:
            phase_index = "BEST_F1"
        else:
            phase_index = "BEST"

        epoch_nr = mp.split("_")[-1][:-4]
        checkpoint = torch.load(mp, map_location=torch.device("cpu"))
        cfg = checkpoint["cfg"]
        cfg.defrost()
        cfg.MODEL.Para.backbone_pretrained = None
        cfg.freeze()
        net = build_model(cfg.MODEL)
        pre_net_dict = checkpoint["net_dict"]
        raw_net_dict = net.state_dict()
        new_state_dict = OrderedDict()
        unloaded_keys = []
        for k, v in pre_net_dict.items():
            if "module" in k and k[7:] in raw_net_dict.keys():
                new_state_dict[k[7:]] = v
            elif k in raw_net_dict.keys():
                new_state_dict[k] = v
            else:
                unloaded_keys.append(k)
        raw_net_dict.update(new_state_dict)
        net.load_state_dict(raw_net_dict)
        net.to(device)
        net.eval()
        logger.info(
            "loaded pretrained weights, unloaded keys: {}".format(unloaded_keys)
        )
        logger.info("\n\n>>> Start infer %s <<<" % mp)
        csvname = None

        logger.info("\n\n >>> Infer on {} <<<".format(cfg.DATA.Val))
        infer_dataloader = build_loader(cfg.DATA.Val)
        state = "test"
        csvname = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cfg.DATA.Val.Class + "_results.csv",
        )
        tbar = tqdm(infer_dataloader, dynamic_ncols=True)
        infer_metrics_logger = MetricLogger()
        data_item = len(infer_dataloader)
        name_pred_dict = {}
        tbar = tqdm(infer_dataloader, dynamic_ncols=True)
        all_logits, all_labels = [], []
        net.eval()
        with torch.no_grad():
            for index, data in enumerate(tbar):
                label = data["label"].to(device, non_blocking=True)
                effective_views = data["effective_views"]
                effective_views_tensors = data["effective_views_tensors"]
                patient_name = data["patient_name"]
                video_names = data["video_names"]
                pred_logits = (
                    net(effective_views, effective_views_tensors, device).cpu().detach()
                )
                logit_sigmoid = F.sigmoid(pred_logits).item()
                all_logits.append(pred_logits)

                if label is not None:
                    all_labels.append(label.cpu().detach())
                    name_pred_dict[patient_name[0]] = dict(
                        name=patient_name[0],
                        videos=video_names,
                        confidence=pred_logits.item(),
                        logit=logit_sigmoid,
                        pred_label=0 if logit_sigmoid < 0.5 else 1,
                        gt_label=label.cpu().detach().item(),
                    )
                else:
                    name_pred_dict[patient_name[0]] = dict(
                        name=patient_name[0],
                        videos=video_names,
                        confidence=pred_logits.item(),
                        logit=F.sigmoid(pred_logits).item(),
                        pred_label=0 if logit < 0.5 else 1,
                    )
            logits = torch.cat(all_logits)
            if label is not None:
                labels = torch.cat(all_labels).numpy()
                metrics, cm = calculate_and_save_metrics(
                    labels,
                    logits,
                    epoch_nr,
                    log_dir=infer_path,
                    phase="test",
                    num_classes=2,
                    threshold=0.5,
                )
                if csvname:
                    write_to_csv(
                        csvname,
                        dict(
                            METHOD=mp,
                            ACC=metrics.get("acc"),
                            RECALL=metrics.get("recall"),
                            F1=metrics.get("f1"),
                            AUC=metrics.get("auc"),
                            AP=metrics.get("ap"),
                        ),
                    )
                    logger.info(
                        ">>> Infer {} \n on state {} Results: {} \n write result to {} <<< \n".format(
                            mp,
                            state,
                            metrics,
                            csvname,
                        )
                    )
                else:
                    logger.info(
                        ">>> Infer {} \n on state {} Results: {} \n".format(
                            mp,
                            state,
                            metrics,
                        )
                    )

            write_json(
                name_pred_dict,
                outfile=infer_path
                + "/Epoch_%s#%s_%s_results.json" % (epoch_nr, phase_index, state),
            )
            logger.info(
                ">>> Finish infer {} write result to {} <<<\n\n".format(
                    mp,
                    infer_path
                    + "/Epoch_%s#%s_%s_results.json" % (epoch_nr, phase_index, state),
                )
            )
    return metrics, name_pred_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer your model")
    parser.add_argument(
        "--infer_path",
        default="/home/wjx/data/code/HeartValve/output_runs/HartValve/PVFNet/T08H320W256/2025-0809-1637#BCELogitLoss-Pretrianed-BackX3D-Seed3490-Fold0/",  # 'Vessel/ToTest/UNetWithAttetnion/02_19-17_52/',
        help="the path of experiments to infer",
        type=str,
    )
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--state", default="test", type=str)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--ensemble",
        default=True,
        help="save the raw output(before sigmoid or softmax) for later ensemble",
    )
    args = parser.parse_args()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    logger.add(args.infer_path + "/output.log")

    metrics = infer(
        infer_path=args.infer_path,
        logger=logger,
        state=args.state,
        epochs=args.epochs,
        args=args,
    )
