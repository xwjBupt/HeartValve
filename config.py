from yacs.config import CfgNode as CN
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
import os

num_classes = 2

_C = CN()

_C.Dist = False
_C.Local_rank = 0


_C.BASIC = CN()
_C.BASIC.Commit_Info = "Baseline"
_C.BASIC.Early_stop = 70
_C.BASIC.Epoch_dis = 15
_C.BASIC.Epochs = 300
_C.BASIC.DEBUG = False
_C.BASIC.Finetune = False
_C.BASIC.Lr_decay = 15
_C.BASIC.Num_gpus = "2"
_C.BASIC.Resume = False
_C.BASIC.Seed = 3490
_C.BASIC.Use_wandb = True
_C.BASIC.Warmup_epoch = 20
_C.BASIC.no_trans_epoch = 15
_C.BASIC.Num_classes = num_classes


_C.DATA = CN()
_C.DATA.Train = CN()
_C.DATA.Train.Class = "HartValve"
_C.DATA.Train.DataPara = CN()
_C.DATA.Train.DataPara.name = "HartValve"
_C.DATA.Train.DataPara.state = "train"
_C.DATA.Train.DataPara.json_file_dir = (
    "/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250619_fold3.json"
)
_C.DATA.Train.DataPara.time_size = 8
_C.DATA.Train.DataPara.fold = "2"
_C.DATA.Train.DataPara.visual_size = (320, 256)
_C.DATA.Train.LoaderPara = CN()
_C.DATA.Train.LoaderPara.batch_size = 1
_C.DATA.Train.LoaderPara.num_workers = 2

_C.DATA.Val = CN()
_C.DATA.Val.Class = _C.DATA.Train.Class
_C.DATA.Val.DataPara = CN()
_C.DATA.Val.DataPara.name = _C.DATA.Train.DataPara.name
_C.DATA.Val.DataPara.fold = _C.DATA.Train.DataPara.fold
_C.DATA.Val.DataPara.state = "test"
_C.DATA.Val.DataPara.time_size = _C.DATA.Train.DataPara.time_size
_C.DATA.Val.DataPara.visual_size = _C.DATA.Train.DataPara.visual_size
_C.DATA.Val.LoaderPara = CN()
_C.DATA.Val.LoaderPara.batch_size = 1
_C.DATA.Val.LoaderPara.num_workers = 2


_C.OPT = CN()
_C.OPT.Name = "AdamW"
_C.OPT.Trans_scaler = 10
_C.OPT.Para = CN()
_C.OPT.Para.lr = 0.0003  # 0.0003
_C.OPT.Para.weight_decay = 0.01

_C.SCHEDULER = CN()
_C.SCHEDULER.Name = "CosineAnnealingWarmRestarts"
_C.SCHEDULER.Para = CN()
_C.SCHEDULER.Para.T_0 = 35
_C.SCHEDULER.Para.T_mult = 2
_C.SCHEDULER.Para.eta_min = 1.0e-6

# _C.LOSS = CN()
# _C.LOSS.Name = "LabelSmoothingCrossEntropy"
# _C.LOSS.Para = CN()
# _C.LOSS.Para.balance = [
#     [0.8, 1.2],
#     [0.8, 1.2],
#     [0.8, 1.2],
#     [0.4, 0.6],
#     [0.4, 0.6],
#     [0.2, 0.3],
#     [0.2, 0.3],
#     [0.1, 0.15],
#     [0.1, 0.15],
# ]
# _C.LOSS.Para.smoothing = 0.4
# _C.LOSS.Para.keep_rate = 0.55
# _C.LOSS.Para.num_classes = num_classes
# _C.LOSS.Para.p = 0.8
# _C.LOSS.Para.q = 1.0
# _C.LOSS.Para.eps = 1e-2

_C.LOSS = CN()
_C.LOSS.Name = "BCELogitLoss"
_C.LOSS.Para = CN()

_C.MODEL = CN()
_C.MODEL.Name = "PVFNet"
_C.MODEL.Para = CN()
_C.MODEL.Para.input_clip_length = _C.DATA.Train.DataPara.time_size
_C.MODEL.Para.input_crop_size = _C.DATA.Train.DataPara.visual_size
_C.MODEL.Para.use_marc = False
_C.MODEL.Para.model_num_class = num_classes
_C.MODEL.Para.use_fusion = "StrongRegularizedBinaryHead"
_C.MODEL.Para.mlp_dropout_rate = 0
_C.MODEL.Para.num_heads = 8
_C.MODEL.Para.expand_dim = 8
_C.MODEL.Para.deep_super = [False, True, False, True]
_C.MODEL.Para.backbone_pretrained = (
    "/home/wjx/data/code/HeartValve/Src/X3D_XS-Kinect.pyth"
)
_C.MODEL.Para.depth_factor = 2.2
_C.MODEL.Para.stage_depths = [1, 2, 3, 3]  # 1253


_C.METHOD = CN()
_C.METHOD.Detail_Desc = (
    "Pretrianed-BackX3D-Seed3490-Fold%s" % _C.DATA.Train.DataPara.fold
)
_C.METHOD.Name = _C.MODEL.Name
_C.METHOD.Desc = "T%02dH%03dW%03d/%s-%s" % (
    _C.DATA.Train.DataPara.time_size,
    _C.DATA.Train.DataPara.visual_size[0],
    _C.DATA.Train.DataPara.visual_size[1],
    _C.LOSS.Name,
    _C.METHOD.Detail_Desc,
)
