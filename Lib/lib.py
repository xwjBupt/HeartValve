import pickle
import os.path as osp
import glob
import tqdm
from git import Repo
import glob
import yaml
import shutil
import json
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import os
import csv
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple
from functools import partial
import torch
import cv2

# import mmcv
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Averagvalue(object):
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


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def flush(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "  [ %s ] [ avg : %.4f ] [ meidan : %.4f ]\n"
                % (name, meter.global_avg, meter.median)
            )
        return self.delimiter.join(loss_str)

    def lineout(self):
        loss_str = []
        loss_str.append("[")
        for name, meter in self.meters.items():
            loss_str.append(" %s:%.4f " % (name, meter.global_avg))
        loss_str.append("]")
        return self.delimiter.join(loss_str)

    def return_dict(self, method=None):
        output = {}
        if method:
            output.update(method)
        for name, meter in self.meters.items():
            output[name] = meter.global_avg
        return output


def get_learning_rate(optimizer):
    """return right now lr"""
    return optimizer.param_groups[0]["lr"]


def right_replace(string, old, new, max=1):
    return string[::-1].replace(old[::-1], new[::-1], max)[::-1]


def write_json(content, outfile):
    with open(outfile, "w") as f:
        json.dump(content, f)


def read_json(input_file):
    with open(input_file, "r") as f:
        content = json.load(f)
        return content


def write_pickle(content, outfile):
    f = open(outfile, "wb")
    data = {"data": content}
    pickle.dump(data, f)
    f.close()


def read_pickle(inputfile):
    f = open(inputfile, "rb")
    data = pickle.load(f, encoding="bytes")
    f.close()
    return data["data"]


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def warmup_lr_changer(maxlr, epoch, warmup_epoch=10):
    last_epoch = -1
    newlr = maxlr * math.e ** ((epoch - last_epoch - warmup_epoch) / warmup_epoch)
    if epoch >= warmup_epoch:
        newlr = maxlr

    return newlr


def nodeTodict(cfg, result={}):
    for k, v in cfg.items():
        if type(v) == type(cfg):
            temp = nodeTodict(v, {})
            result[k] = temp
        else:
            result[k] = v
    return result


def read_yaml(yaml_file):
    with open(yaml_file, encoding="utf8") as a_yaml_file:
        parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return parsed_yaml_file


def write_yaml(yaml_file, content):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)


def iter_update_dict(new_dict, raw_dict):
    for k, v in raw_dict.items():
        if type(v) == dict:
            iter_update_dict(new_dict, v)
        else:
            if k in new_dict:
                raw_dict[k] = new_dict[k]
    return raw_dict


def git_commit(
    work_dir,
    timestamp,
    levels=5,
    postfixs=[".py", ".sh"],
    commit_info="",
    debug=False,
):
    cid = "not generate"
    branch = "master"
    if not debug:
        repo = Repo(work_dir)
        toadd = []
        branch = repo.active_branch.name
        for i in range(levels):
            for postfix in postfixs:
                filename = glob.glob(work_dir + (i + 1) * "/*" + postfix)
                for x in filename:
                    if (
                        not ("play" in x)
                        and not ("local" in x)
                        and not ("Untitled" in x)
                        and not ("wandb" in x)
                    ):
                        toadd.append(x)
        index = repo.index
        index.add(toadd)
        index.commit(commit_info)
        cid = repo.head.commit.hexsha

    commit_tag = (
        commit_info
        + "\n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    record_commit_info = "\n" + "***" * 10 + "\n%s\n" % commit_tag + "***" * 10 + "\n"
    return record_commit_info


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def nodeTodict(cfg, result={}):
    for k, v in cfg.items():
        if type(v) == type(cfg):
            temp = nodeTodict(v, {})
            result[k] = temp
        else:
            result[k] = v
    return result


def read_yaml(yaml_file):
    with open(yaml_file, encoding="utf8") as a_yaml_file:
        parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return parsed_yaml_file


def write_yaml(yaml_file, content):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)


def iter_update_dict(new_dict, raw_dict):
    for k, v in raw_dict.items():
        if type(v) == dict:
            iter_update_dict(new_dict, v)
        else:
            if k in new_dict:
                raw_dict[k] = new_dict[k]
    return raw_dict


def save_checkpoint(
    save_model,
    epoch,
    save_dict,
    best_loss_flag=False,
    best_acc_flag=False,
    best_recall_flag=False,
    best_f1_flag=False,
    best_auc_flag=False,
    cover_up=True,
):
    if best_acc_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Acc_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Acc_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_recall_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Recall_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Recall_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_loss_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Loss_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Loss_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_f1_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_F1_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_F1_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_auc_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_AUC_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_AUC_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    else:
        save_name = save_model + "/checkpoint.pth"
        torch.save(save_dict, save_name)


def iter_wandb_log(wandb, metric, phase="Train", index=1):
    towrite = {"index": index}
    for k, v in metric.meters.items():
        towrite[phase + "_" + k] = v.global_avg
    wandb.log(towrite)


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)


def fuse_order_rule(
    fuse_pre, ova_pre=None, num_classes=4, less_confi=0.3, high_confi=0.85, **kwargs
):
    """
    input must be logit not tags, aka before sigmoid or softmax
    current only supports bs==1
    """
    if ova_pre is None:
        return torch.argmax(F.softmax(fuse_pre, dim=-1), dim=-1).detach().cpu()
    else:
        assert (
            fuse_pre.shape[0] == ova_pre.shape[0]
        ), "fuse_pre has different batch dim with ova_pre"
        batch = fuse_pre.shape[0]
        final_tags = []
        for b in range(batch):
            fuse_logit = F.softmax(fuse_pre[b], dim=-1).detach().cpu()
            fuse_tag = torch.argmax(fuse_logit, dim=-1)

            ova_logits = F.sigmoid(ova_pre[b]).detach().cpu()
            ### rule filter ###
            if fuse_logit[0] > high_confi and ova_logits[0] > high_confi:
                final_tags.append(0)


def imap_tqdm(
    function, iterable, processes, chunksize=1, desc=None, disable=False, **kwargs
):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    Results are always ordered and the performance should be the same as of Pool.map.
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
    :param desc: The description displayed by tqdm in the progress bar.
    :param disable: Disables the tqdm progress bar.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function_wrapper = partial(_wrapper, function=function, **kwargs)
    else:
        function_wrapper = partial(_wrapper, function=function)

    results = [None] * len(iterable)
    with Pool(processes=processes) as p:
        with tqdm.tqdm(
            desc=desc,
            total=len(iterable),
            disable=disable,
            dynamic_ncols=True,
        ) as pbar:
            for i, result in p.imap_unordered(
                function_wrapper, enumerate(iterable), chunksize=chunksize
            ):
                results[i] = result
                pbar.update()
    return results


def _wrapper(enum_iterable, function, **kwargs):
    i = enum_iterable[0]
    result = function(enum_iterable[1], **kwargs)
    return i, result


# def save_tensor_video_mmcv(video_tensor: torch.Tensor, output_path: str, fps: int = 5):
#     """
#     使用 mmcv 将视频 tensor 写入本地文件，自动判断取值范围 ([0, 1] or [0, 255])。

#     参数:
#         video_tensor (Tensor): 视频数据，形状为 (T, C, H, W)，C 必须为 3
#         output_path (str): 输出视频路径（如 'video.mp4'）
#         fps (int): 帧率，默认 25
#     """
#     assert isinstance(video_tensor, torch.Tensor), "输入必须是 torch.Tensor"
#     assert video_tensor.ndim == 4, "视频 Tensor 应为 (T, C, H, W)"
#     assert video_tensor.shape[1] == 3, "当前仅支持 RGB 三通道视频"

#     video_tensor = video_tensor.detach().cpu()

#     # 判断是否为 [0, 1] 还是 [0, 255]
#     max_val = video_tensor.max().item()
#     if max_val <= 1.0:
#         video_tensor = video_tensor * 255.0
#         print("🔍 检测到取值范围为 [0, 1]，已转换为 [0, 255]")
#     else:
#         print("🔍 检测到取值范围为 [0, 255]，无需转换")

#     # 转换为 uint8，形状 (T, H, W, C)
#     video_tensor = video_tensor.clamp(0, 255).byte()
#     video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()

#     # 获取分辨率
#     h, w = video_numpy.shape[1:3]

#     # 初始化写入器
#     writer = mmcv.VideoWriter(output_path, fps=fps, frame_size=(w, h))

#     for frame in video_numpy:
#         writer.write_frame(frame)

#     writer.release()
#     print(f"✅ 视频已保存至: {output_path}")


def save_tensor_video_cv2(tensor, save_path, fps=5):
    """
    将视频 Tensor 保存为 MP4 文件，自动适配 0-1 或 0-255 的范围。

    参数:
    - tensor: torch.Tensor 或 numpy.ndarray, 形状 [T, C, H, W] 或 [T, H, W, C]
    - save_path: 输出视频文件路径
    - fps: 帧率
    """
    # 转成 numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 确保维度是 [T, H, W, C]
    if tensor.shape[1] in [1, 3]:  # [T, C, H, W]
        tensor = np.transpose(tensor, (0, 2, 3, 1))  # 转成 [T, H, W, C]

    T, H, W, C = tensor.shape

    # 自动检测范围
    min_val, max_val = tensor.min(), tensor.max()
    if max_val <= 1.0:
        tensor = (tensor * 255.0).clip(0, 255)
    else:
        tensor = tensor.clip(0, 255)

    # 转成 uint8
    tensor = tensor.astype(np.uint8)

    # 如果是单通道灰度，转成三通道
    if C == 1:
        tensor = np.repeat(tensor, 3, axis=-1)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 使用 OpenCV 保存 MP4 视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for frame in tensor:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"✅ 视频已保存到: {save_path}")
