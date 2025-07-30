from PIL.Image import Image
import matplotlib
import glob
import torch
import tqdm
import pickle
import os
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
import numpy as np
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import lmdb
import time
import torch.nn.functional as F
import torchvision.transforms as tvt
import Data.transforms as TRANS
from multiprocessing import Pool, Manager
import multiprocessing
from typing import Callable, List, Optional, Tuple
from Lib import read_json, save_tensor_video_cv2
from functools import partial
from torchsampler import ImbalancedDatasetSampler
import copy
from loguru import logger
from collections import Counter
import cv2
import kornia.augmentation as K
import albumentations as A


class VideoAugment(torch.nn.Module):
    def __init__(
        self,
        augment_config,
        target_frames=8,
        image_size=(224, 2224),
        mode="train",
        rand_n=3,
    ):
        super().__init__()
        self.augment_config = augment_config
        self.target_frames = target_frames
        self.image_size = image_size
        self.mode = mode
        self.rand_n = rand_n  # RandAugment: 每个batch随机选N个增强

        # Kornia增强
        self.flip_aug = K.RandomHorizontalFlip(p=1.0)
        self.affine_aug = K.RandomAffine(degrees=10, translate=0.1, p=1.0)
        self.rotate_aug = K.RandomRotation(degrees=15.0, p=1.0)
        self.color_aug = K.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0
        )
        self.noise_aug = K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0)
        self.erase_aug = K.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=1.0)
        self.resize_aug = K.Resize(image_size)

        # Albumentations增强
        self.albu_aug = A.Compose(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
            ]
        )

    def salt_and_pepper(self, video, prob=0.02):
        mask = torch.rand_like(video)
        video = video.clone()
        video[mask < prob / 2] = 0.0
        video[mask > 1 - prob / 2] = 1.0
        return video

    def drop_random_frames(self, video, max_drop=3):
        T = video.size(0)
        drop_count = random.randint(1, max_drop)
        drop_idx = sorted(random.sample(range(T), drop_count))
        keep = [i for i in range(T) if i not in drop_idx]
        video = video[keep]
        video = F.interpolate(
            video.unsqueeze(0),
            size=(self.target_frames, video.size(2), video.size(3)),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return video

    def temporal_crop(self, video):
        """随机时间裁剪"""
        T = video.size(0)
        crop_len = random.randint(int(0.6 * self.target_frames), self.target_frames)
        if T <= crop_len:
            start = 0
        else:
            start = random.randint(0, T - crop_len)
        video = video[start : start + crop_len]
        if video.size(0) < self.target_frames:
            repeat_count = self.target_frames // video.size(0) + 1
            video = video.repeat(repeat_count, 1, 1, 1)[: self.target_frames]
        return video

    def speed_change(self, video):
        """加速或减速视频"""
        T = video.size(0)
        factor = random.choice([0.5, 0.75, 1.25, 1.5])  # 速度系数
        new_T = int(T * factor)
        video = F.interpolate(
            video.unsqueeze(0),
            size=(new_T, video.size(2), video.size(3)),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        if new_T > self.target_frames:
            start = random.randint(0, new_T - self.target_frames)
            video = video[start : start + self.target_frames]
        else:
            repeat_count = self.target_frames // new_T + 1
            video = video.repeat(repeat_count, 1, 1, 1)[: self.target_frames]
        return video

    def forward(self, video, params=None):
        if video.max() > 1.0:
            video = video / 255.0

        if self.mode == "test":
            video = self.resize_aug(video)
            return video.clamp(0, 1)

        # RandAugment: 只保留 N 个 True
        if self.rand_n and params is not None:
            keys = list(params.keys())
            active = random.sample(keys, min(self.rand_n, len(keys)))
            for k in keys:
                params[k] = k in active

        if params["do_flip"]:
            video = self.flip_aug(video)
        if params["do_affine"]:
            video = self.affine_aug(video)
        if params["do_rotate"]:
            video = self.rotate_aug(video)
        if params["do_color"]:
            video = self.color_aug(video)
        if params["do_noise"]:
            video = self.noise_aug(video)
        if params["do_erase"]:
            video = self.erase_aug(video)

        video = self.resize_aug(video).squeeze(0)

        if params["do_temporal_crop"]:
            video = self.temporal_crop(video)
        if params["do_speed"]:
            video = self.speed_change(video)
        if params["do_reverse"]:
            video = torch.flip(video, dims=[0])
        if params["do_drop"]:
            video = self.drop_random_frames(video)
        if params["do_saltpepper"]:
            video = self.salt_and_pepper(video)

        if (
            params["do_motion_blur"]
            or params["do_gaussian_blur"]
            or params["do_brightness"]
        ):
            v_np = video.permute(0, 2, 3, 1).cpu().numpy()
            frames = [self.albu_aug(image=f)["image"] for f in v_np]
            v_np = np.stack(frames)
            video = torch.from_numpy(v_np).permute(0, 3, 1, 2).float()

        return video.clamp(0, 1)


class HartValve(data.Dataset):
    def __init__(
        self,
        visual_size=(320, 256),  # witdh*height
        time_size=8,
        augment_config={
            "horizontal_flip": 0.5,
            "affine": 0.3,
            "rotate": 0.3,
            "color_jitter": 0.5,
            "gaussian_noise": 0.3,
            "erase": 0.3,
            "temporal_crop": 0.5,
            "speed_change": 0.5,
            "reverse": 0.2,
            "drop_frames": 0.3,
            "saltpepper": 0.2,
            "motion_blur": 0.3,
            "gaussian_blur": 0.3,
            "brightness_contrast": 0.3,
        },
        state="train",
        json_file_dir="/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250619_fold3.json",
        databasedir="/home/wjx/data/code/HeartValve/DatabaseLmdb",
        fold="0",
        phase="all",
        batch_size=4,
        **kwargs,
    ):

        self.visual_size = visual_size
        self.state = state
        self.time_size = time_size
        self.fold = fold
        self.batch_size = batch_size
        self.phase = phase
        self.num_classes = 2
        all_samples = read_json(json_file_dir)
        self.patientid = all_samples["split_" + self.fold]["%s" % self.state]
        self.patientid_videos = all_samples["patientid_videos"]
        self.env = lmdb.open(
            os.path.join(
                databasedir,
                "T%02dH%03dW%03d" % (time_size, visual_size[0], visual_size[1]),  # H,W
            ),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.augment_config = augment_config
        self.augmenter = VideoAugment(
            augment_config, time_size, visual_size, mode=state
        )
        self.batch_params = None

    def __len__(self):
        return len(self.patientid)

    def __getitem__(self, index):
        if index % self.batch_size == 0:
            self.batch_params = self.__sample_batch_params()

        self.patient_name = self.patientid[index]
        self.label = self.__map_label(self.patient_name)
        self.video_names = self.patientid_videos[self.patient_name]
        (
            gray_long_view,
            gray_short_view,
            color_long_view,
            color_short_view,
            effective_views,
        ) = self.__load_arrays(
            self.video_names, time_size=self.time_size, visual_size=self.visual_size
        )
        raw_views = dict(
            gray_long_view=gray_long_view,
            gray_short_view=gray_short_view,
            color_long_view=color_long_view,
            color_short_view=color_short_view,
        )
        effective_views_tensors = {}
        for k, v in raw_views.items():
            effective_views_tensors[k] = (
                self.augmenter(v.transpose(0, 1), params=self.batch_params)
                .contiguous()
                .transpose(0, 1)
            )

        return dict(
            raw_views=raw_views,
            effective_views=effective_views,
            effective_views_tensors=effective_views_tensors,
            label=self.label,
            patient_name=self.patient_name,
            video_names=self.video_names,
        )

    def get_weighted_count(self):
        labels_list = [0 for i in range(self.num_classes)]
        for sample in self.patientid:
            labels_list[self.__map_label(sample).item()] += 1
        labels_list = [i for i in labels_list if i]
        weighted_list = [
            len(self.patientid) / labels_list[self.__map_label(i).item()]
            for i in self.patientid
        ]
        self.labels_list = labels_list
        return weighted_list

    def __sample_batch_params(self):
        return {
            "do_flip": random.random() < self.augment_config.get("horizontal_flip", 0),
            "do_affine": random.random() < self.augment_config.get("affine", 0),
            "do_rotate": random.random() < self.augment_config.get("rotate", 0),
            "do_color": random.random() < self.augment_config.get("color_jitter", 0),
            "do_noise": random.random() < self.augment_config.get("gaussian_noise", 0),
            "do_erase": random.random() < self.augment_config.get("erase", 0),
            "do_temporal_crop": random.random()
            < self.augment_config.get("temporal_crop", 0),
            "do_speed": random.random() < self.augment_config.get("speed_change", 0),
            "do_reverse": random.random() < self.augment_config.get("reverse", 0),
            "do_drop": random.random() < self.augment_config.get("drop_frames", 0),
            "do_saltpepper": random.random() < self.augment_config.get("saltpepper", 0),
            "do_motion_blur": random.random()
            < self.augment_config.get("motion_blur", 0),
            "do_gaussian_blur": random.random()
            < self.augment_config.get("gaussian_blur", 0),
            "do_brightness": random.random()
            < self.augment_config.get("brightness_contrast", 0),
        }

    def __load_arrays(
        self, video_names, time_size, visual_size
    ):  # TODO (8, 320, 256, 3)
        gray_long_view = np.ones(
            (3, time_size, visual_size[0], visual_size[1]), dtype=np.uint8
        )
        gray_short_view = np.ones(
            (3, time_size, visual_size[0], visual_size[1]), dtype=np.uint8
        )
        color_long_view = np.ones(
            (3, time_size, visual_size[0], visual_size[1]), dtype=np.uint8
        )
        color_short_view = np.zeros(
            (3, time_size, visual_size[0], visual_size[1]), dtype=np.uint8
        )
        effective_views = dict(
            gray_long_view=False,
            gray_short_view=False,
            color_long_view=False,
            color_short_view=False,
        )
        with self.env.begin(write=False) as txn:
            for video_name in video_names:
                try:
                    view = pickle.loads(txn.get((video_name).encode()))
                except Exception as e:
                    logger.error(f"{video_name} is not right in {video_names}")
                    continue
                axis = video_name.split("#")[-1]
                if "A" == axis[0]:
                    gray_short_view = view.transpose(3, 0, 1, 2)
                    effective_views["gray_short_view"] = True
                elif "B" == axis[0]:
                    color_short_view = view.transpose(3, 0, 1, 2)
                    effective_views["color_short_view"] = True
                elif "C" == axis[0]:
                    gray_long_view = view.transpose(3, 0, 1, 2)
                    effective_views["gray_long_view"] = True
                elif "D" == axis[0]:
                    color_long_view = view.transpose(3, 0, 1, 2)
                    effective_views["color_long_view"] = True
                else:
                    logger.error("video_name {} not supported".format(video_name))
                    continue
        return (
            torch.tensor(gray_long_view, dtype=torch.float32),
            torch.tensor(gray_short_view, dtype=torch.float32),
            torch.tensor(color_long_view, dtype=torch.float32),
            torch.tensor(color_short_view, dtype=torch.float32),
            effective_views,
        )  # CDHW

    def __map_label(self, file_dir):
        label_str = file_dir.split("/")[-1].split("#")[0]
        if label_str == "ABN":
            return torch.tensor(0).unsqueeze(0)
        elif label_str == "NOR":
            return torch.tensor(1).unsqueeze(0)
        else:
            assert False, "Label {} not supported".format(file_dir)


if __name__ == "__main__":
    hv = HartValve(state="test", batch_size=4)
    logger.info(hv.get_weighted_count())
    start = time.time()
    for index, datas in tqdm.tqdm(enumerate(hv)):
        print(datas)
    # labels_list = [0 for i in range(2)]
    # for index, datas in tqdm.tqdm(enumerate(hv)):
    #     labels_list[datas["label"].item()] += 1
    #     if index%10==0:
    #         logger.info(f"{datas['patient_name']},{datas['video_names']},{datas['effective_views']},{datas['effective_views_tensors']['gray_long_view'].shape},{datas['effective_views_tensors']['gray_short_view'].shape},{datas['effective_views_tensors']['color_long_view'].shape},{datas['effective_views_tensors']['color_short_view'].shape}")
    # fps = len(hv) / (time.time() - start)
    # logger.info (fps)
    # logger.info (labels_list)
