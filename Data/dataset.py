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
import transforms as TRANS
from multiprocessing import Pool, Manager
import multiprocessing
from typing import Callable, List, Optional, Tuple
from Lib import read_json
from functools import partial
from torchsampler import ImbalancedDatasetSampler
import copy
from loguru import logger

class HartValve(data.Dataset):
    def __init__(
        self,
        visual_size=(240,320),#witdh*height
        time_size=8,
        state="train",
        json_file_dir='/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416_fold3.json',
        databasedir = '/home/wjx/data/code/HeartValve/DatabaseLmdb',
        fold = '0',
        phase='all',
        **kwargs
    ):
        
        self.visual_size = visual_size
        self.state = state
        self.time_size = time_size
        self.fold = fold
        self.phase = phase
        self.num_classes = 2
        all_samples = read_json(json_file_dir)
        self.patientid = all_samples["split_"+self.fold]["%s" % self.state]
        self.patientid_videos = all_samples['patientid_videos']
        self.env = lmdb.open(
            os.path.join(
                databasedir,'T%02dW%03dH%03d'%(time_size,visual_size[0],visual_size[1]) # W,H
            ),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        self.train_trans = TRANS.Compose(
                [
                    TRANS.RandomErode(k=3, high=192, low=16, p=0.15),
                    TRANS.RandomDilate(k=3, high=192, low=16, p=0.15),
                    TRANS.TioClamp(clamps=(16, 192), p=0),
                    TRANS.TioRandomFlip(p=0.25),
                    TRANS.TioRandomAnisotropy(p=0.25),
                    TRANS.TioRandomMotion(p=0.25),
                    TRANS.TioRandomGhosting(p=0.25),
                    TRANS.TioRandomSpike(p=0.25),
                    TRANS.TioRandomBiasField(p=0.25),
                    TRANS.TioRandomBlur(p=0.25),
                    TRANS.TioRandomNoise(p=0.25),
                    TRANS.TioRandomGamma(p=0.25),
                    TRANS.RandomRotation(degrees=(-30, 30), fill=0, p=0.25),
                    TRANS.Crop(crop=(0.2, 0.2, 0.2, 0.2)),
                    TRANS.Resize(
                        t=self.time_size,
                        visual=(self.visual_size[1], self.visual_size[0]),
                    ),
                    TRANS.TioZNormalization(p=1, div255=True),
                ]
            )
        self.val_trans = TRANS.TioZNormalization(p=1, div255=True)

    def __len__(self):
        return len(self.patientid)

    def __getitem__(self, index):
        self.patient_name = self.patientid[index]
        self.label = self.__map_label(self.patient_name)
        self.video_names = self.patientid_videos[self.patient_name]
        gray_long_view, gray_short_view, color_long_view, color_short_view, effective_views= self.__load_arrays(
            self.video_names, time_size=self.time_size, visual_size=self.visual_size
        )
        raw_views = dict(gray_long_view = gray_long_view, gray_short_view = gray_short_view,color_long_view = color_long_view,color_short_view = color_short_view)
        effective_views_tensors = {}
        for k,v in effective_views.items():
            if v ==True and self.state=="train":
                effective_views_tensors[k] = self.train_trans(raw_views[k]).contiguous()
            else:
                effective_views_tensors[k] = self.val_trans(raw_views[k]).contiguous()


        return dict(
            raw_views = raw_views,
            effective_views= effective_views,
            effective_views_tensors = effective_views_tensors,
            label=self.label,
            patient_name=self.patient_name,
            video_names = self.video_names
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

    def __load_arrays(self, video_names, time_size, visual_size): # TODO (8, 320, 240, 3)
        gray_long_view = np.ones((3,time_size,visual_size[1],visual_size[0]),dtype = np.uint8)
        gray_short_view = np.ones((3,time_size,visual_size[1],visual_size[0]),dtype = np.uint8)
        color_long_view = np.ones((3,time_size,visual_size[1],visual_size[0]),dtype = np.uint8)
        color_short_view = np.zeros((3,time_size,visual_size[1],visual_size[0]),dtype = np.uint8)
        effective_views = dict(gray_long_view = False, gray_short_view = False,color_long_view = False,color_short_view = False)
        with self.env.begin(write=False) as txn:
            for video_name in video_names:
                try:
                    view = pickle.loads(
                        txn.get((video_name).encode())
                    )
                except Exception as e:
                    logger.error (f'{video_name} is not right in {video_names}')
                    continue
                axis = video_name.split('#')[-1]
                if 'A' == axis[0]:
                    gray_short_view =  view.transpose(3,0,1,2)
                    effective_views['gray_short_view'] = True
                elif 'B' == axis[0]:
                    color_short_view = view.transpose(3,0,1,2)
                    effective_views['color_short_view'] = True
                elif 'C' == axis[0]:
                    gray_long_view = view.transpose(3,0,1,2)
                    effective_views['gray_long_view'] = True
                elif 'D' == axis[0]:
                    color_long_view = view.transpose(3,0,1,2)
                    effective_views['color_long_view'] = True
                else:
                    logger.error("video_name {} not supported".format(video_name)) 
                    continue             
        return torch.tensor(gray_long_view,dtype = torch.float32), torch.tensor(gray_short_view,dtype = torch.float32), torch.tensor(color_long_view,dtype = torch.float32), torch.tensor(color_short_view,dtype = torch.float32),effective_views # CDHW

    def __map_label(self, file_dir):
        label_str = file_dir.split("/")[-1].split("#")[0]
        if label_str=='ABN':
            return torch.tensor(0).unsqueeze(0)
        elif label_str=='NOR':
            return torch.tensor(1).unsqueeze(0)
        else:
            assert False, "Label {} not supported".format(file_dir)


if __name__ == "__main__":
    hv = HartValve(state="train") 
    logger.info(hv.get_weighted_count())
    start = time.time()
    labels_list = [0 for i in range(2)]
    for index, datas in tqdm.tqdm(enumerate(hv)):
        labels_list[datas["label"].item()] += 1
        if index%10==0:
            logger.info(f"{datas['patient_name']},{datas['video_names']},{datas['effective_views']},{datas['effective_views_tensors']['gray_long_view'].shape},{datas['effective_views_tensors']['gray_short_view'].shape},{datas['effective_views_tensors']['color_long_view'].shape},{datas['effective_views_tensors']['color_short_view'].shape}")
    fps = len(hv) / (time.time() - start)
    logger.info (fps)
    logger.info (labels_list)
