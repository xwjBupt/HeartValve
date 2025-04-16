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


class HartValve(data.Dataset):
    def __init__(
        self,
        visual_size=(240,320),#witdh*height
        time_size=8,
        state="train",
        json_file_dir='/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416_fold3.json',
        databasedir = '/home/wjx/data/code/HeartValve/Database',
        fold = '0',
        phase='all',
        **kwargs
    ):
        
        self.visual_size = visual_size
        self.state = state
        self.time_size = time_size
        self.fold = fold
        self.phase = phase
        all_samples = read_json(json_file_dir)
        self.patientid = all_samples["split_"+self.fold]["%s" % self.state]
        self.patientid_videos = all_samples['patientid_videos']
        self.env = lmdb.open(
            os.path.join(
                databasedir,'T%02dV%03dx%03d'%(time_size,visual_size[0],visual_size[1]) # W,H
            ),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if self.state=='train':
            self.trans = TRANS.Compose(
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
        else:
            self.trans = TRANS.TioZNormalization(p=1, div255=True)

    def __len__(self):
        return len(self.patientid)

    def __getitem__(self, index):
        self.name = self.patientid[index]
        self.label = self.__map_label(self.name)
        self.video_names = self.patientid_videos[self.name]
        gray_long_view, gray_short_view, color_long_view, color_short_view= self.__load_arrays(
            self.video_names, time_size=self.time_size, visual_size=self.visual_size
        )
        raw_views = dict(gray_long_view = gray_long_view, gray_short_view = gray_short_view,color_long_view = color_long_view,color_short_view = color_short_view)
        gray_long_view = self.trans(gray_long_view).contiguous()
        gray_short_view = self.trans(gray_short_view).contiguous()
        color_long_view = self.trans(color_long_view).contiguous()
        color_short_view = self.trans(color_short_view).contiguous()

        return dict(
            gray_long_view=gray_long_view,
            gray_short_view=gray_short_view,
            color_long_view=color_long_view,
            color_short_view=color_short_view,
            raw_vies = raw_views,
            label=self.label,
            name=self.name,
            video_names = self.video_names
        )

    def get_weighted_count(self):
        labels_list = [0 for i in range(5)]
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
        gray_long_view = np.ones((time_size,visual_size[1],visual_size[0],3),dtype = np.uint8)
        gray_short_view = np.ones((time_size,visual_size[1],visual_size[0],3),dtype = np.uint8)
        color_long_view = np.ones((time_size,visual_size[1],visual_size[0],3),dtype = np.uint8)
        color_short_view = np.zeros((time_size,visual_size[1],visual_size[0],3),dtype = np.uint8)
        with self.env.begin(write=False) as txn:
            for video_name in video_names:
                view = pickle.loads(
                    txn.get((video_name).encode())
                )
                axis = video_name.split('#')[-1]
                if 'A' == axis[0]:
                    gray_short_view =  view.transpose(3,0,1,2)
                elif 'B' == axis[0]:
                    color_short_view = view.transpose(3,0,1,2)
                elif 'C' == axis[0]:
                    gray_long_view = view.transpose(3,0,1,2)
                elif 'D' == axis[0]:
                    color_long_view = view.transpose(3,0,1,2)
                else:
                    assert False, "video_name {} not supported".format(video_name)                
        return torch.tensor(gray_long_view,dtype = torch.float32), torch.tensor(gray_short_view,dtype = torch.float32), torch.tensor(color_long_view,dtype = torch.float32), torch.tensor(color_short_view,dtype = torch.float32) # CDHW

    def __map_label(self, file_dir):
        label_str = file_dir.split("/")[-1].split("#")[0]
        if label_str=='ABN':
            return torch.tensor(0).unsqueeze(0)
        elif label_str=='NOR':
            return torch.tensor(1).unsqueeze(0)
        else:
            assert False, "Label {} not supported".format(file_dir)


if __name__ == "__main__":
    hv = HartValve() 
    print(hv.get_weighted_count())
    start = time.time()
    labels_list = [0 for i in range(5)]
    for index, datas in tqdm.tqdm(enumerate(hv)):
        labels_list[datas["label"].item()] += 1
        if index%10==0:
            print (datas['name'],datas['video_names'],gray_long_view.shape(),gray_short_view.shape(),color_long_view.shape(),color_short_view.shape())
    fps = len(hv) / (time.time() - start)
    print(fps)
    print(labels_list)
