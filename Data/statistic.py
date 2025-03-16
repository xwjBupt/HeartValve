import os
import glob
from loguru import logger
import mmcv
import numpy as np
import shutil

logger.add(r"/home/wjx/data/dataset/Heart/cropped_processed/log.log")
root = r"/home/wjx/data/dataset/Heart/cropped_processed"
videos = glob.glob(os.path.join(root, "*.avi"))
frames = []
fps = []
widht = []
height = []
A = []
B = []
C = []
D = []

for video in videos:
    video_content = mmcv.VideoReader(video)
    frames.append(len(video_content))
    widht.append(video_content.width)
    height.append(video_content.height)
    fps.append(int(video_content.fps))
    if '#A1.' in video:
       A.append(video) 
    elif '#B1.' in video:
        B.append(video) 
    elif '#C1.' in video:
        C.append(video) 
    elif '#D1.' in video:
        D.append(video) 
    else:
        logger.error('video named with %s is not right'%video)

frames = np.array(frames)
fps = np.array(fps)
widht = np.array(widht)
height = np.array(height)
logger.info('\n'*3)
logger.info('##################### statistics of dataset #####################')
logger.info(f"Frames mean {np.mean(frames)} median {np.median(frames)} mode {np.argmax(np.bincount(frames))}")
logger.info(f"FPS(int) mean {np.mean(fps)} median {np.median(fps)} mode {np.argmax(np.bincount(fps))}")
logger.info(f"WIDTH mean {np.mean(widht)} median {np.median(widht)} mode {np.argmax(np.bincount(widht))}")
logger.info(f"HEIGHT mean {np.mean(height)} median {np.median(height)} mode {np.argmax(np.bincount(height))}")
logger.info(f'{len(A)} A1 - {len(B)} B1 - {len(C)} C1 - {len(D)} D1')
logger.info('##################### statistics of dataset #####################')

