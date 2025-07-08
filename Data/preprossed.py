import os
import glob
from loguru import logger
import numpy as np
import shutil
import lmdb
import pickle
from tqdm import tqdm
import cv2
import numpy as np


def process_video_to_array(
    input_path, logger, target_frame_count=48, target_resolution=(480, 640)
):
    # 打开输入视频
    video = cv2.VideoCapture(input_path)

    # 获取原始视频属性
    original_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算帧采样间隔
    if target_frame_count >= original_frame_count:
        frame_interval = 1  # 如果目标帧数大于等于原始帧数，每帧都取
        actual_target_frames = original_frame_count
    else:
        frame_interval = original_frame_count / target_frame_count
        actual_target_frames = target_frame_count

    # 初始化帧列表
    frames = []
    current_frame = 0
    processed_frame = 0

    logger.info(f"原始帧数: {original_frame_count}")
    logger.info(f"目标帧数: {target_frame_count}")
    logger.info(f"目标分辨率: {target_resolution[0]}x{target_resolution[1]}")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 只处理需要的帧
        if (
            current_frame >= processed_frame * frame_interval
            and processed_frame < actual_target_frames
        ):
            # 调整分辨率
            resized_frame = cv2.resize(
                frame, target_resolution, interpolation=cv2.INTER_AREA
            )
            # 添加到帧列表
            frames.append(resized_frame)
            processed_frame += 1

        current_frame += 1
        # 显示进度
        if current_frame % 100 == 0:
            logger.info(f"已处理 {current_frame}/{original_frame_count} 帧")

    # 释放资源
    video.release()

    # 如果实际帧数不足目标帧数，复制最后一帧填充
    while len(frames) < target_frame_count:
        frames.append(frames[-1].copy())

    # 转换为numpy数组
    video_array = np.array(frames)
    logger.info(
        f"处理完成！输出数组形状 - 类型 - 最大值 - 最小值 - 平均值： {video_array.shape,video_array.dtype,video_array.max(),video_array.min(),video_array.mean()}"
    )
    return video_array


target_frame_count = 8
target_resolution = (256, 320)  # width,height
databasedir = "/home/wjx/data/code/HeartValve/DatabaseLmdb/T%02dW%03dH%03d" % (
    target_frame_count,
    target_resolution[0],
    target_resolution[1],
)
os.makedirs(databasedir, exist_ok=True)
env = lmdb.open(databasedir, map_size=1099511627776)
lib = env.begin(write=True)
meta_data = {}
logger.add(r"/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416_log.log")
root = r"/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416"
videos = glob.glob(os.path.join(root, "*.avi"))
index = 0
logger.info("\n" * 2)
for video in tqdm(videos):
    name = video.split("/")[-1]
    logger.info(f">>> process {name}")
    new_video = process_video_to_array(
        video,
        logger,
        target_frame_count=target_frame_count,
        target_resolution=target_resolution,
    )
    lib.put(
        key=(name).encode(),
        value=pickle.dumps(new_video),
    )
    index += 1
    if (index + 1) % 5 == 0:
        lib.commit()
        lib = env.begin(write=True)  # commit 之后需要再次 begin
    logger.info("\n" * 2)
