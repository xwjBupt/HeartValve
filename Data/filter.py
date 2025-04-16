import os
import glob
from loguru import logger
#import mmcv
import shutil

# root = r"/home/wjx/data/dataset/Heart/cropped_raw/normal_one_video"
# prefix = "NOR"
# saved = r"/home/wjx/data/dataset/Heart/cropped_processed"
# os.makedirs(saved, exist_ok=True)
# logger.add(r"/home/wjx/data/dataset/Heart/cropped_processed/log.log")
# files = os.listdir(root)
# for file in files:
#     file_dir = os.path.join(root, file)
#     videos = glob.glob(os.path.join(file_dir, "*.avi"))
#     for video in videos:
#         try:
#             video_content = mmcv.VideoReader(video)
#             if len(video_content) == 0:
#                 logger.error(f"video_name {video} has 0 frames, please check")
#             logger.info(
#                 f"video_name {file+'#'+video.split('/')[-1]} frames {len(video_content)} -- width {video_content.width} -- height {video_content.height} resolution {video_content.resolution} fps {video_content.fps}"
#             )
#         except Exception as e:
#             logger.error(f"video_name {video} can not be read, please check")
#             continue
#         saveingname = prefix + "#" + file + "#" + video.split("/")[-1]
#         shutil.copy(video, os.path.join(saved, saveingname))
# print("DONE")



root = r"/home/wjx/data/dataset/Heart/cropped_processed"
logger.add(r"/home/wjx/data/dataset/Heart/cropped_processed/log.log")
allvs = os.listdir(root)
patient_ids = []
for i in allvs:
    if '.log' in i :
        continue
    patient_ids.append(i.split('#')[1])
patient_ids = list(set(patient_ids))
for patient_id in patient_ids:
    patient_videos = glob.glob(root+'/*%s*.avi'%patient_id)
    if len(patient_videos)>4:
        logger.error(patient_videos)



# os.makedirs(saved, exist_ok=True)
# files = os.listdir(root)
# for file in files:
#     file_dir = os.path.join(root, file)
#     videos = glob.glob(os.path.join(file_dir, "*.avi"))
#     for video in videos:
#         try:
#             video_content = mmcv.VideoReader(video)
#             if len(video_content) == 0:
#                 logger.error(f"video_name {video} has 0 frames, please check")
#             logger.info(
#                 f"video_name {file+'#'+video.split('/')[-1]} frames {len(video_content)} -- width {video_content.width} -- height {video_content.height} resolution {video_content.resolution} fps {video_content.fps}"
#             )
#         except Exception as e:
#             logger.error(f"video_name {video} can not be read, please check")
#             continue
#         saveingname = prefix + "#" + file + "#" + video.split("/")[-1]
#         shutil.copy(video, os.path.join(saved, saveingname))
# print("DONE")
