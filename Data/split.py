import glob
import json
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold


# 示例样本名（NOR#代表正类，ABN#代表负类）
def getids(names):
    patientid = []
    patientid_videos = {}
    for n in names:
        try:
            pid = n.split("#")[0] + "#" + n.split("#")[1]
            patientid.append(pid)
            if pid in patientid_videos.keys():
                patientid_videos[pid].append(n)
            else:
                patientid_videos[pid] = []
                patientid_videos[pid].append(n)

        except Exception as e:
            print((n))
            continue
    patientid = list(set(patientid))
    return patientid, patientid_videos


videos = os.listdir("/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416")
output_json_save_path = (
    "/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250619.json"
)
patientid, patientid_videos = getids(videos)
with open(output_json_save_path, "a+") as f:
    json.dump(patientid_videos, f)

random.shuffle(patientid)
random.shuffle(patientid)

labels = [1 if name.startswith("NOR#") else 0 for name in patientid]

# 转成numpy数组方便索引
sample_names = np.array(patientid)
labels = np.array(labels)

# 初始化 StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 保存每一折的划分结果
split_data = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sample_names, labels)):
    fold_info = {
        "fold": fold_idx,
        "train_samples": sample_names[train_idx].tolist(),
        "val_samples": sample_names[val_idx].tolist(),
    }
    split_data.append(fold_info)

# 写入 JSON 文件
with open(output_json_save_path, "w") as f:
    json.dump(split_data, f, indent=4)
