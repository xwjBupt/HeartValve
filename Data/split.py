import glob
import json
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import random

def getids(names):
    patientid = []
    patientid_videos = {}
    for n in names:
        try:
            pid = n.split('#')[0]+'#'+n.split('#')[1]
            patientid.append(pid)
            if pid in patientid_videos.keys():
                patientid_videos[pid].append(n)
            else:
                patientid_videos[pid] = []
                patientid_videos[pid].append(n)
            
        except Exception as e:
            print ((n))
            continue
    patientid = list(set(patientid))
    return patientid,patientid_videos

videos = os.listdir('/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416')
output_json_save_path = '/home/wjx/data/dataset/Heart/cropped_processed_DrLiu_250416_fold3.json'
patientid,patientid_videos = getids(videos)
with open(output_json_save_path,"a+") as f:
        json.dump(patientid_videos,f)

random.shuffle(patientid)
random.shuffle(patientid)

kf = KFold(n_splits=3, shuffle=True)
splits = {}
index = 0
for train_idx, test_idx in kf.split(patientid):
    train_list, test_list = list(np.array(patientid)[train_idx]),list(np.array(patientid)[test_idx])
    new_train_list = [str(i).split('.')[0] for i in train_list]
    new_test_list = [str(i).split('.')[0] for i in test_list]

    splits['split_%d'%index]={}
    splits['split_%d'%index]['train'] = new_train_list
    splits['split_%d'%index]['test'] = new_test_list
    index+=1
    print('--------------------------------------------')
    print('# of train:', len(new_train_list))
    print('# of test:', len(new_test_list))
    print('--------------------------------------------')    
with open(output_json_save_path,"a+") as f:
        json.dump(splits,f)
print('DONE')