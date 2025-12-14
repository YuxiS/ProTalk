import numpy as np
from glob import glob 
from tqdm import tqdm
import sys
import os
import torch
import torch.nn
sys.path.append('.')
sys.path.append('..')
from metrics import frechet_video_distance
from CoeffDataset import read_video

VIDEO_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/all'
PATH_TO_MODEL_WEIGHTS = "/home/songyifei9/code/prosody/StyleProsody/mellotron/metrics/FVD/pytorch_i3d_model/models/rgb_imagenet.pt"

subdirs = ['only_mfcc', 'final_no_smooth', 'final', 'GT']
target_dir = 'real'

device = torch.device('cuda:0')

videos = glob(os.path.join(VIDEO_ROOT, target_dir, '*.mp4'))
for sub_dir in subdirs:
    res = []
    for v in tqdm(videos):
        with torch.no_grad():
            basename = os.path.basename(v)
            video_S = read_video(os.path.join(VIDEO_ROOT, sub_dir, basename))
            index = np.arange(0, len(video_S), step=(len(video_S)-1)//32)
            video_S = np.stack(video_S, axis=0)
            video_S = torch.from_numpy(video_S).unsqueeze(0).float()
            video_S = video_S[:, index, ...]
            video_T = read_video(v)
            video_T = np.stack(video_T, axis=0)
            video_T = torch.from_numpy(video_T).unsqueeze(0).float()
            video_T = video_T[:, index, ...]
            video_S = video_S.to(device)
            video_T = video_T.to(device)
            fvd_score = frechet_video_distance(video_S, video_T, PATH_TO_MODEL_WEIGHTS, device)
            res.append(fvd_score)
    print('{} : {}'.format(sub_dir, np.mean(res)))





