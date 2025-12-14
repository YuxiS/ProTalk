import os
import shutil
import scipy.io as scio
import numpy as np
from tqdm import tqdm


data = '/home/songyifei9/code/prosody/StyleProsody/mellotron/data/data_test.txt'
coeff_root = '/home/songyifei9/data/MEAD_VIDEO/Coeff_3D'
video_root = '/home/songyifei9/data/MEAD_VIDEO/videos_256'
target_root = '/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/GT'

res = []
with open(data) as f:
    res = f.readlines()

new_res =[]
for r in res:
    new_res.append(r.split('|')[0].strip())

for r in tqdm(new_res):
    basename = os.path.basename(r)
    id = r.split(os.sep)[-2]
    # coeff = scio.loadmat(os.path.join(coeff_root, id, basename[:-4]+'.mat'))['coeff']
    # coeff_path=os.path.join(target_root, id+'_'+basename[:-4]+'.npy')
    # np.save(coeff_path, coeff)
    video = os.path.join(video_root, id, basename[:-4]+'.mp4')
    target_video_file = os.path.join(target_root, id+'_'+basename[:-4]+'.mp4')
    shutil.copy(video, target_video_file)
