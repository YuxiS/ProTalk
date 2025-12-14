import numpy as np
import sys
import os 
sys.path.append('.')
sys.path.append('..')
# from metrics import beat_align_score
import pdb
import scipy.io as scio
from tqdm import tqdm


AUDIO_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/test/MEAD_AUDIO'
COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_style/landmarks'


if __name__=='__main__':
    target = 'GT' 
    subdir = ['no_style', 'with_style', 'with_style_smooth', 'finetune', 'finetune_smooth']
    target_landmarks = os.listdir(os.path.join(COEFF_ROOT, target))
    for sub in subdir:
        lmd_list = []
        for l in tqdm(target_landmarks):
            source_keypoints = np.load(os.path.join(COEFF_ROOT, sub, l))[:, 48:68, :]
            # source_keypoints = source_keypoints.astype(np.int)
            target_keypoints = np.load(os.path.join(COEFF_ROOT, target, l))[:, 48:68, :]
            # target_keypoints = target_keypoints.astype(np.int)
            lmd = np.abs(target_keypoints, source_keypoints)
            lmd_list.append(lmd.mean())
        print('{}:{}'.format(sub, np.array(lmd_list).mean()))
    