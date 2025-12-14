import numpy as np
import sys
import os 
sys.path.append('.')
sys.path.append('..')
from metrics import beat_align_score
import pdb
import scipy.io as scio
from tqdm import tqdm


AUDIO_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/test/MEAD_AUDIO'
COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_exp/landmarks'


if __name__=='__main__':

    subdir = ['lstm_init', 'exp_branch', 'finetune']
    for sub in subdir:
        keypoints_root = os.path.join(COEFF_ROOT, sub)
        videos = os.listdir(os.path.join(COEFF_ROOT, sub))
        # is_video = lambda v: os.path.splitext(v)[1]=='npy' 
        # videos = [v for v in files if is_video(v)]
        bt_scores = []
        for v in tqdm(videos, leave=True):
            basename = os.path.splitext(os.path.basename(v))[0]
            audio_file = os.path.join(AUDIO_ROOT, basename+'.wav')
            keypoints = np.load(os.path.join(keypoints_root, v))
            # pdb.set_trace()
            keypoints = keypoints[:, 17:48, :]
            # keypoints = np.squeeze(keypoints, axis=0)
            # keypoints = np.expand_dims(keypoints, axis=1)
            # print(audio_file, end='\t')
            bt_score = beat_align_score.calc_ba_score(keypoints, audio_file)
            if bt_score>0:
                # print(bt_score)
                bt_scores.append(bt_score)
            # else:
            #     print(-1)
        print('{}:{}'.format(sub, np.mean(bt_score)))