import numpy as np
import sys
import os 
sys.path.append('.')
sys.path.append('..')
from metrics import beat_align_score
import pdb
import scipy.io as scio
from tqdm import tqdm
import glob


AUDIO_ROOT = '/remote-home/share/yfsong/MEAD_VIDEO/wav'
COEFF_ROOT = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia/'
# EXP_COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_style/params'

if __name__=='__main__':

    # subdir = ['GT','MFCC', 'prosody','finetune_style', 'finetune_style_smooth']
    # subdir =['param']
    subdir = ['head_sample', 'head_nopro', 'head_rnn']
    for sub in subdir:
        keypoints_root = os.path.join(COEFF_ROOT, sub)
        # videos = os.listdir(os.path.join(COEFF_ROOT, sub))
        videos = glob.glob(os.path.join(COEFF_ROOT, sub, 'coeff', '*.npy'))
        # is_video = lambda v: os.path.splitext(v)[1]=='npy' 
        # videos = [v for v in files if is_video(v)]
        bt_scores = []
        for v in tqdm(videos):
            basename = os.path.splitext(os.path.basename(v))[0]
            ###################### MEAD ###########################
            id, name = basename.split('_')[0],  basename[5:]
            audio_file = os.path.join(AUDIO_ROOT, id, name+'.wav')
            ####################### CREMA-D ############################
            # audio_file = os.path.join(AUDIO_ROOT, basename+'.wav')
            # ####################### RAVEDSS ############################
            # actor_id = basename.split('-')[-1]
            # audio_file = os.path.join(AUDIO_ROOT, 'Actor_'+str(actor_id)+'-'+basename+'.wav')
            # pdb.set_trace()
            if not os.path.exists(audio_file):
                continue
            
            keypoints = np.load(v)[0]
            angle = keypoints[:, 224:227]
            trans = keypoints[:, 254:]
            keypoints = np.concatenate((angle, trans), axis=1)/2
            # keypoints = keypoints[:, :, :6]
            # keypoints = np.squeeze(keypoints, axis=0)
            keypoints = np.expand_dims(keypoints, axis=1)
            # print(audio_file, end='\t')
            bt_score = beat_align_score.calc_ba_score(keypoints, audio_file)
            if bt_score>0:
                # print(bt_score)
                bt_scores.append(bt_score)
            # else:
            #     print(-1)
        # print(bt_score)
        print('{}:{}'.format(sub, np.mean(bt_scores)))

    # subdir = ['GT','no_style', 'with_style','finetune', 'finetune_smooth']
    # subdir =['head-rnn']
    # for sub in subdir:
    #     keypoints_root = os.path.join(EXP_COEFF_ROOT, sub)
    #     videos = os.listdir(os.path.join(EXP_COEFF_ROOT, sub))
    #     # is_video = lambda v: os.path.splitext(v)[1]=='npy' 
    #     # videos = [v for v in files if is_video(v)]
    #     bt_scores = []
    #     for v in tqdm(videos):
    #         basename = os.path.splitext(os.path.basename(v))[0]
            # audio_file = os.path.join(AUDIO_ROOT, basename+'.wav')
            # keypoints = np.load(os.path.join(keypoints_root, v))
    #         # keypoints = keypoints[:, :, :6]
    #         # keypoints = np.squeeze(keypoints, axis=0)
    #         keypoints = np.expand_dims(keypoints, axis=1)
    #         # print(audio_file, end='\t')
    #         bt_score = beat_align_score.calc_ba_score(keypoints, audio_file)
    #         if bt_score>0:
    #             # print(bt_score)
    #             bt_scores.append(bt_score)
    #         # else:
    #         #     print(-1)
    #     print('{}:{}'.format(sub, np.mean(bt_scores)))
    

    # keypoints_root = os.path.join(COEFF_ROOT, 'MFCC')
    # videos = os.listdir(keypoints_root)
    # COEFF_ROOT = '/home/songyifei9/data/MEAD_VIDEO/Coeff_3D'
    # GT_root = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/headpose/params/GT' 
    # for v in tqdm(videos):
    #     basename = os.path.splitext(v)[0]
    #     id = basename.split('_')[0]
    #     file = basename[5:]
    #     coeff_file = os.path.join(COEFF_ROOT, id, file+'.mat')
    #     coeff = scio.loadmat(coeff_file)['coeff']
    #     coeff_data = np.concatenate([coeff[:, 224:227], coeff[:, 254:]], axis=1)
    #     coeff_data = np.expand_dims(coeff_data, axis=0) #[1, T, 6]
    #     np.save(os.path.join(GT_root, basename+'.npy'), coeff_data)
