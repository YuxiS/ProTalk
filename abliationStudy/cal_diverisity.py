import numpy  as np
import os
from tqdm import tqdm
from scipy.io import loadmat
import pdb 
# from dtaidistance import dtw

np.random.seed(0)
AUDIO_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/test/MEAD_AUDIO'
COEFF_ROOT = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia'
EXP_COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_style/params'

if __name__=='__main__':
    
    # subdir = ['GT','MFCC', 'prosody','finetune_style', 'finetune_style_smooth']
    # subdir = ['keypoints']
    # subdir = ['params']
    subdir = ['head_nopro', 'head_rnn', 'head_sample']
    for sub in subdir:
        keypoints_root = os.path.join(COEFF_ROOT, sub, 'coeff')
        videos = os.listdir(os.path.join(COEFF_ROOT, sub, 'coeff'))
        # is_video = lambda v: os.path.splitext(v)[1]=='npy' 
        # videos = [v for v in files if is_video(v)]
        variance = []
        for v in tqdm(videos):
            basename = os.path.splitext(os.path.basename(v))[0]
            # pdb.set_trace()
            if v.endswith('.npy'):
                keypoints = np.load(os.path.join(keypoints_root, v))
                keypoints = np.squeeze(keypoints, axis=0)
            if v.endswith('.mat'):
                keypoints = loadmat(os.path.join(keypoints_root, v))['coeff']
            angle = keypoints[:, 224:227]
            trans = keypoints[:, 254:]
            keypoints = np.concatenate((angle, trans), axis=1)
            v = np.var(keypoints, axis=0)
            variance.append(v.mean())
        print('{}:{}'.format(sub, np.mean(variance)))
"""GT:0.004 MFCC:0.16 Prosody:0.276 fintune_sytle:0.0774  finetune_smooth:0.06"""

# def sliding_window_average(seq, window_size):
#     # seq = [T, C]
#     res = []
#     for i in range(0, seq.shape[0], window_size):
#         window = seq[i:min(i+window_size, seq.shape[0])]
#         average = np.mean(window, axis=0)
#         res.append(average)
#     res = np.stack(res, axis=0)
#     return res

# def warping_distance(seq1, seq2):
#     seq1 = sliding_window_average(seq1, window_size=3)
#     seq2 = sliding_window_average(seq2, window_size=3)
#     warping_dis = []
#     for d in range(seq2.shape[1]):
#         dis = dtw.distance(seq1[:, d], seq2[:, d])
#         warping_dis.append(dis)
#     return np.sum(warping_dis)

# def random_pairs(length, pairs_num=500):
#     seq = np.arange(length)
#     indices = []
#     while len(indices) < pairs_num:
#         pair = np.random.choice(seq, size=(2, ))
#         # pdb.set_trace()
#         if pair[0] == pair[1]:
#             continue
#         else:
#             indices.append(pair)
#     return indices

# if __name__=='__main__':
    
#     # subdir = ['GT','MFCC', 'prosody','finetune_style', 'finetune_style_smooth']
#     # subdir = ['GT','no_style', 'with_style','finetune', 'finetune_smooth']
#     # subdir = 
#     for sub in subdir:
#         keypoints_root = os.path.join(EXP_COEFF_ROOT, sub)
#         videos = os.listdir(os.path.join(EXP_COEFF_ROOT, sub))
#         # is_video = lambda v: os.path.splitext(v)[1]=='npy' 
#         # videos = [v for v in files if is_video(v)]
#         indices = random_pairs(len(videos), pairs_num=300)
#         warp_distance = []
#         for index in tqdm(indices):
#             i, j = index[0], index[1]
#             # basename = os.path.splitext(os.path.basename(videos[i]))[0]
#             keypoints_i = np.load(os.path.join(keypoints_root, videos[i]))[:, :64]
#             keypoints_j = np.load(os.path.join(keypoints_root, videos[j]))[:, :64]
#             # keypoints_i = np.squeeze(keypoints_i, axis=0)
#             # keypoints_j = np.squeeze(keypoints_j, axis=0)
#             dis = warping_distance(keypoints_i, keypoints_j)
#             warp_distance.append(dis)
#         print('{}:{}'.format(sub, np.mean(warp_distance)))

        

