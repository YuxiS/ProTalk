import numpy as np
from metrics.beat_scores import bas
import pickle 
from scipy import linalg
import json
import librosa
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import pdb

import matplotlib.pyplot as plt 

music_root = '/mnt/lustre/lisiyao1/dance/dance2/DanceRevolution/data/aistpp_test_full_wav'

def get_hpss(audio):
    audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
    # print(f'{audio_percussive.shape} -> audio_percussive')
    return audio_harmonic, audio_percussive

def get_onset_beat(onset_env, sr=22050):
        onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        beats_one_hot = np.zeros(len(onset_env))
        peaks_one_hot = np.zeros(len(onset_env))
        for idx in onset_beats:
            beats_one_hot[idx] = 1
        for idx in peaks:
            peaks_one_hot[idx] = 1

        beats_one_hot = beats_one_hot.reshape(1, -1)
        peaks_one_hot = peaks_one_hot.reshape(1, -1)

        # print(f'{beats_one_hot.shape} -> beats_feature')
        return beats_one_hot, peaks_one_hot

def get_onset_strength(audio_percussive, sr):
    onset_env = librosa.onset.onset_strength(audio_percussive, aggregate=np.median, sr=sr)
    # print(f'{onset_env.reshape(1, -1).shape} -> onset_env')
    return onset_env

def get_mb(path, length=None, sr=22050):
    # path = os.path.join(music_root, key)
    audio, _ = librosa.load(path, sr)
    audio_harmonic, audio_percussive = get_hpss(audio)
    onset_env = get_onset_strength(audio_percussive, sr)

    onset_beat = get_onset_beat(onset_env, sr)[0]
    beats = onset_beat.astype(bool)[0]
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
    return beat_axis


    # with open(path) as f:
        #print(path)
        # sample_dict = json.loads(f.read())
        # if length is not None:
        #     beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        # else:
        #     beats = np.array(sample_dict['music_array'])[:, 53]
         


        # beats = beats.astype(bool)
        # beat_axis = np.arange(len(beats))
        # beat_axis = beat_axis[beats]
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')

        # print(len(beats))
        # return beat_axis


def calc_db(keypoints, name='head'):
    # 仅仅计算不包括嘴部和边缘的关点， 一共是32个, 可能会有长度的问题吧
    # pdb.set_trace()
    # keypoints = np.array(keypoints).reshape(-1, 31, 2)
    # pdb.set_trace()
    if name =='exp':
        keypoints = np.array(keypoints).reshape(-1, 1, 64)
    else:
        keypoints = np.array(keypoints).reshape(-1, 1, 6) #头部动作参数来计算
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.greater)
    return motion_beats, len(kinetic_vel)



def BA(music_beats, motion_beats):
    return bas(music_beats, motion_beats[0])

def calc_ba_score(keypoints, audio_file, mode='head'):
    motion_beats, length = calc_db(keypoints, mode)
    audio_beats = get_mb(audio_file, length)
    if len(motion_beats[0]) == 0 or len(audio_beats)==0:
        return -1
    else:
        be_score = BA(audio_beats, motion_beats)
        return be_score

# def calc_ba_score(root):

#     # gt_list = []
#     ba_scores = []

#     for pkl in os.listdir(root):
#         # print(pkl)
#         if os.path.isdir(os.path.join(root, pkl)):
#             continue
#         joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:, :]

#         dance_beats, length = calc_db(joint3d, pkl)        
#         music_beats = get_mb(pkl.split('.')[0] + '.json', length)

#         ba_scores.append(BA(music_beats, dance_beats))
        
#     return np.mean(ba_scores)

if __name__ == '__main__':
    keypoints = np.load('/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/headpose/params/prosody/W014_angry-level_3-017.npy')
    keypoints = keypoints[:, :, :6]
    keypoints = np.squeeze(keypoints, axis=0)
    keypoints = np.expand_dims(keypoints, axis=1)
    motion_beats, length = calc_db(keypoints)
    audio_beats = get_mb('/home/songyifei9/code/prosody/StyleProsody/mellotron/test/MEAD_AUDIO/W014_angry-level_3-017.wav')
    print(audio_beats)
    print(motion_beats, length)
