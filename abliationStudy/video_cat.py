from moviepy.editor import * 
import os.path as osp
from glob import glob
import cv2
import pdb
import numpy as np
import sys
import os
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
from CoeffDataset import read_video
from face_utils.utils import write_video

def merge(audio_file, video_file, video_name, video_root):
    audio = AudioFileClip(audio_file)
    video = VideoFileClip(video_file)
    # pdb.set_trace()
    if audio.duration > video.duration:
        audio = audio.set_duration(video.duration)
        # final_video = concatenate_videoclips([video, audio], method="compose")
    else:
        video = video.set_duration(audio.duration)
    final_video = video.set_audio(audio)
        # pdb.set_trace()
        # return final_video
    final_video.write_videofile(os.path.join(video_root, video_name), fps=30)

def cat_videos(videofiles):
    # videos: [v_file, v_file]
    videos = []
    for v_f in videofiles:
        videos.append(read_video(v_f))
    video_length = len(videos[0])
    frames = []
    for i in range(video_length):
        temp = []
        for v in videos: 
            temp.append(v[i])
        frame = np.concatenate(temp, axis=1)
        frames.append(frame)
    frames = np.stack(frames, axis=0)
    return frames

if __name__=='__main__':
    AUDIO_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/test/MEAD_AUDIO'
    # COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/no_headmotion/'
    # HEADPOSE_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/headpose'
    # HEADPOSE_RES_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/headpose/catted_video'
    # RES_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/no_headmotion/cat_videos'
    HEADPOSE_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_style'
    HEADPOSE_RES_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_style/ca_video'
    subdir = ['GT', 'no_style', 'with_style', 'with_style_smooth']
    # subdir = ['GT','lstm_init', 'facial_exp', 'finetune']
    # subdir = ['real', 'MFCC', 'prosody', 'finetune', 'finetune_smooth']
    videofiles = os.listdir(os.path.join(HEADPOSE_ROOT, subdir[0]))
    for v in tqdm(videofiles):
        videos = []
        for sub in subdir:
            videos.append(os.path.join(HEADPOSE_ROOT, sub, v))
        basename = basename = os.path.splitext(os.path.basename(v))[0]
        audio_file = os.path.join(AUDIO_ROOT, basename+'.wav')
        catted_video = cat_videos(videos)
        write_video(osp.join(HEADPOSE_RES_ROOT, v), catted_video)
        # merge(audio_file, osp.join(RES_ROOT, v), v, RES_ROOT)
        





    


        
