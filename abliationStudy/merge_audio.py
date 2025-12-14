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


if __name__=='__main__':
    AUDIO_ROOT = '/remote-home/share/yfsong/MEAD_VIDEO/wav'
    # AUDIO_ROOT = '/remote-home/share/yfsong/RAVEDSS/wav'
    # AUDIO_ROOT = '/remote-home/share/yfsong/CREMA-D/wav'
    VIDEO_ROOT = '/remote-home/yfsong/code/ProTalk/test_res/ablia_videos/head/cat_videos'
    # VIDEO_ROOT = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/compareStudy/res/RAVEDSS'
    # VIDEO_ROOT = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/CAT_DIR/CREMA-D'
    RES_ROOT= VIDEO_ROOT+'/with_audio'
    os.makedirs(RES_ROOT, exist_ok=True)
    videofiles = os.listdir(VIDEO_ROOT)
    for v in tqdm(videofiles):
        if v[-4:] == '.mp4':
            basename =  os.path.splitext(os.path.basename(v))[0]
            id = basename[:4]
            audio_file = os.path.join(AUDIO_ROOT, id, basename[5:]+'.wav')
            # audio_name = basename.split('-')[-1] # CREMA-D
            # audio_name = 'Actor_'+str(basename.split('-')[2])+'-'+basename
            # audio_file = os.path.join(AUDIO_ROOT,  audio_name+'.wav')
            if os.path.exists(audio_file):
                merge(audio_file, osp.join(VIDEO_ROOT, v), v, RES_ROOT)
        





    


        
