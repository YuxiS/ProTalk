import cv2
import os
from glob import glob
import numpy as np
from moviepy.editor import VideoFileClip
import random
import shutil
from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm
import pdb


def detect_Video_bbox(args):
    video, save_dir=args
    basename =os.path.basename(video)
    if os.path.exists(os.path.join(save_dir, basename)):
        print("File exists!")
        return 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier('/remote-home/yfsong/code/ProTalk/checkpoints/haarcascade_frontalface_default.xml')
    # 打开视频文件，或者替换为摄像头索引（通常为 0）
    video_capture = cv2.VideoCapture(video)
    bbox_list = []
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        # 读取一帧
        ret, frame = video_capture.read()
        if not ret:
            break
        # 将图像转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # 绘制检测框
        # for (x, y, w, h) in faces:
            # 确保检测框不超过画面边界\
        if len(faces)==0:
            continue
        else:
            x, y, w, h = faces[0]
            x = max(0, x)
            y = max(0, y)
            r_x = min(frame.shape[1], x+w)
            r_y = min(frame.shape[0], y+h)
            # 在图像上绘制矩形框
            bbox_list.append([x, y, r_x, r_y])
    try:
        bbox_list = np.array(bbox_list)
        # pdb.set_trace()
        min_x, min_y = bbox_list.min(axis=0)[0], bbox_list.min(axis=0)[1]
        max_r_x, max_r_y = bbox_list.max(axis=0)[2], bbox_list.max(axis=0)[3]
        if int(max_r_x-min_x) == video_width and int(max_r_y-min_y) == video_height:
            # shutil.copy(video, os.path.join(save_dir, basename))
            print("file Error!")
            return
        else:
            if (max_r_x-min_x)>(max_r_y-min_y):
                crop_width = max_r_x-min_x
            else:
                crop_width = max_r_y-min_y
            min_x = max(0, int(min_x-0.15*crop_width))
            min_y = max(0, int(min_y-0.15*crop_width))
            # max_r_x = min(video_width, int(max_r_x+0.2*crop_width))
            # max_r_y = min(video_height, int(max_r_y+0.2*crop_width))
            # 显示结果
            
        v = VideoFileClip(video)
        cropped_video = v.crop(x1=min_x, y1=min_y, width=int(crop_width*1.3), height=int(crop_width*1.3))
        cropped_video = cropped_video.resize(width=256, height=256)
        cropped_video.write_videofile(os.path.join(save_dir, basename), audio_codec='aac', temp_audiofile='temp-audio-{}-{}.mp4'.format(basename, random.randint(0, 10000)), remove_temp=True)
    except:
        pass 
if __name__=='__main__':
    videos = glob(os.path.join('/remote-home/yfsong/code/TalkingHead/Real3DPortrait-main/infer_out/CREMA_D', '*.mp4'))
    save_dir = '/remote-home/yfsong/code/TalkingHead/Real3DPortrait-main/infer_out/CREMA_D'
    pool = Pool(8)
    args = cycle([save_dir])
    for v in tqdm(pool.imap_unordered(detect_Video_bbox, zip(videos, args)), total=len(videos)):
        None
    # detect_Video_bbox(('test3.mp4', './res'))