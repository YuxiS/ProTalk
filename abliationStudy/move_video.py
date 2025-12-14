import os
import shutil
from glob import glob
from tqdm import tqdm

SOURCE_DIR = '/home/songyifei9/data/MEAD_VIDEO/videos_256'
TARGET_DIR = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/all/real'
GT_DIR = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/all/GT'
videofiles = glob(os.path.join(GT_DIR, '*.mp4'))

for v in tqdm(videofiles):
    video = os.path.basename(v)
    id, basename = video[:4], video[5:]
    f = os.path.join(SOURCE_DIR, id, basename)
    assert os.path.exists(f), "{} not exist!".format(f)
    shutil.copy(f, os.path.join(TARGET_DIR, video))
