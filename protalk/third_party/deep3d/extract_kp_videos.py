import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
import pdb
from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, Process, set_start_method

class KeypointExtractor():
    def __init__(self, device='cuda'):
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

    def extract_keypoint(self, images, name=None):
        if isinstance(images, list):
            keypoints = []
            for image in images:
                current_kp = self.extract_keypoint(image)
                if current_kp is None:
                    print("No face detected in {}".format(name))
                    return None 
                
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])
            if len(keypoints)==0:
                 print(name, 'No face in video')
                 return None 
            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[0]
                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print(e)
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break    
                except TypeError:
                    # print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)              
                    # break
                    return None
                    # break
            # if name is not None:
            #     np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    # frames_list = glob.glob(os.path.join(filename, '*.jpg'))
    # for f in frames_list:
    #     frame = cv2.imread(f)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame = Image.fromarray(frame)
    #     frames.append(frame)
    return frames

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    # print(name)
    # os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir,  name[-1])
    )

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, default='/home/songyifei9/data/MEAD_VIDEO/videos_256', help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, default='/home/songyifei9/data/MEAD_VIDEO/keypoints', help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0, 1, 2, 3')
    parser.add_argument('--workers', type=int, default=6)

    opt = parser.parse_args()
    # filenames = os.listdir('/home/songyifei9/data/MEAD/re_aligned/M032_aligned')
    # filenames = [os.path.join('/home/songyifei9/data/MEAD/re_aligned/M032_aligned', v) for v in filenames]
    filenames = []
    os.makedirs(opt.output_dir, exist_ok=True)
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    # ids = ['Actor_08','Actor_09','Actor_10']
    # for id in ids:
    ids = ['M022', 'W016', 'W018', 'W037']
    for id in ids:
        opt.input_dir = '/home/songyifei9/data/MEAD_VIDEO/videos_256'
        opt.output_dir = '/home/songyifei9/data/MEAD_VIDEO/keypoints'
        opt.input_dir = os.path.join(opt.input_dir, id)
        os.makedirs(os.path.join(opt.output_dir, id), exist_ok=True)
        opt.output_dir = os.path.join(opt.output_dir, id)
        for ext in extensions:
            filenames += sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
        # filenames = filenames[int(0.3*len(filenames)):]
            print('Total number of videos:', len(filenames))
        pool = Pool(opt.workers)
        args_list = cycle([opt])
        device_ids = opt.device_ids.split(",")
        device_ids = cycle(device_ids)
        for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids)), total=len(filenames)):
            None
        pool.close()