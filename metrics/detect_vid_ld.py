import cv2
from face_alignment import FaceAlignment, LandmarksType
import numpy as np

detector = FaceAlignment(LandmarksType._2D, device='cuda')

def detect_landmarks(video_name):
    landmarks = []
    