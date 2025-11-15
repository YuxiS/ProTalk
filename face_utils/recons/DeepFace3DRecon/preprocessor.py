import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from .utils.bfm import load_lm3d
from .utils.align_batch import align_img_batch
from .utils.align import align_img

import pdb


class Preprocessor(nn.Module):
    def __init__(self, bfm_folder, **kwargs):
        super(Preprocessor, self).__init__()
        self.lm3d_std = load_lm3d(bfm_folder)
        self.device = 'cpu'
        self.kwargs = kwargs
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **kwargs)
    
    def to(self, device):
        self.device = device
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self
    
    def cuda(self):
        self.device = 'cuda'
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self
    
    def cpu(self):
        self.device = 'cpu'
        self.fa = FaceAlignment(LandmarksType._2D, device=self.device, **self.kwargs)
        return self

    def extract_bboxs(self, x):
        """
        Extract bounding boxes from input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            list[np.array]: Bounding boxes in (4, )
        """
        x = x.permute(0, 3, 1, 2).float().to(self.device)
        return self.fa.face_detector.detect_from_batch(x)
    
    def extract_keypoints(self, x):
        """
        Extract keypoints from input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            list[np.array]: Keypoints in (68, 2)
        """
        x = x.permute(0, 3, 1, 2).float().to(self.device)
        # x = x.to(self.device)
        return self.fa.get_landmarks_from_batch(x)

    def align_and_recrop(self, x):
        """
        Perform face alignment and recrop the input frames.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            torch.tensor: Transform parameters in (N, 5)
            torch.tensor: Aligned and recropped frames in (N, H, W, C)
            torch.tensor: Aligned and recropped landmarks in (N, 68, 2)
            torch.tensor: Aligned and recropped masks in (N, H, W)
        """
        keypoints = self.extract_keypoints(x)
        keypoints = np.array(keypoints)
        # keypoints = np.stack([e if e is not None else -1 * np.ones((68, 2)) for e in keypoints])
        # keypoints = torch.load('keypoints_v2.pt')
        if keypoints.shape[1]==0:
            return None, None, None, None
        trans_params, new_images, new_landmarks, new_masks = align_img_batch(
            x,
            torch.from_numpy(keypoints).to(self.device),
            torch.from_numpy(self.lm3d_std).to(self.device),
        )
        return trans_params, new_images, new_landmarks, new_masks
    
    def prepare_input(self, x):
        """
        Convert input frames to the format required by the network. The device is not changed.

        Args:
            x (torch.tensor): Input frames in (N, H, W, C)
        Returns:
            torch.tensor: Normalized input frames in (N, C, H, W)
        """
        x = x.permute(0, 3, 1, 2).float()
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x /= 255.
        return x
########################################
    def image_transform(self, image_tensor, image_pil):
        """
        param:
            images:          -- PIL image 
            lm:              -- numpy array
        """
        lm = self.extract_keypoints(image_tensor)
        lm = np.array(lm[0])
        if lm is None:
            return None, None        
        images = image_pil
        W,H = images.size
    
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, trans_params
