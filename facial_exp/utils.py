import torch
import  torchvision.transforms.functional as F_v
import numpy as np
import cv2
def generate_mask(images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""
        B, T, C, H, W = images.shape
        masks = []
        for b in range(landmarks.shape[0]):
            mask_sub_seq = []
            for frame_index in range(landmarks.shape[1]):
                # pdb.set_trace()
                center_x, center_y = torch.mean(landmarks[b, frame_index, 48:68, :], dim=0)
                center_x = center_x.round()
                center_y = center_y.round()
                height = 32
                width = 32
                threshold = 5
                img = images[b, frame_index, ...]
                if center_y - height < 0:
                    center_y = height
                if center_y - height < 0 - threshold:
                    raise Exception('too much bias in height')
                if center_x - width < 0:
                    center_x = width
                if center_x - width < 0 - threshold:
                    raise Exception('too much bias in width')
                if center_y + height > img.shape[-2]:
                    center_y = img.shape[-2] - height
                if center_y + height > img.shape[-2] + threshold:
                    raise Exception('too much bias in height')
                if center_x + width > img.shape[-1]:
                    center_x = img.shape[-1] - width
                if center_x + width > img.shape[-1] + threshold:
                    raise Exception('too much bias in width')
                mask = np.zeros((H, W, 1), np.uint8)
                mouse_contours = np.array([
                    [int(center_x-width), int(center_y-height)],
                    [int(center_x+width), int(center_y-height)],
                    [int(center_x+width), int(center_y+height)],
                    [int(center_x-width), int(center_y+height)]
                ]).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [mouse_contours], color=[1])
                mask_sub_seq.append(1-mask)
            masks.append(np.stack(mask_sub_seq, axis=0))
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.contiguous().permute(0, 1, 4, 2, 3)
        return masks

def shuffle_tensor(tensor):
    n = tensor.shape[1]
    perm = torch.randperm(n)
    shuffle_img = tensor[:, perm, ...]
    return shuffle_img


def merge_mask(mask_face, mask_mouse):
    mask = mask_face * mask_mouse
    return mask