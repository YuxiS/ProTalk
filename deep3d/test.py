import torch
import scipy.io as scio
import os
import numpy as np
import sys

sys.path.append('.')
sys.path.append('..')
from models.face_render import Face_render
from options.face_render_options import Face_Render_Options

opt = Face_Render_Options().init_options()

# data = []
# for mat in data_list:
def merge_vector(data):
    """

    Args:
        data:{dict}

    Returns:

    """
    id = np.array(data['id'])
    exp = np.array(data['exp'])
    tex = np.array(data['tex'])
    angle = np.array(data['angle'])
    gamma = np.array(data['gamma'])
    trans = np.array(data['trans'])
    landmarks_gt = np.array(data['lm68'])
    vec = np.concatenate((id, exp, tex, angle, gamma, trans), axis=1)
    return vec, landmarks_gt

data_dir = './checkpoints'
data_list = ['000002.mat', '000006.mat', '000007.mat', '000031.mat']

data = []
landmarks=[]
for mat in data_list:
    vec, lm = merge_vector(scio.loadmat(os.path.join(data_dir, mat)))
    data.append(vec)
    landmarks.append(lm)

# data = scio.loadmat(os.path.join(data_dir, data_list[0]))
# print(merge_vector(data), merge_vector(data).shape)
# print(data)
data = np.concatenate(data, axis=0)
landmarks_gt = np.concatenate(landmarks, axis=0)
tensor = torch.from_numpy(data).cuda()
landmarks_gt = torch.from_numpy(landmarks_gt).cuda()
print(data.shape)
model = Face_render(opt)
print(model)
model.forward(tensor)
print(model.pred_lm.shape)
print(model.compute_losses(landmarks_gt))

