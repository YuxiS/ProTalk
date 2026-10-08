"""Historical Gaussian sliding-window kernel used during generation."""
import numpy as np
import torch
from torch import nn

def gaussian_kernel(in_channel, out_channel, kernel_size=5, std=1, mean=0):
     # 标准高斯分布
    radius = kernel_size//2
    f = lambda x: np.exp(-((x-mean)**2)/ (2*std**2))
    weight = np.array([f(i) for i in range(-radius, radius+1)])
    weight = weight/sum(weight)
    kernel = torch.FloatTensor(weight).view(1, 1, kernel_size)
    kernel = kernel.repeat(out_channel, 1, 1)
    kernel = nn.parameter.Parameter(kernel, requires_grad=False)
    return kernel
