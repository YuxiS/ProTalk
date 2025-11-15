import torch
import os
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time
import numpy as np
torch.cuda.manual_seed(int(time.time()))


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, residual=False):
        super(ConvBlock, self).__init__()
        self.residual = residual
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.norm = nn.InstanceNorm1d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        if self.residual:
            out = out + x
        return out


class PoseSampler(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim) -> None:
        """
        out_dim: codebook的容量，因为最后的输出是一个one_hot的向量
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.preLinear = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim))
        self.sampler = nn.LSTM(hidden_dim, hidden_dim, num_layers=4, dropout=0.3, bidirectional=True, batch_first=True)
        # self.sampler = nn.Sequential(
        #     ConvBlock(hidden_dim, 2*hidden_dim, kernel_size=7, residual=False),
        #     ConvBlock(2*hidden_dim, 2*hidden_dim, kernel_size=5, residual=True),
        #     nn.Dropout(0.3),
        #     ConvBlock(2*hidden_dim, 4*hidden_dim, residual=False),
        #     ConvBlock(4*hidden_dim, 4*hidden_dim, residual=True),
        #     nn.Dropout(0.3),
        #     ConvBlock(4*hidden_dim, 2*hidden_dim, residual=False),
        #     ConvBlock(2*hidden_dim, 2*hidden_dim, residual=True),
        #     nn.Dropout(0.3),
        #     ConvBlock(2*hidden_dim, hidden_dim, residual=False)
        # )
        self.fuse_layer = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=3, padding=2),
            nn.InstanceNorm1d(num_features=hidden_dim),
            nn.LeakyReLU(0.2)) 
            # nn.GELU() # For Prosody 
        
        # self.fuse_layer = nn.Sequential(
        #     nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.2)) # For MFCC


        # self.window_layer = nn.Sequential(
        #     nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
        #     nn.InstanceNorm1d(num_features=hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
        #     nn.InstanceNorm1d(num_features=hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     ) # For Prosody 
        self.postLinear = nn.Sequential(
                nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim),
                nn.GELU(),
                nn.Linear(in_features=hidden_dim, out_features=out_dim)
            )
        # nn.init.kaiming_normal_(self.preLinear.weight)
        # nn.init.kaiming_normal_(self.postLinear.weight)

    def forward(self, x, length):
        """
            x: [B, T, C]
        """
        ######################################
        B, T, C = x.shape
        x = x.contiguous().view(B*T, C)
        x = self.preLinear(x)
        x = x.contiguous().view(B, T, -1).permute(0, 2, 1) #[B, C, T]
        x = self.fuse_layer(x)
        # x = self.sampler(x)
        x = x.contiguous().permute(0, 2, 1)
        total_length = max(length)
        h_0 = torch.rand((2*4, B, self.hidden_dim), dtype=torch.float).to(x.get_device())
        c_0 = torch.rand((2*4, B, self.hidden_dim), dtype=torch.float).to(x.get_device())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        x, _ = self.sampler(x, (h_0, c_0))
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length) 
        B, T, C = x.shape
        x = x.contiguous().view(B*T, -1)
        x = self.postLinear(x)
        x = x.view(B, T, -1)
        ############################################
        # mask = torch.zeros_like(x)
        # if x.is_cuda:
        #     mask = mask.to(x.get_device())
        # # pdb.set_trace()
        # for i, l in enumerate(length):
        #     l = int(l//3)
        #     mask[i, :l, :]=1.
        # x = x * mask
        ############################################
        return x
    

        

    # def sample(self, x, length):
    #     B, T, C = x.shape
    #     x = x.contiguous().view(B*T, C)
    #     x = self.preLinear(x)
    #     x = x.contiguous().view(B, T, -1).permute(0, 2, 1) #[B, C, T]
    #     x = self.fuse_layer(x)
    #     x = x.contiguous().permute(0, 2, 1) #[B, T, C]
    #     # length = torch.div(length, 3, rounding_mode='floor').to(torch.int32)
    #     x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
    #     x, _ = self.sampler(x)
    #     x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True) 
    #     B, T, C = x.shape
    #     x = x.contiguous().view(B*T, -1)
    #     x = self.postLinear(x)
    #     x = x.view(B, T, -1)
    #     x = F.softmax(x, dim=2)
    #     return x


if __name__=='__main__':
    torch.cuda.set_device(1)
    model = PoseSampler(4, 256, 128).cuda()
    x = torch.randn((2, 255, 4)).cuda()
    length = [255//3, 135//3]
    out = model(x, length)
    print(out.shape)