from math import sqrt
import numpy as np
from numpy import finfo
import torch
import os
import sys
sys.path.append('.')
# from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm,LinearNormReLU, ConvNormAct2D, ResBlock
# from model import LocationLayer, Attention 
# from utils import to_gpu, get_mask_from_lengths
from modules import GST
from collections import OrderedDict
from multi_head_attention import SelfAttention, PositionalEncoder
import pdb
from facial_exp import ProsoExp, FrameEncoder



class Audio_Exp_Model(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.mel_channels = hparams.n_mel_channels
        self.feature_channels = 244
        self.encoder = torch.nn.LSTM(input_size = self.feature_channels, batch_first=True, dropout=0.1,
            hidden_size=512, num_layers=4, bidirectional=True)
        self.gst = GST(hparams)
        self.position_embedding = PositionalEncoder(512, max_seq_len=1024)
        self.Linear = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 64)
        )
        self.exp_branch = ProsoExp(2, 512, 1024)
        self.gst_transfrom = nn.Sequential(
            nn.Linear(hparams.token_embedding_size, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 512),
        )
        self.gst.eval()
        self.encoder.eval()


    def forward(self, x, proso_data, length):
        mels = x[:, :, :80]
        gst_featrues = self.cal_gst_feature(mels, length)
        exp_enforce = self.exp_branch(proso_data, gst_featrues)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        with torch.no_grad():
            out, _ = self.encoder(x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.mul(out, exp_enforce)
        B,T,C = out.shape
        out = out.contiguous().view(-1, out.shape[-1])
        out = self.Linear(out)
        out = out.contiguous().view(B, T, -1) 
        return out

    def load_weight(self, checkpoint_path):
        dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_dict = OrderedDict()
        for k,v in dict['LSTM_init'].items():
            if k.split('.')[0] =='module':
                k = '.'.join(k.split('.')[1:])
            new_dict[k] = v
        # dict = OrderedDict(dict['state_dict'])
        pretrain = OrderedDict(torch.load(self.hparams.gst_weight, map_location='cpu')['state_dict'])
        gst_dict = OrderedDict()
        encoder_dict = OrderedDict()
        linear_dict = OrderedDict()
        for k, v in pretrain.items():
            if 'gst' == k.split('.')[0]:
                gst_dict['.'.join(k.split('.')[1:])] = v
        for k, v in new_dict.items():
            if 'encoder' == k.split('.')[0]:
                encoder_dict['.'.join(k.split('.')[1:])] = v
            if 'Linear' == k.split('.')[0]:
                temps = k.split('.') 
                linear_dict['.'.join(temps[1:])] = v
        
        self.encoder.load_state_dict(encoder_dict)
        self.gst.load_state_dict(gst_dict)
        self.Linear.load_state_dict(linear_dict)
        

    def cal_gst_feature(self, mels, length):
        with torch.no_grad():
            gst_embedding = self.gst(mels, torch.tensor(length).cuda())
            gst_embedding = gst_embedding.repeat(1, mels.shape[1], 1) #（B, T）
            gst_embedding = self.gst_transfrom(gst_embedding)
        mask = torch.zeros(mels.shape[0], mels.shape[1], gst_embedding.shape[-1])
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        embedings = self.position_embedding(gst_embedding)
        if mels.is_cuda:
            mask = mask.cuda()
        embedings = embedings * mask
        return gst_embedding