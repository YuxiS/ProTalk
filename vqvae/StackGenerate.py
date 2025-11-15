from .models import VAEDecoder, VectorQuantizer, WindowDecoder, VectorQuantizerEMA
from .pose_sampler import PoseSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb




class StackSampler(nn.Module):
    def __init__(self, in_dim, n_embeddings, embedding_dim, hidden, out_dim, beta=0.25, vae_weight=None, sampling_weight=None, overlap=4):
        super().__init__()
        self.in_dim = in_dim
        self.n_embeddings = n_embeddings
        self.out_dim = out_dim
        # self.codebook = VectorQuantizer(self.n_embeddings, embedding_dim, beta)
        self.codebook = VectorQuantizerEMA(self.n_embeddings, embedding_dim, beta)
        # self.decoder = VAEDecoder(embedding_dim, out_dim)
        self.decoder = WindowDecoder(embedding_dim, out_dim)
        self.sampler = PoseSampler(self.in_dim, n_embeddings, hidden)
        self.load_weight(vae_weight, sampling_weight)
        self.stacknum = int(overlap/(8-overlap)) # overlap>=4
        self.seg_len = int(8-overlap)
        self.stack_size = int(overlap/(8-overlap))
        self.stack_list = [ [ ] ]*self.stack_size 
        
        self.codebook.eval()
        self.decoder.eval()

    def load_weight(self, vae_weight, sampling_weight):
        if vae_weight is not None and sampling_weight is not None:
            vae_weight = torch.load(vae_weight)
            sampler_weight = torch.load(sampling_weight)
            self.codebook.load_state_dict(vae_weight['CodeBook'])
            self.decoder.load_state_dict(vae_weight['Decoder'])
            self.sampler.load_state_dict(sampler_weight)

    def load_CodeBook(self, vae_weight):
        if vae_weight is not None:
            vae_weight = torch.load(vae_weight)
            self.codebook.load_state_dict(vae_weight['CodeBook'])
            self.decoder.load_state_dict(vae_weight['Decoder'])

    

    # def forward(self, proso, length):
    #     B = proso.shape[0]
    #     coding_index = self.sampler(proso, length) # [B, T, C]
    #     coding_index = coding_index.contiguous().view(-1, coding_index.shape[-1]) #(B*T, C)
    #     coding_index = torch.argmax(coding_index, dim=1)
    #     sampler_z_q = self.generate_samplers(coding_index)
    #     sampler_z_q = sampler_z_q.contiguous().view(B, -1, sampler_z_q.shape[-1])
    #     pose_seq =  self.decoder(sampler_z_q)
    #     return pose_seq

    def forward(self, proso, length):
        step = self.seg_len
        B = proso.shape[0]
        coding_index = self.sampler(proso, length) # [B, T, C]
        ##################################################################
        coding_index = coding_index[:, ::step, :]
        T = coding_index.shape[1]
        res = []
        # pdb.set_trace()
        for t in range(T):
            coding_index_temp = coding_index[:, t, :]
        ##################################################################
            coding_index_temp = coding_index_temp.contiguous().view(-1, coding_index_temp.shape[-1])
            # coding_index = coding_index.contiguous().view(-1, coding_index.shape[-1]) #(B*T, C)
            index = torch.argmax(coding_index_temp, dim=1)
            sampler_z_q = self.generate_samplers(index)
            sampler_z_q = sampler_z_q.contiguous().view(B, -1, sampler_z_q.shape[-1])
            pose_seq =  self.decoder(sampler_z_q)
            base = self.average(pose_seq)
            res.append(base)
            # half = int(pose_seq.shape[1]/2)
            # if len(res)==0:
            #     res.append(pose_seq[:, :half, :])
            #     res.append(pose_seq[:, half:, :])
            # else:
            #     last = res.pop()
            #     temp = (last + pose_seq[:, :half, :])/2
            #     res.append(temp)
            #     res.append(pose_seq[:, half:, :])
        #################################################################### 
        # pdb.set_trace()
        pose_res = torch.cat(res, dim=1)
        ####################################################################
        return pose_res, coding_index


    def generate_samplers(self, e_index):
        """
            e_index:[B, 1]

        """
        min_encodings = torch.zeros((e_index.shape[0], self.n_embeddings), dtype=torch.float).to(e_index.get_device())
        min_encodings.scatter_(1, e_index.unsqueeze(1), 1)
        e_weights = self.codebook.embedding.weight
        z_q = torch.matmul(min_encodings, e_weights)
        return z_q

    def average(self, pose):
        base = pose[:, :self.seg_len, :]
        temp = [base]
        for i, stack in enumerate(self.stack_list):
            if len(stack)==0:
                for k in range(self.stack_size):
                    stack.append(pose[:, (k+1)*self.seg_len:(k+2)*self.seg_len, :])
                break
            else:
                temp.append(stack.pop(0))
                if len(stack)==0:
                    for k in range(self.stack_size):
                        stack.append(pose[:, (k+1)*self.seg_len:(k+2)*self.seg_len, :])
        base_new = 0.
        for t in temp:
            base_new = base_new + t
        base_new = base_new/len(temp)
        return base_new
                

    # def move_average(self, res_list):







