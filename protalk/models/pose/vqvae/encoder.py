import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.residual import ResidualStack
import pdb

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False):
        super().__init__()
        self.residual = residual
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out

class Debug(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}:', x.shape, x)
        print('E(x):', [e.mean() for e in x])
        print('V(x):', [e.var() for e in x])
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, residual=False):
        super().__init__()
        self.stride = 1
        self.residual = residual
        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_dim),
            # nn.InstanceNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.02)
        )

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out
    
class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim,  out_dim, h_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            [nn.Linear(in_dim, h_dim),
            nn.LeakyReLU(negative_slope=0.2)]            
        )
        for i in range(2):
            self.encoder.append(
                Linear(h_dim, h_dim, residual=True)
                # Linear(h_dim, h_dim)
                # nn.Linear(h_dim, h_dim)              
            )
            self.encoder.append(
                Linear(h_dim, h_dim)
                #  nn.LeakyReLU(negative_slope=0.2)
                
            )
        self.encoder.append(nn.Linear(h_dim, out_dim))

    def forward(self, x):
        # input_shape [B, T, C]
        B, T, C = x.shape
        x = x.contiguous().view(B*T, C)
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        x = x.view(B, T, -1)
        return x

class WindowEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim = 64
        self.encoder = nn.ModuleList(
            [
            ConvBlock(in_dim, h_dim  , residual=False),
            ConvBlock(h_dim, h_dim*2, kernel_size=3, stride=2, padding=1, residual=False),
            nn.Dropout(p=0.3),
            ConvBlock(h_dim*2, h_dim*2, residual=True),
            ConvBlock(h_dim*2, h_dim*4, kernel_size=3, stride=1, padding=0, residual=False),
            nn.Dropout(p=0.3),
            ConvBlock(h_dim*4, h_dim*4, residual=True),
            ConvBlock(h_dim*4, h_dim*2, kernel_size=2, stride=2, padding=0, residual=False),
            nn.Dropout(p=0.3),
            ConvBlock(h_dim*2, h_dim*2, residual=True),
            nn.Conv1d(h_dim*2, out_dim, kernel_size=1, stride=1)
        ])
    
    def forward(self, x):
        x = x.contiguous().permute(0, 2, 1)
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        x = x.contiguous().permute(0, 2, 1)
        return x

if __name__ == "__main__":
    # random data
    # x = np.random.random_sample((3, 40, 40, 200))
    # x = torch.tensor(x).float()
    x = torch.randn((4, 8, 9), dtype=torch.float32)
    # test encoder
    encoder = WindowEncoder(9, 256)
    print(encoder)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
    print(encoder_out)
