
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.residual import ResidualStack
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

class TransConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, upsampling=False):
        super().__init__()
        if not upsampling:
            self.block = nn.Sequential(
                nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
                # nn.InstanceNorm1d(out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.InstanceNorm1d(out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        out = self.block(x)
        return out
        

        

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, out_dim, h_dim=256):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList(
            [nn.Linear(in_dim, h_dim),            
            nn.LeakyReLU(negative_slope=0.02)]            
        )
        for i in range(2):
            self.decoder.append(
                Linear(h_dim, h_dim, residual=True) 
                # nn.Linear(h_dim, h_dim)              
            )
            self.decoder.append(
                # nn.LeakyReLU(negative_slope=0.2)
                Linear(h_dim, h_dim)   
            )
        self.decoder.append(nn.Linear(h_dim, out_dim))

    def forward(self, x):
        # input_shape [B, T, C]
        B, T, C = x.shape
        x = x.contiguous().view(B*T, C)
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        x = x.view(B, T, -1)
        return x

class WindowDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim=64
        self.decoder=nn.ModuleList()
        
        self.decoder.append(TransConvBlock(in_dim, h_dim, upsampling=True))
        self.decoder.append(TransConvBlock(h_dim, h_dim*2, upsampling=False))
        self.decoder.append(nn.Dropout(p=0.4))
        self.decoder.append(TransConvBlock(h_dim*2, h_dim*4, upsampling=True))
        self.decoder.append(nn.Dropout(p=0.4))
        self.decoder.append(TransConvBlock(h_dim*4, h_dim*4, upsampling=False))
        self.decoder.append(nn.Dropout(p=0.4))
        self.decoder.append(TransConvBlock(h_dim*4, h_dim*2, upsampling=True))
        self.decoder.append(nn.Dropout(p=0.4))
        self.decoder.append(TransConvBlock(h_dim*2, h_dim*2, upsampling=False))
        self.decoder.append(nn.Conv1d(h_dim*2, out_dim, kernel_size=1))
        # self.decoder = nn.Sequential(
        #     TransConvBlock(in_dim, h_dim, upsampling=True),
        #     TransConvBlock(h_dim, h_dim*2, upsampling=False),
        #     TransConvBlock(h_dim*2, h_dim*4, upsampling=True),
        #     TransConvBlock(h_dim*4, h_dim*4, upsampling=False),
        #     TransConvBlock(h_dim*4, h_dim*2, upsampling=True),
        #     TransConvBlock(h_dim*2, h_dim*2, upsampling=False),
        #     nn.Conv1d(h_dim*2, out_dim, kernel_size=1)
        # )

    def forward(self, x):
        x = x.contiguous().permute(0, 2, 1)
        for layer in self.decoder:
            x = layer(x)
        # x =self.decoder(x)
        x = x.contiguous().permute(0, 2, 1)
        return x
        

if __name__ == "__main__":
    # random data
    # x = np.random.random_sample((3, 40, 40, 200))
    # x = torch.tensor(x).float()
    x = torch.randn((2, 1, 256), dtype=torch.float32)

    # test decoder
    decoder = WindowDecoder(256,  9)
    print(decoder)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
    print(decoder_out)
