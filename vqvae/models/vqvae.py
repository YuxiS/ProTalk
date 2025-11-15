
import torch
import torch.nn as nn
import numpy as np
import pdb

from .encoder import Encoder, WindowEncoder
from .quantizer import VectorQuantizer, GumbelQuantizer, VectorQuantizerEMA
from .decoder import Decoder, WindowDecoder


class VQVAE(nn.Module):
    def __init__(self, in_dim, n_embeddings, 
        embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        # self.encoder = Encoder(in_dim, embedding_dim)
        self.encoder = WindowEncoder(in_dim, embedding_dim)
        # self.pre_quantization_conv = nn.Conv2d(
        #     h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        # self.vector_quantization = VectorQuantizer(
        #     n_embeddings, embedding_dim, beta)
        self.vector_quantization = VectorQuantizerEMA(
            n_embeddings, embedding_dim, beta
        )
        # self.vector_quantization = GumbelQuantizer(
        #     n_embeddings, embedding_dim, beta
        # )
        # decode the discrete latent representation
        # self.decoder = Decoder(embedding_dim,  in_dim)
        self.decoder = WindowDecoder(embedding_dim, in_dim)

    def forward(self, x, verbose=False, istrain=False):
        # print(x.shape)
        # import pdb
        # pdb.set_trace()
        # x = torch.randn(128, 9, 8)

        z_e = self.encoder(x)
        # pdb.set_trace()
        # z_e = self.pre_quantization_conv(z_e)
        # embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        # embedding_loss, z_q, ind = self.vector_quantization(z_e)
        z_q, embedding_loss, (perplexity, _, _) = self.vector_quantization(z_e, istrain=istrain)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
        # return embedding_loss, x_hat, ind



if __name__=='__main__':
    model = VQVAE(9, 512, embedding_dim=256, beta=0.25).cuda()
    print(model)
    x = torch.randn((4, 8, 9), dtype=torch.float32).cuda()
    out = model(x, verbose=True)