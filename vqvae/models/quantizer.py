import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B, T, C)
            2. flatten input to (B*T,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        # pdb.set_trace()
        # B, T, C = z.shape
        # z = z.permute(0, 2, 1).contiguous()
        # pdb.set_trace()
        z_flattened = z.contiguous().view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # pdb.set_trace()
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device, dtype=z.dtype)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        # loss = self.beta * torch.mean((z_q.detach()-z)**2) +  torch.mean((z_q - z.detach()) ** 2)
        # loss = self.beta * F.mse_loss(z, z_q.detach()) + F.mse_loss(z.detach(), z_q)
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        # pdb.set_trace()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()
        # z_q = z_q.view

        return z_q, loss,  (perplexity, min_encodings, min_encoding_indices)



class GumbelQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, straight_through=False):
        super(GumbelQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.straight_through = straight_through

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.linear = nn.Linear()
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.temperature = 1.0
        self.kld_scale = 5e-4



    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B, T, C)
            2. flatten input to (B*T,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        # pdb.set_trace()
        # B, T, C = z.shape
        # z = z.permute(0, 2, 1).contiguous()
        # pdb.set_trace()
        # z_flattened = z.contiguous().view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hard = self.straight_through if self.training else True
        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum('b t n, n d -> b t d', soft_one_hot, self.embedding.weight)
        qy = F.softmax(z, dim=2)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_e + 1e-10), dim=2).mean()
        ind = soft_one_hot.argmax(dim=2)

        min_encodings = torch.zeros(
            z.shape[0], self.n_e).cuda()
        min_encodings.scatter_(1, ind, 1)
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return  z_q, diff, (perplexity, min_encodings, ind)
        # compute loss for embedding
        # loss = self.beta * torch.mean((z_q.detach()-z)**2) +  torch.mean((z_q - z.detach()) ** 2)
        # loss = self.beta * F.mse_loss(z, z_q.detach()) + F.mse_loss(z.detach(), z_q)
        # loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
        #            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        # z_q = z + (z_q - z).detach()
        # pdb.set_trace()

        # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()
        # z_q = z_q.view

        # return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_e, e_dim, beta, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(self.n_e))
        self.ema_w = nn.Parameter(self.embedding.weight.detach().clone())
        # self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, z, istrain = False):
        # z = z.permute(0, 2, 1).contiguous()#(b,8,64)
        z_flattened = z.view(-1, self.e_dim) #(b*8,64)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t()) # (n*8,128)

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) #(n*8,1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)#(b*8,128)
        min_encodings.scatter_(1, min_encoding_indices, 1) #(b*8,128)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  #(n*8,128)*(128,64)->(n*8,64)->(n,8,64)

        if istrain:
            counts = torch.sum(min_encodings, 0)
            dw = torch.matmul(min_encodings.t(), z_flattened)
            if getattr(self, 'sync_ema', False):
                if not torch.distributed.is_initialized():
                    raise RuntimeError('sync_ema requires an initialized process group')
                torch.distributed.all_reduce(counts)
                torch.distributed.all_reduce(dw)
            # EMA updates are state updates, not newly allocated optimizer parameters.
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(
                    counts, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = ((self.ema_cluster_size + self.epsilon) /
                            (n + self.n_e * self.epsilon) * n)
                self.ema_cluster_size.copy_(smoothed)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                self.embedding.weight.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()#(n,64,8)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        #(b,64,8)
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0, 2, 1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)  # (n*8,128)
        min_encodings.scatter_(1, indices[:, None], 1) #onehot

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)  # (n*8,64)

        if shape is not None:
            z_q = z_q.view(shape)  # (b,256,4)

            # reshape back to match original input shape
            # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    
    
    # soften code by distances
    @torch.no_grad()
    def get_soft_code(self, z, temp=0.5, stochastic=False, nucleus_sampling=False, top_p=0.5, filter_value=-float('Inf')):
        
        #(b,64,8)
        distances = self.get_distance(z).reshape(z.shape[0],z.shape[2],-1) #(b*8,128)->(b,8,128)
        soft_code = F.softmax(-distances / temp, dim=-1) #(b,8,128) logits
        
        if nucleus_sampling:
            sorted_logits, sorted_indices = torch.sort(
                soft_code,
                descending=True,
                dim=-1
            )
            cumulative_probs = torch.cumsum(
                sorted_logits,
                dim=-1
            )
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            soft_code = soft_code.masked_fill(indices_to_remove, filter_value)
            soft_code = F.softmax(soft_code, dim=-1) #(b,8,128)

        if stochastic:
            # non-deterministic
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1]) #(b*8,128)
            code = torch.multinomial(soft_code_flat, 1) # [b*8, 1]
            code = code.reshape(*soft_code.shape[:-1]) #[b*8]
        else:
            # deterministic
            code = distances.argmin(dim=-1)

        quant = self.get_codebook_entry(code.reshape(-1),shape=None).view(code.shape[0],code.shape[1],-1)#(b,8,64)
        return soft_code, code, quant
    

    # TODO test soften code by similarity(attention based)
    @torch.no_grad()
    def get_soft_code_v2(self, z, temp=0.5, stochastic=False, nucleus_sampling=False, top_p=0.5, filter_value=-float('Inf')):
        
        #(b,64,8)
        distances = self.get_distance(z) #(n*8,128)
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #(n*8,1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)#(n*8,128)
        min_encodings.scatter_(1, min_encoding_indices, 1) #(n*8,128)
        #(n*8,128)*(128,64)->(n*8,64)
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        #(n*8,64)*(64,128)->(n*8,128)
        similarity = torch.matmul(z_q, self.embedding.weight.t()) * z.shape[1]**-0.5
        #(n*8,128)->(n,8,128)
        soft_code = F.softmax(similarity.reshape(z.shape[0],z.shape[2],-1), dim=-1)
         
        if nucleus_sampling:
            sorted_logits, sorted_indices = torch.sort(
                soft_code,
                descending=True,
                dim=-1
            )
            cumulative_probs = torch.cumsum(
                sorted_logits,
                dim=-1
            )
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            soft_code = soft_code.masked_fill(indices_to_remove, filter_value)
            soft_code = F.softmax(soft_code, dim=-1) #(b,8,128)

        if stochastic:
            # non-deterministic
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1]) #(b*8,128)
            code = torch.multinomial(soft_code_flat, 1) # [b*8, 1]
            code = code.reshape(*soft_code.shape[:-1]) #[b*8]
        else:
            # deterministic
            code = distances.argmin(dim=-1)

        quant = self.get_codebook_entry(code.reshape(-1),shape=None).view(code.shape[0],code.shape[1],-1)#(b,8,64)
        return soft_code, code, quant