import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT
import math
import pdb


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # torch.nn.init.xavier_uniform_(
        #     self.linear_layer.weight,
        #     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LinearNormReLU(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNormReLU, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        # self.norm_layer = nn.BatchNorm1d(out_dim)
        # self.norm_layer = nn.InstanceNorm1d(num_features=out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        # self.act = nn.Tanh()
        

        # torch.nn.init.xavier_uniform_(
        #     self.linear_layer.weight,
        #     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.linear_layer(x)
        # x = self.norm_layer(x)
        x = self.act(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        # torch.nn.init.xavier_uniform_(
        #     self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



class ConvNorm2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias)
        # torch.nn.init.xavier_uniform_(
        #     self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ConvNormAct2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias)
        # self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        # self.act = nn.Tanh()
        # torch.nn.init.xavier_uniform_(
        #     self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
            

    def forward(self, signal):
        conv_signal = self.conv(signal)
        # conv_signal = self.norm(conv_signal)
        out = self.act(conv_signal)
        return out

class CAMConV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias)
        # self.norm = nn.InstanceNorm2d(num_features=out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        # self.act = nn.Tanh()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Softmax(),
        )
        # torch.nn.init.xavier_uniform_(
        #     self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
            

    def forward(self, signal):
        conv_signal = self.conv(signal)
        # conv_signal = self.norm(conv_signal)
        out = self.act(conv_signal)
        attention = self.attention(signal)
        out = out * attention
        return out

class CAMLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        # self.norm_layer = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        # self.act = nn.Tanh()
        self.attentionn = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=bias),
                nn.Softmax()
            )
        )
        # torch.nn.init.xavier_uniform_(
        #     self.linear_layer.weight,
        #     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        out = self.linear_layer(x)
        # out = self.norm_layer(out)
        out = self.act(out)
        atten = self.attentionn(x)
        out = out * atten
        return out

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        self.conv1 = ConvNorm(in_channels=in_channels, out_channels=in_channels*2,kernel_size=kernel_size, stride=stride, 
        padding=padding, dilation=dilation)
        # self.norm1 = nn.BatchNorm1d(in_channels*2)
        self.act_1 = nn.ReLU()
        self.conv2 = ConvNorm(in_channels=in_channels*2, out_channels=out_channels, kernel_size=kernel_size, stride=1, 
        padding=padding, dilation=dilation)
        self.step = nn.Sequential(
            # nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            ConvNorm(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.act_1(self.conv1(x))
        out = self.conv2(out)
        x = self.step(x)
        out = out + x
        return out

class ResBlock1d_v2(nn.Module):
    def __init__(self, in_channels,  kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        self.conv1 = ConvNorm(in_channels=in_channels, out_channels=in_channels*2,kernel_size=1, stride=1, 
        padding=0, dilation=dilation)
        # self.norm1 = nn.BatchNorm1d(in_channels*2)
        self.act_1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            ConvNorm(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=kernel_size, stride=stride, 
        padding=padding, dilation=dilation),
            nn.ReLU()
        )
        self.step = nn.Sequential(
            ConvNorm(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        
        # out = self.act_1(self.norm1( self.conv1(x)))
        out = self.act_1(self.conv1(x))
        out = self.conv2(out)
        out = self.step(out)
        out = out+x
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        self.conv1 = ConvNorm2D(in_channels=in_channels, out_channels=in_channels,kernel_size=kernel_size, stride=stride, 
        padding=padding, dilation=dilation)
        self.act = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = out + x
        out = self.act(out)
        return out


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, ref_level_db = 20, magnitude_power=1.5):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # max_len 是解码器生成句子的最长的长度，假设是 10
        position = torch.arange(0, max_len).unsqueeze(1)
        self.max_len = max_len
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *-(math.log(10000.0) / dim)))


        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        # self.drop_out = nn.Dropout(p=dropout)
        self.dim = torch.tensor([dim], dtype=torch.float32)

    def forward(self, pos):
        # print(torch.sqrt(self.dim))
        # emb = pos * torch.sqrt(self.dim)
        # pdb.set_trace()
        # if step is None:
        #     emb = emb + self.pe[:emb.size(0)]
        # else:
        #     emb = emb + self.pe[step]
        # emb = self.drop_out(emb)
        # pos [0, 1] 一个列表， 列表中的元素都是0-1之间的小数， 表示相对位置
        # pdb.set_trace()
        outs = []
        for p in pos:
            outs.append(self.pe[int(p*self.max_len), :])
        outs = torch.stack(outs, dim=0)

        return outs

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def mu(self, x):
        # x = [B, C, T]
        B, C = x.size()[:2]
        # return torch.sum(x, dim=2).unsqueeze(2)/x.shape[2]
        return torch.mean(x, dim=2).view(B, C, -1)
    
    def sigma(self, x):
        B, C = x.size()[:2]
        # return torch.sqrt((torch.sum((x-self.mu(x))**2, dim=2).unsqueeze(2)+0.000000023)/x.shape[2])
        var = torch.var(x, dim=2)+0.000000023
        std = var.sqrt().view(B, C, -1)
        return std
    
    def forward(self, x, y):
        return self.sigma(y)*((x - self.mu(x))/self.sigma(x))+self.mu(y)



if __name__=='__main__':
    x = torch.randn(2, 16, 32)
    y = torch.randn(2, 16, 32)
    adain = AdaIN()
    mu = adain.mu(x)
    print(mu.shape)
    std = adain.sigma(x)
    print(std.shape)
    out = adain(x, y)
    print(out.shape)

