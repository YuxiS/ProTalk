import torch
import torch.nn as nn
from modules import GST
from layers import AdaIN
from collections import OrderedDict
from multi_head_attention import PositionalEncoder
import pdb

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False):
        super().__init__()
        self.residual = residual
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            nn.InstanceNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                     padding=kernel_size//2, bias=True)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, residual=False):
        super(ConvBlock, self).__init__()
        self.residual = residual
        self.conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.norm = nn.InstanceNorm1d(planes)
        self.relu = nn.SiLU(inplace=True)
        if residual:
            self.shotcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True),
                nn.SiLU()
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        if self.residual:
            out = out + self.shotcut(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, residual=False):
        super().__init__()
        self.conv1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size, stride=stride, residual=True)
        self.conv2 = ConvBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, residual=True)
        self.shotcut = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.adain = AdaIN()
        self.norm = nn.GroupNorm(8, planes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, style_code):
        skip_connection = self.shotcut(x)
        x = self.conv1(x)
        assert x.shape == style_code.shape, "AdaIn x.shape!=style.shape"
        x = self.adain(x, style_code)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x + skip_connection
        x = self.norm(x)
        return x

class BasicBlockRes(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, residual=False):
        super().__init__()
        self.conv1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size, stride=stride, residual=True)
        self.conv2 = ConvBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, residual=True)
        self.shotcut = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.norm = nn.GroupNorm(8, planes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        skip_connection = self.shotcut(x)
        x = self.conv1(x)
        # assert x.shape == style_code.shape, "AdaIn x.shape!=style.shape"
        # x = self.adain(x, style_code)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x + skip_connection
        x = self.norm(x)
        return x

class ProsoStyleEasy(nn.Module):
    def __init__(self, in_dim, hidden_dim=[512, 256, 256, 256]):
        super().__init__()
        self.input_transform = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim[0], kernel_size=5, stride=3, padding=2)
        self.activation =nn.LeakyReLU(negative_slope=0.2)
        self.layer_0 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[0])
        self.layer_1 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[1])
        self.layer_2 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[2])
        self.layer_3 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[3])
        
    def forward(self, x, gst_feature):
        # [B, T, C]
        x = x.contiguous().permute(0, 2, 1) #[B, C, T]
        x = self.activation(self.input_transform(x))
        x = x.contiguous().permute(0, 2, 1)#[B, T, C]
        # pdb.set_trace()
        x = torch.cat([x, gst_feature], dim=2)
        out_0 = self.activation(self.layer_0(x))
        out_1 = self.activation(self.layer_1(x))
        out_2 = self.activation(self.layer_2(x))
        out_3 = self.activation(self.layer_3(x)) 
        out_0 = out_0.contiguous().permute(0, 2, 1)
        out_1 = out_1.contiguous().permute(0, 2, 1)
        out_2 = out_2.contiguous().permute(0, 2, 1)
        out_3 = out_3.contiguous().permute(0, 2, 1)
        return out_0, out_1, out_2, out_3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes*2, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes*2)
        self.conv2 = nn.Conv1d(planes*2, planes, kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(inplanes, planes , kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU( negative_slope=0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer1 = BasicBlockRes(512, 256)
        self.layer2 = BasicBlockRes(256, 256)
        self.layer3 = BasicBlockRes(256, 256)
        self.layer4 = BasicBlockRes(256, 256)
        self.conv_merge = nn.Conv1d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, length):
        # x [B, T, C]
        x = x.contiguous().permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_merge(x)
        x = x.contiguous().permute(0, 2, 1)
        mask = torch.zeros_like(x)
        if x.is_cuda:
            mask = mask.to(x.get_device())
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        x = x * mask
        return x

class ProsoResNet(nn.Module):
    def __init__(self, hparams, mel_channels, pro_channels, out_channels, gst_channels=512):
        super().__init__()
        self.hparams = hparams
        self.style_model=ProsoStyleEasy(pro_channels, hidden_dim=[512, 256, 256, 256])
        self.conv1 = nn.Conv1d(mel_channels, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer1 = BasicBlock(512, 256)
        self.layer2 = BasicBlock(256, 256)
        self.layer3 = BasicBlock(256, 256)
        self.layer4 = BasicBlock(256, 256)
        self.conv_merge = nn.Sequential(
            nn.Linear(256, 256),
            nn.InstanceNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.InstanceNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )
        self.gst = GST(hparams)
        self.gst_transfrom = nn.Linear(hparams.token_embedding_size, 512)
        self.position_embedding = PositionalEncoder(512, max_seq_len=2048)
        self.load_gst_weight()
        self.gst.eval()       
        for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, proso, length):
        gst_embedding = self.cal_gst_feature(x[:, :, :80], length)
        style_0, style_1, style_2, style_3 = self.style_model(proso, gst_embedding) 
        x = x.contiguous().permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x, style_0)
        x = self.layer2(x, style_1) 
        x = self.layer3(x, style_2)
        x = self.layer4(x, style_3)
        x = x.contiguous().permute(0, 2, 1)
        x = self.conv_merge(x)
        mask = torch.zeros_like(x)
        if x.is_cuda:
            mask = mask.to(x.get_device())
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        x = x * mask
        return x

    def load_gst_weight(self):
        pretrain = OrderedDict(torch.load(self.hparams.gst_weight, map_location='cpu')['state_dict'])
        gst_dict = OrderedDict()
        for k, v in pretrain.items():
            if 'gst' == k.split('.')[0]:
                gst_dict['.'.join(k.split('.')[1:])] = v
        self.gst.load_state_dict(gst_dict)
    
    def cal_gst_feature(self, mels, length):
        with torch.no_grad():
            gst_embedding = self.gst(mels, torch.as_tensor(length, device=mels.device))
            gst_embedding = gst_embedding.repeat(1, mels.shape[1], 1) #（B, T）
            gst_embedding = self.gst_transfrom(gst_embedding)
        mask = torch.zeros(mels.shape[0], mels.shape[1], gst_embedding.shape[-1])
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        embedings = self.position_embedding(gst_embedding)
        if mels.is_cuda:
            mask = mask.to(mels.get_device())
        embedings = embedings * mask
        return gst_embedding

class ProsoLinear(nn.Module):
    def __init__(self, hparams, mel_channels, pro_channels, out_channels, gst_channels=512):
        super().__init__()
        self.hparams = hparams
        # self.style_model = ProsoStyle(pro_channels,[512, 256, 256, 256], gst_dim=gst_channels)
        self.style_model = ProsoStyleEasy(pro_channels, [512, 256, 256, 256], gst_dim=gst_channels)
        # self.conv1 = nn.Conv1d(mel_channels, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer1 = Linear(mel_channels, 512)
        self.layer2 = Linear(512, 256)
        self.layer3 = Linear(256, 256)
        self.layer4 = Linear(256, 256)
        # self.layer2 = nn.GRU(512, 256, num_layers=1, dropout=0.3, bidirectional=True, batch_first=True)
        # self.layer3 = nn.GRU(256, 256, num_layers=1, dropout=0.3, bidirectional=True, batch_first=True)
        # self.layer4 = nn.GRU(256, 256, num_layers=1, dropout=0.3, bidirectional=True, batch_first=True)

        # self.conv_merge = nn.Conv1d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_merge = nn.Linear(256, out_channels)
        self.adain = AdaIN()
        self.gst = GST(hparams)
        self.gst_transfrom = nn.Linear(hparams.token_embedding_size, 512)
        self.position_embedding = PositionalEncoder(512, max_seq_len=1024)
        self.load_gst_weight()
        self.gst.eval()
        
        for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, proso, length):
        # x [B, T, C]
        gst_embedding = self.cal_gst_feature(x[:, :, :80], length)
        style_0, style_1, style_2, style_3 = self.style_model(proso, gst_embedding) 

        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        x = self.layer1(x)
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.permute(0, 2, 1).contiguous()
        x = self.adain(x, style_0)
        x = x.permute(0, 2, 1).contiguous()  
        
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        x = self.layer2(x)
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.permute(0, 2, 1).contiguous()
        x = self.adain(x, style_1)
        x = x.permute(0, 2, 1).contiguous()  

        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        x = self.layer3(x)
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.permute(0, 2, 1).contiguous()
        x = self.adain(x, style_2)
        x = x.permute(0, 2, 1).contiguous()  

        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        x = self.layer4(x)
        # x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.permute(0, 2, 1).contiguous()
        x = self.adain(x, style_3)
        x = x.permute(0, 2, 1).contiguous()
  

        x = self.conv_merge(x)
        mask = torch.zeros_like(x)
        if x.is_cuda:
            mask = mask.to(x.get_device())
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        x = x * mask
        return x

    def load_gst_weight(self):
        pretrain = OrderedDict(torch.load(self.hparams.gst_weight, map_location='cpu')['state_dict'])
        gst_dict = OrderedDict()
        for k, v in pretrain.items():
            if 'gst' == k.split('.')[0]:
                gst_dict['.'.join(k.split('.')[1:])] = v
        self.gst.load_state_dict(gst_dict)
    
    def cal_gst_feature(self, mels, length):
        with torch.no_grad():
            gst_embedding = self.gst(mels, torch.as_tensor(length, device=mels.device))
            gst_embedding = gst_embedding.repeat(1, mels.shape[1], 1) #（B, T）
            gst_embedding = self.gst_transfrom(gst_embedding)
        mask = torch.zeros(mels.shape[0], mels.shape[1], gst_embedding.shape[-1])
        for i, l in enumerate(length):
            mask[i, :l, :]=1.
        embedings = self.position_embedding(gst_embedding)
        if mels.is_cuda:
            mask = mask.to(mels.get_device())
        embedings = embedings * mask
        return gst_embedding

if __name__=='__main__':
    from hparams import create_hparams
    hparams = create_hparams()
    mel = torch.randn((2, 16, 120))
    proso = torch.randn((2, 48, 2))
    length  = [13, 16]
    gst = torch.randn((2, 16, 512))
    model = ProsoResNet(hparams, 120, 2, 64, 512)
    out = model(mel, proso, length)
    print(out.shape)