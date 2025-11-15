import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import pdb
import sys
sys.path.append('.')
sys.path.append('..')
from .utils import shuffle_tensor
from multi_head_attention import MultiHeadAttention, PositionalEncoder
from Visual.expression_loss import ExpressionLossNet



class ProsoExp(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim, head_num=4):
        super().__init__()
        self.input_transform = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=5, stride=3, padding=2)
        self.embed_key_0 = nn.Linear(hidden_dim, hidden_dim)
        self.embed_value_0 = nn.Linear(hidden_dim, hidden_dim)
        self.activation =nn.LeakyReLU(negative_slope=0.2)
        self.norm_0 = nn.LayerNorm(hidden_dim)
        self.attention_layer_0 = MultiHeadAttention(in_features=hidden_dim, head_num=head_num)
        self.embed_key_1 = nn.Linear(hidden_dim, hidden_dim)
        self.embed_value_1 = nn.Linear(hidden_dim, hidden_dim)
        self.attention_layer_1 = MultiHeadAttention(in_features=hidden_dim, head_num=head_num)
        self.out_linear_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm_1 = nn.LayerNorm(hidden_dim)

        
    def forward(self, x, gst_feature):
        query = gst_feature
        x = x.contiguous().permute(0, 2, 1)
        x = self.activation(self.input_transform(x))
        x = x.contiguous().permute(0, 2, 1)
        key_0 = self.activation(self.embed_key_0(x))
        value_0 = self.activation(self.embed_value_0(x))
        out_0 = self.attention_layer_0(query, key_0, value_0)
        out_0 = self.norm_0(out_0 + x)
        key_1 = self.activation(self.embed_key_1(out_0))
        value_1 = self.activation(self.embed_value_1(out_0))
        out_1 = self.attention_layer_1(query, key_1, value_1)
        out = self.norm_1(out_1+out_0)
        B, T, C = out_1.shape
        out = out.contiguous().view(-1, C)
        out = self.out_linear_1(out)
        out = out.contiguous().view(B, T, out.shape[-1])
        return out

class ProsoStyle(nn.Module):
    def __init__(self,in_dim, hidden_dim=[512, 256, 256, 256], gst_dim=512, head_num=4):
        super().__init__()
        self.input_transform = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim[0], kernel_size=5, stride=3, padding=2)
        self.activation =nn.LeakyReLU(negative_slope=0.2)
            
        self.embed_key_0 = nn.Linear(hidden_dim[0], hidden_dim[0])
        self.embed_value_0 = nn.Linear(hidden_dim[0], hidden_dim[0])
        self.embed_query_0 = nn.Linear(gst_dim, hidden_dim[0])
        self.attention_layer_0 = MultiHeadAttention(in_features=hidden_dim[0], head_num=head_num)
    
        self.embed_key_1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.embed_value_1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.embed_query_1 = nn.Linear(gst_dim, hidden_dim[1])
        self.attention_layer_1 = MultiHeadAttention(in_features=hidden_dim[1], head_num=head_num)
        

        self.embed_key_2 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.embed_value_2 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.embed_query_2 = nn.Linear(gst_dim, hidden_dim[2])
        self.attention_layer_2 = MultiHeadAttention(in_features=hidden_dim[2], head_num=head_num)
        

        self.embed_key_3 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.embed_value_3 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.embed_query_3 = nn.Linear(gst_dim, hidden_dim[3])
        self.attention_layer_3 = MultiHeadAttention(in_features=hidden_dim[3], head_num=head_num)
        
        
    def forward(self, x, gst_feature):
        # [B, T, C]
        query = gst_feature
        x = x.contiguous().permute(0, 2, 1) #[B, C, T]
        x = self.activation(self.input_transform(x))
        x = x.contiguous().permute(0, 2, 1)#[B, T, C]
        key_0 = self.activation(self.embed_key_0(x))
        value_0 = self.activation(self.embed_value_0(x))
        query_0 = self.activation(self.embed_query_0(query))
        out_0 = self.attention_layer_0(query_0, key_0, value_0)

        key_1 = self.activation(self.embed_key_1(out_0))
        value_1 = self.activation(self.embed_value_1(out_0))
        query_1 = self.activation(self.embed_query_1(query))
        out_1 = self.attention_layer_1(query_1, key_1, value_1)
        

        key_2 = self.activation(self.embed_key_2(out_1))
        value_2 = self.activation(self.embed_value_2(out_1))
        query_2 = self.activation(self.embed_query_2(query))
        out_2 = self.attention_layer_2(query_2, key_2, value_2)
        

        key_3 = self.activation(self.embed_key_3(out_2))
        value_3 = self.activation(self.embed_value_3(out_2))
        query_3 = self.activation(self.embed_query_3(query))
        out_3 = self.attention_layer_3(query_3, key_3, value_3)

        out_0 = out_0.contiguous().permute(0, 2, 1)
        out_1 = out_1.contiguous().permute(0, 2, 1)
        out_2 = out_2.contiguous().permute(0, 2, 1)
        out_3 = out_3.contiguous().permute(0, 2, 1)
        return (out_0, out_1, out_2, out_3) 


# class ProsoStyleEasy(nn.Module):
#     def __init__(self,in_dim, hidden_dim=[512, 512, 256, 256, 128, 128], gst_dim=512, head_num=4):
#         super().__init__()
#         self.input_transform = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim[0], kernel_size=5, stride=3, padding=2)
#         self.activation =nn.LeakyReLU(negative_slope=0.2)
            
#         self.layer_0 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[0])
#         self.norm_0 = nn.InstanceNorm1d(hidden_dim[0])
#         self.layer_1 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[1])
#         self.norm_1 = nn.InstanceNorm1d(hidden_dim[1])
#         self.layer_2 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[2])
#         self.norm_2 = nn.InstanceNorm1d(hidden_dim[2])
#         self.layer_3 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[3])
#         self.norm_3 = nn.InstanceNorm1d(hidden_dim[3])
#         self.layer_4 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[4])
#         self.norm_4 = nn.InstanceNorm1d(hidden_dim[4])
#         self.layer_5 = nn.Linear(in_features=hidden_dim[0]+512, out_features=hidden_dim[5])
#         self.norm_5 = nn.InstanceNorm1d(hidden_dim[5])
        
        
#     def forward(self, x, gst_feature):
#         # [B, T, C]
#         x = x.contiguous().permute(0, 2, 1) #[B, C, T]
#         x = self.activation(self.input_transform(x))
#         x = x.contiguous().permute(0, 2, 1)#[B, T, C]
#         x = torch.cat([x, gst_feature], dim=2)

#         out_0 = self.activation(self.norm_0(self.layer_0(x).contiguous().permute(0, 2, 1)))
#         out_1 = self.activation(self.norm_1(self.layer_1(x).contiguous().permute(0, 2, 1)))
#         out_2 = self.activation(self.norm_2(self.layer_2(x).contiguous().permute(0, 2, 1)))
#         out_3 = self.activation(self.norm_3(self.layer_3(x).contiguous().permute(0, 2, 1)))
#         out_4 = self.activation(self.norm_4(self.layer_4(x).contiguous().permute(0, 2, 1)))
#         out_5 = self.activation(self.norm_5(self.layer_5(x).contiguous().permute(0, 2, 1)))

#         return (out_0, out_1, out_2, out_3, out_4, out_5) 
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
        return (out_0, out_1, out_2, out_3) 




# class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = resnet18(pretrained=True)
        # self.layers = list(self.backbone.children())[:-1]
        # self.backbone = nn.Sequential(*self.layers)
        self.expression_net = ExpressionLossNet()
        self.emotion_checkpoint = torch.load("/home/songyifei9/code/prosody/StyleProsody/mellotron/data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt")['state_dict']
        self.emotion_checkpoint['linear.0.weight'] = self.emotion_checkpoint['linear.weight']
        self.emotion_checkpoint['linear.0.bias'] = self.emotion_checkpoint['linear.bias']
        m, u = self.expression_net.load_state_dict(self.emotion_checkpoint, strict=False)
        for name, param in self.expression_net.named_parameters():
            param.requires_grad = False
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(8192, 512, kernel_size=1),
            nn.LeakyReLU(0.2)
        )
        # self.lstm = torch.nn.LSTM(input_size = 512, batch_first=True, dropout=0.1,
        #     hidden_size=512, num_layers=2, bidirectional=True)
        self.lstm = nn.Sequential(
            WaveNet(512, 10)
        )

        

    def forward(self, frames, shuffle=False):
        """
            frames:[B, T, C, H, W]
        """
        B, T = frames.shape[0], frames.shape[1]
        frames = frames.contiguous().view(B*T, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        # features = self.backbone(frames)
        features = self.expression_net(frames)
        features = features.contiguous().view(features.shape[0], features.shape[1], 1, 1)
        features = self.reduce_dim(features)
        features = features.contiguous().view(B, T, features.shape[1])
        features = features.contiguous().permute(0, 2, 1)
        out = self.lstm(features)[:, :, -1]
        if not shuffle:
            return features, out
        else:
            shuffle_features = shuffle_tensor(features)
            negative_out = self.lstm(features)[:, :, -1]
            return features, out, negative_out
            
if __name__=='__main__':
    # model = ProsoStyle(in_dim=2)
    model =ProsoStyleEasy(2)
    x = torch.randn((2, 12, 2))
    gst = torch.randn((2, 4, 512))
    out = model(x, gst)
    for s in out:
        print(s.shape)

    # model = FrameEncoder()
    # x = torch.randn((2, 16, 3, 256, 256))
    # features, out = model(x)
    # print(features.shape, out.shape)
    

