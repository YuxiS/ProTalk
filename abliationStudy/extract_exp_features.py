import torch
import torch.nn as nn
import sys
sys.path.append('.')
sys.path.append('..')
from torchvision import transforms
from Visual.expression_loss import ExpressionLossNet
from CoeffDataset import read_video

COEFF_ROOT = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/res/facial_exp'
TARGET = '/home/songyifei9/code/prosody/StyleProsody/mellotron/abliationStudy/visual/facial_exp'


if __name__=='__main__':
    expression_net = ExpressionLossNet()
    emotion_checkpoint = torch.load("/home/songyifei9/code/prosody/StyleProsody/mellotron/data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt")['state_dict']
    emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
    emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
    m, u = expression_net.load_state_dict(emotion_checkpoint, strict=False)
    frame_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]
        )
    
    
