import torch
# from lpips import  LPIPS
from torch import nn
import torch.nn.functional as F
# import pdb
from Visual.expression_loss import ExpressionLossNet
from configparser import ConfigParser
from Visual.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from Visual import lossfunc
# from Discrim.S
import  torchvision.transforms.functional as F_v
import sys
import pdb
sys.path.append('.')
sys.path.append('..')

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class CoffLoss(nn.Module):
    def __init__(self, w_exp=1., w_delta_exp=1., 
                        w_pose=1., w_pose_delta=1.,):
        super(CoffLoss, self).__init__()
        self.l1 = nn.PairwiseDistance()
        # self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)
        # self.KL_div = nn.KLDivLoss(reduction="batchmean")
        self.w_exp = w_exp
        self.w_delta_exp = w_delta_exp
        self.w_pose = w_pose
        self.w_pose_delta = w_pose_delta

        # self.w_delta_frame = w_delta_frame
        # nn.LeakyReLU()

    def forward(self, exp_coeff, exp_target, pose_coeff, pose_target):
        # target.requires_grad = False
        # pre_frame_coeff_loss = self.l1(model_ouputs, target).mean()
        exp_loss = torch.mean(self.l1(exp_coeff, exp_target))

        # exp_cos = (1. - self.cos_sim(model_ouputs[:, :, :-6], target[:, :, :-6])).mean()
        exp_delta_pred = exp_coeff[:, 1:, :]-exp_coeff[:, :-1, :]
        exp_delta_gt = exp_target[:, 1:, :]-exp_target[:, :-1, :]
        exp_delta_loss = torch.mean(self.l1(exp_delta_pred, exp_delta_gt))

        pose_loss = torch.mean(self.l1(pose_coeff[:, :, -3:], pose_target[:, :, -3:]))
        # trans_loss = torch.mean(self.l1(pose_coeff[:, :, 3:], pose_target[:, :, 3:]))
        pose_delta_loss = torch.mean(self.l1((pose_coeff[:, 1:, :]-pose_coeff[:, :-1, :]), (pose_target[:, 1:, :]-pose_target[:, :-1, :])))
        
        loss = self.w_exp * exp_loss + self.w_delta_exp * exp_delta_loss + \
             self.w_pose*pose_loss + self.w_pose_delta * pose_delta_loss
        # loss = self.w_pose*pose_loss
        return loss, exp_loss, exp_delta_loss, pose_loss, pose_delta_loss

class ImgLoss(nn.Module):
    def __init__(self, w_l1):
        super(ImgLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        self.w_l1 = w_l1

    def forward(self, img_pred, img_gt):
        # img_gt.requires_grad = False
        l1_loss = self.l1_loss(img_pred, img_gt)*self.w_l1
        return l1_loss

class ExpLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {
            'landmark':0.0,
            'relative_landmark': 0,
            'lip_landmarks': 0.05, 
            'expression':30.,
            'lipread':10.
        }
        self.expression_net = ExpressionLossNet()
        self.emotion_checkpoint = torch.load("data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt")['state_dict']
        self.emotion_checkpoint['linear.0.weight'] = self.emotion_checkpoint['linear.weight']
        self.emotion_checkpoint['linear.0.bias'] = self.emotion_checkpoint['linear.bias']
        # self.lipsyncnet = 
        self.l1 = nn.L1Loss()
        m, u = self.expression_net.load_state_dict(self.emotion_checkpoint, strict=False)
        config = ConfigParser()

        # config.read('configs/lipread_config.ini')
        # self.lip_reader = Lipreading(
        #     config
        # )
        # self._crop_width = 48
        # self._crop_height = 48
        # self._window_margin = 12
        # self._start_idx = 48
        # self._stop_idx = 68
        # crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        # self.mouth_transform = Compose([
        #     Normalize(-1.0,  2.0),
        #     CenterCrop(crop_size),
        #     Normalize(mean, std),
        #     Identity()]
        # )
        self.expression_net.eval()
        # self.lip_reader.eval()
        for param in self.expression_net.parameters():
            param.requires_grad = False
        # for param in self.lip_reader.parameters():
        #     param.requires_grad = False
        # self.lip_reader.model.eval()
    
    def forward(self, real_landmarks, fake_landmarks, pred_faces, real_faces, 
        real_landmarks_frames,fake_landmarks_frames):
        """
            ld: B*T*K*68*2
            face: B*8*3*W*H
            ld_frames:B, 

        """
        B, T, C, W, H = real_faces.shape
        # real_landmarks_frames = []
        # fake_landmarks_frames = []
        # for b in range(B):
        #     real_landmarks_frames.append(real_landmarks[b, frames_index[b], ...])
        #     fake_landmarks_frames.append(fake_landmarks[b, frames_index[b], ...])
        # real_landmarks_frames = torch.cat(real_landmarks_frames, dim=0)
        # fake_landmarks_frames = torch.cat(fake_landmarks_frames, dim=0)
        pred_faces = pred_faces.contiguous().view(-1, pred_faces.shape[-3], pred_faces.shape[-2], pred_faces.shape[-1])
        real_faces = real_faces.contiguous().view(-1, real_faces.shape[-3], real_faces.shape[-2], real_faces.shape[-1])
        # pdb.set_trace()
        real_landmarks_delta = (real_landmarks[:, 1:, ...]-real_landmarks[:, :-1, ...]).view(-1, real_landmarks.shape[-2], real_landmarks.shape[-1])
        fake_landmarks_delta = (fake_landmarks[:, 1:, ...]-fake_landmarks[:, :-1, ...]).view(-1, fake_landmarks.shape[-2], fake_landmarks.shape[-1])
        real_landmarks = real_landmarks.view(-1, real_landmarks.shape[-2], real_landmarks.shape[-1])
        fake_landmarks = fake_landmarks.view(-1, fake_landmarks.shape[-2], fake_landmarks.shape[-1])
        losses = {}

        losses['landmark'] = lossfunc.weighted_landmark_loss(fake_landmarks, real_landmarks)
        losses['relative_landmark'] = lossfunc.relative_landmark_loss(fake_landmarks, real_landmarks)
        losses['lip_landmarks'] = F.mse_loss(fake_landmarks_delta[:,48:68,:2],real_landmarks_delta[:,48:68,:2])

        emotion_features_pred = self.expression_net(pred_faces)

        with torch.no_grad():
            emotion_features_gt = self.expression_net(real_faces)

        losses['expression'] = F.mse_loss(emotion_features_pred, emotion_features_gt)
        # real_faces = real_faces.view(B, T, C, W, H)
        # pred_faces = pred_faces.view(B, T, C, W, H)
        # mouths_gt = self.cut_mouth(real_faces, real_landmarks_frames)
        # mouths_pred = self.cut_mouth(pred_faces, fake_landmarks_frames)
        
        
        # mouths_gt = self.mouth_transform(mouths_gt)
        # mouths_pred = self.mouth_transform(mouths_pred)
        
        # mouths_gt = mouths_gt.contiguous().view(B, T, mouths_gt.shape[-2], mouths_gt.shape[-1])
        # mouths_pred = mouths_pred.contiguous().view(B, T, mouths_gt.shape[-2], mouths_gt.shape[-1])
        # lip_features_gt = self.lip_reader.model.encoder(
        #     mouths_gt,
        #     None,
        #     extract_resnet_feats=True
        # )

        # lip_features_pred = self.lip_reader.model.encoder(
        #     mouths_pred,
        #     None,
        #     extract_resnet_feats=True
        # )

        # lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        # lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        
        # losses['lipread'] = F.l1_loss(lip_features_pred, lip_features_gt)

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key] * self.cfg[key]
        losses['all_loss'] = all_loss
         
        return losses


    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []

        # landmarks = landmarks * 112 + 112
        # for frame_idx,frame in enumerate(images):
        #     window_margin = min(self._window_margin // 2, frame_idx, landmarks.shape[1] - 1 - frame_idx)
        #     smoothed_landmarks = landmarks[:,frame_idx-window_margin:frame_idx + window_margin + 1, ...].mean(dim=1)
        #     smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=1)
        #     pdb.set_trace()
        #     center_x, center_y = torch.mean(smoothed_landmarks[:, :, self._start_idx:self._stop_idx, :], dim=0)
        # pdb.set_trace()
        for b in range(landmarks.shape[0]):
            mouth_sub_seq = []
            for frame_index in range(landmarks.shape[1]):
                # pdb.set_trace()
                center_x, center_y = torch.mean(landmarks[b, frame_index, self._start_idx:self._stop_idx, :], dim=0)
                center_x = center_x.round()
                center_y = center_y.round()

                height = self._crop_height//2
                width = self._crop_width//2

                threshold = 5

                if convert_grayscale:
                    img = F_v.rgb_to_grayscale(images[b, frame_index, ...])
                else:
                    img = images[b, frame_index, ...]
                # pdb.set_trace()
                if center_y - height < 0:
                    center_y = height
                if center_y - height < 0 - threshold:
                    raise Exception('too much bias in height')
                if center_x - width < 0:
                    center_x = width
                if center_x - width < 0 - threshold:
                    raise Exception('too much bias in width')

                if center_y + height > img.shape[-2]:
                    center_y = img.shape[-2] - height
                if center_y + height > img.shape[-2] + threshold:
                    raise Exception('too much bias in height')
                if center_x + width > img.shape[-1]:
                    center_x = img.shape[-1] - width
                if center_x + width > img.shape[-1] + threshold:
                    raise Exception('too much bias in width')

                mouth = img[..., int(center_y - height): int(center_y + height),
                                    int(center_x - width): int(center_x + round(width))]
                # pdb.set_trace()
                mouth_sub_seq.append(mouth)
                
            mouth_sub_seq = torch.stack(mouth_sub_seq, dim=0)    
            mouth_sequence.append(mouth_sub_seq)
        # pdb.set_trace()
        mouth_sequence = torch.stack(mouth_sequence,dim=0)

        return mouth_sequence

if __name__ == '__main__':
    # inputs_1 = torch.randn((2, 3, 4, 5))
    # inputs_2 = torch.randn((2, 3, 4, 5))
    # cos = nn.CosineSimilarity(dim=3, eps=1e-6)
    # out = 1.-torch.mean(cos(inputs_1, inputs_2))
    # print(out)
    # mse = nn.MSELoss()
    # print(mse(inputs_1, inputs_2))
    # import time
    from hparams import create_hparams
    from torch.utils.data import DataLoader
    from CoeffDataset import TextMelLoader, TextMelCollate
    hparams = create_hparams()
    trainset = TextMelLoader(hparams.training_files, hparams)
    collate_fn = TextMelCollate(hparams)
    train_sampler = None
    shuffle = False
    train_loader = DataLoader(trainset, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=2, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    it = next(iter(train_loader))
    mel, f0, coff_padded, video_length, frame_index = it
    mel = mel.cuda()
    f0 = f0.cuda()
    for key, value in coff_padded.items():
        coff_padded[key] = value.cuda()
    video_length = video_length.cuda()
    frame_index = frame_index.cuda()

    frames_real = torch.randn((2, 8, 3, 224, 224)).cuda()
    frames_fake = torch.randn((2, 8, 3, 224, 224)).cuda()
    landmarks_pred = coff_padded['landmarks'].clone()
    exp_net = ExpLoss().cuda()
    B,T = frame_index.shape
    real_landmarks_frames = []
    fake_landmarks_frames = []
    for b in range(B):
        real_landmarks_frames.append(coff_padded['landmarks'][b, frame_index[b], ...])
        fake_landmarks_frames.append(landmarks_pred[b, frame_index[b], ...])
    real_landmarks_frames = torch.stack(real_landmarks_frames)
    fake_landmarks_frames = torch.stack(fake_landmarks_frames)
    # pdb.set_trace()
    loss = exp_net(landmarks_pred, coff_padded['landmarks'],frames_fake, 
    frames_real, real_landmarks_frames, fake_landmarks_frames)
    print(loss)