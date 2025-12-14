import torch
import numpy as np
import torch
from torchvision import transforms
import sys
sys.path.append('.')
sys.path.append('..')
# from model import Tacotron2
# from model_vae import ProsoNet
from deep3d.models.face_render import Face_render as DeepFace3D
from proso_features import Energy
from CoeffDataset import TextMelLoader, collate_fn_test, collate_fn
from torch.utils.data import DataLoader
from hparams import create_hparams
from deep3d.util.preprocess import align_img
from model import Audio_Exp_Model
from deep3d.util.load_mats import load_lm3d
from models_easy import AudioEncoder
import os
from train_exp_coeff_toImg import model_to_gpu, initModel, StructureCoeff
from models_easy import AudioEncoder
from stylemodel import ProsoResNet, ResNet
from vqvae.generate import VAE as PoseGEN
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
from face_utils.utils import write_video
from vqvae.generate import VAE as ExpVAE
from vqvae.models.vqvae import VQVAE
import argparse
import pdb
from PIL import Image
import audio_wav2lip
from collections import OrderedDict
import torch.nn.functional as F
import shutil
import scipy.io as scio
import cv2
from utils import read_mats, gaussian_kernel
from yin import compute_yin
import ffmpeg
from tqdm import tqdm
# torch.cuda.set_device(0)
pirender_weight = '/home/songyifei9/code/prosody/vico_challenge_baseline-main/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
# lstm_init = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/ALL_MEAD2023-02-08-20_07/checkpoint_epoch_129.pth'
lstm_init_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Init_LSTM/LSTM2023-02-24-16_29/checkpoint_exp_epoch_94.pth'
audio_exp_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Exp_coeff/Exp_Upper2023-02-24-18_35/checkpoint_exp_epoch_400.pth'
finetune_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_ATTEN_GST_PRO2023-02-28-13_47/checkpoint_epoch_0.pth'
style_resnet_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_EXP_STYLE2023-03-03-20_45/checkpoint_epoch_49.pth'
resnet_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_NO_STYLE2023-03-04-14_31/checkpoint_epoch_49.pth'
finetune_style = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD2023-03-06-17_01/checkpoint_epoch_1.pth'
exp_sampler_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/posesampler-exp/PoseSampler-exp2023-04-29-05_48/pose_sampler_epoch_129.pth'
vae_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/new_vqvae/VQVAE-Exp-2023-04-29-02_19/vqvae_epoch_149.pth' 
def read_img(img_path):
    im = Image.open(img_path).convert('RGB')
    W, H = im.size
    # if to_tensor:
    im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    im = (im - 0.5)/0.5
    # lm = torch.tensor(lm).unsqueeze(0)
    return im 


def prepare_dataloader(hparams):
    testset = TextMelLoader(hparams.testing_files, hparams, mode='test')
    test_loader = DataLoader(testset, batch_size=1, collate_fn=collate_fn_test, shuffle=False)
    return test_loader




def get_coeff_dict(coeff_vect, coeff_crop):
    # Pirender只需要预测的三个参数即可
    coeff ={}
    coeff['exp'] = coeff_vect[:, 80:144].cpu().numpy() 
    coeff['angle'] = coeff_vect[:, 224:227].cpu().numpy()
    coeff['trans'] = coeff_vect[:, 254:257].cpu().numpy()
    coeff_crop[:, 1:] = coeff_crop[:, 1:]*256
    coeff['crop'] = coeff_crop.cpu().detach().numpy()
    # pdb.set_trace()
    return coeff


def inference(opt, hparams):
    # audio_exp_model = VAE(244)
    audio_exp_model = ProsoResNet(hparams, 244, 2, 512).cuda()
    audio_exp_model.load_state_dict(torch.load(exp_sampler_weight))
    vae_model = VQVAE(64, 512, 256, 0.25).cuda()
    vae_model.encoder.load_state_dict(torch.load(vae_weight)['Encoder'])
    vae_model.decoder.load_state_dict(torch.load(vae_weight)['Decoder'])
    vae_model.vector_quantization.load_state_dict(torch.load(vae_weight)['CodeBook'])
    vae_model.eval()
    # audio_exp_model.load_state_dict(torch.load(style_resnet_weight, map_location='cpu')['audio_model'])
    # audio_exp_model = ResNet(244, 64)
    # audio_exp_model.load_state_dict(torch.load(resnet_weight, map_location='cpu')['audio_model'])
    #######################################
    # audio_exp_model = Audio_Exp_Model(hparams)
    # audio_exp_model.load_state_dict(torch.load(audio_exp_path, map_location='cpu')['audio_model_with_exp'])
    # audio_exp_model.load_state_dict(torch.load(finetune_path, map_location='cpu')['audio_model'])
    ##############################################
    # audio_init_model = AudioEncoder(hparams)
    # dict = torch.load(lstm_init_path, map_location=torch.device('cpu'))
    # new_dict = OrderedDict()
    # for k,v in dict['LSTM_init'].items():
    #         if k.split('.')[0] =='module':
    #             k = '.'.join(k.split('.')[1:])
    #         new_dict[k] = v
    # audio_init_model.load_state_dict(new_dict)
    ####################################################
    # pose_sampler = PoseGEN(hparams.PoseModel.in_dim, hparams.PoseModel.n_embeddings, 
    #     hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
    #     hparams.PoseModel.beta, hparams.PoseModel.vae_weight, hparams.PoseModel.sampling_weight)
    ######################################################
    pirender = PIRenderFaceGenerator()
    pirender.load_state_dict(torch.load(pirender_weight)['net_G_ema'])
    ###################################################
    # audio_exp_model = audio_exp_model.cuda()
    # audio_init_model = audio_init_model.cuda()
    pirender = pirender.cuda()
    pirender.eval()
    audio_exp_model.eval()
    # audio_init_model.eval()
    mesh_render = DeepFace3D(opt)
    preprocessor = PIRenderPreprocessor(semantic_radius=13)
    poseprocessor = PIRenderPostProcessor()
    smooth_kernel = gaussian_kernel(64, 64, 5, std=3).cuda()
    dataloader = prepare_dataloader(hparams)
    coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean.npy'))).view(1, 1, -1).float().cuda()
    coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std.npy'))).view(1, 1, -1).float().cuda()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # if i > 20:
            #     break
            mel = batch[0].cuda()
            f0 = batch[1].cuda()
            energy = batch[2].cuda()
            coeff_dynamic = batch[3].cuda()
            coeff_static = batch[4].cuda()
            coeff_crop = batch[5].cuda()
            audio_length = batch[6]
            ref_imgs = batch[7].cuda()
            audio_path = batch[8]
            id, base_name = audio_path[0].split(os.sep)[-2], audio_path[0].split(os.sep)[-1][:-4]
            pose_proso = torch.cat((f0, energy), dim=2)
            coeff_exp = coeff_dynamic[:, :, :64]
            coeff_pose = coeff_dynamic[:, :, 64:]
            # coeff_pose = coeff_pose.repeat([1, coeff_dynamic.shape[1], 1])
            crop = coeff_crop[:, :, :]
            # crop = crop.repeat([1, coeff_crop.shape[1], 1])

            prosody_data = torch.cat([f0, energy], dim=2) #[B, T, C]
            # prosody_data = mel
            proso_index = audio_exp_model(mel, prosody_data, audio_length)
            pred_index = proso_index[:, ::4, :]
            sampled_index = proso_index[:, ::4, :]
            res = []
            # pred_index = []
            B = prosody_data.shape[0]
            # pdb.set_trace()
            for t in range(sampled_index.shape[1]):
                sampled_index_temp = sampled_index[:, t, :]
                sampled_index_temp = sampled_index_temp.contiguous().view(-1, sampled_index_temp.shape[-1])
                min_index = torch.argmax(sampled_index_temp, dim=1)
                min_encodings = torch.zeros((min_index.shape[0], 512), dtype=torch.float).cuda()
                min_encodings.scatter_(1, min_index.unsqueeze(1), 1)
                e_weights = vae_model.vector_quantization.embedding.weight
                z_q = torch.matmul(min_encodings, e_weights)
                z_q = z_q.contiguous().view(B, -1, z_q.shape[-1])
                pose_seq_temp = vae_model.decoder(z_q)
                # res.append(pose_seq_temp)
                half = int(pose_seq_temp.shape[1]/2)
                if len(res)==0:
                    res.append(pose_seq_temp[:, :half, :])
                    res.append(pose_seq_temp[:, half:, :])
                else:
                    half = int(pose_seq_temp.shape[1]/2)
                    last = res.pop()
                    temp = (last + pose_seq_temp[:, :half, :])/2
                    res.append(temp)
                    res.append(pose_seq_temp[:, half:, :])
            # pdb.set_trace()
            coff_exp_pred_exp = torch.cat(res, dim=1)
            #############################################
            coff_exp_pred_smooth = coff_exp_pred_exp.contiguous().permute(0, 2, 1)
            coff_exp_pred_smooth = F.conv1d(coff_exp_pred_smooth, weight=smooth_kernel, stride=1, padding=0, groups=64)
            coff_exp_pred_smooth = coff_exp_pred_smooth.contiguous().permute(0, 2, 1)
            start = coff_exp_pred_smooth[:, 0, :].repeat([1, 2, 1])
            end = coff_exp_pred_smooth[:, -1, :].repeat([1, 2, 1])
            coff_exp_pred_exp = torch.cat([start, coff_exp_pred_exp, end], dim=1)
            if coff_exp_pred_exp.shape[1]>coeff_pose.shape[1]:
                coff_exp_pred_exp = coff_exp_pred_exp[:, :coeff_pose.shape[1], :]
            else:
                coeff_pose = coeff_pose[:, :coff_exp_pred_exp.shape[1], :]
            ################################################################
            coff_vect_pred_exp = torch.cat([coff_exp_pred_exp, coeff_pose], dim=2) 
            coeff_3dmm_pred_exp = StructureCoeff(coeff_static, coff_vect_pred_exp)*coeff_std+coeff_mean
            ####################################
            # exp_params = coeff_3dmm_pred_exp[0, :, 80:144].cpu().numpy()
            os.makedirs('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/exp_vq/params', exist_ok=True)
            params_file = os.path.join('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/exp_vq/params', id+'_'+base_name+'.npy')
            np.save(params_file, coeff_3dmm_pred_exp.cpu().numpy())
            #######################################
            # landmarks_pred, _, _, _ = mesh_render.forward(coeff_3dmm_pred_exp.contiguous().view(-1, coeff_3dmm_pred_exp.shape[-1]))
            coeff_dict_pred_exp = get_coeff_dict(coeff_3dmm_pred_exp[0], crop[0])
            semantic_exp = preprocessor.prepare_coeffs(coeff_dict_pred_exp).cuda().float()
            exp_result_imgs = []
            for i in tqdm(range(len(semantic_exp))):
                exp_result_img = pirender(ref_imgs, semantic_exp[i].unsqueeze(0))
                exp_result_imgs.append(exp_result_img.cpu())
            exp_result_imgs = torch.stack(exp_result_imgs, dim=0)

            # landmarks_pred = landmarks_pred.cpu().numpy()
            # os.makedirs('res/facial_style/landmarks/finetune_smooth_5', exist_ok=True)
            # landmarks_file = os.path.join('res/facial_style/landmarks/finetune_smooth_5', id+'_'+base_name+'.npy')
            # np.save(landmarks_file, landmarks_pred)

            exp_result_imgs = poseprocessor.recover_video(exp_result_imgs)
            exp_result_imgs = exp_result_imgs.cpu().numpy()
            os.makedirs('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/exp_vq', exist_ok=True)
            exp_video_name = os.path.join('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/exp_vq', id+'_'+base_name+'.mp4')
            write_video(exp_video_name, exp_result_imgs)


class DataProcess:
    def __init__(self, hparams) -> None:
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.hparams = hparams
        self.id_name = hparams.id_name
        self.cmudict = None
        self.energy = Energy(self.filter_length, self.hop_length, hparams.win_length)

        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]
        )
        self.get_std_mean()

    def get_std_mean(self):
        root = self.hparams.mean_std_root
        self.coeff_mean = torch.from_numpy(np.load(os.path.join(root,  'mean.npy'))).view(1, -1).float()
        self.coeff_std = torch.from_numpy(np.load(os.path.join(root,  'std.npy'))).view(1, -1).float()
        self.mfcc_mean = torch.from_numpy(np.load(os.path.join(root,  'mfcc_mean.npy'))).view(1, -1).float()
        self.mfcc_std = torch.from_numpy(np.load(os.path.join(root,  'mfcc_std.npy'))).view(1, -1).float()
        

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
                hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
            f0, harmonic_rates, argmins, times = compute_yin(
                audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
                harm_thresh)
            pad = int((frame_length / hop_length) / 2)
            f0 = [0.0] * pad + f0 + [0.0] * pad
            f0 = np.array(f0, dtype=np.float32)
            f0 = np.nan_to_num(f0, copy=True, nan=0.)
            return f0
    def get_mel_and_f0(self, filepath):
        id = filepath.split(os.sep)[-2]
        basename = os.path.basename(filepath)[:-4]
        wav = audio_wav2lip.load_wav(filepath, self.sampling_rate)
        # melspec = audio_wav2lip.melspectrogram(wav)
        melspec = torch.from_numpy(np.load(os.path.join(self.hparams.mfcc_dir, id, basename+'.npy'))).float()
        # pdb.set_trace()
        melspec = (melspec-self.mfcc_mean)/self.mfcc_std  ### normalize
        f0 = self.get_f0(np.array(wav), self.sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0).unsqueeze(0)
        # f0 = f0[:, :melspec.size(0)]
        # pdb.set_trace()
        energy = self.get_energy(np.array(wav))
        energy = torch.from_numpy(energy).unsqueeze(0)
        # 调整f0和energy的长度，使其和视频长度相匹配
        # assert energy.shape[1] == f0.shape[1], print(filepath, energy.shape[1], f0.shape[1])

        if energy.shape[1] != f0.shape[1]:
            if energy.shape[1] > f0.shape[1]:
                energy = energy[:, :f0.shape[1]]
            else:
                f0 = f0[:, :energy.shape[1]] 
                
        if energy.shape[1] != 3*melspec.shape[0]:
            if energy.shape[1] > 3*melspec.shape[0]:
                energy = energy[:, :3*melspec.shape[0]]
                f0 = f0[:, :3*melspec.shape[0]]
            else:
                padded = torch.zeros((1, 3*melspec.shape[0]-energy.shape[1]), dtype=torch.float32)
                energy = torch.cat([energy, padded], dim=1)
                f0 = torch.cat([f0, padded], dim=1)
        return melspec, f0, energy

def reference(hparams):
    energy_calculator = Energy(hparams.filter_length, hparams.hp_length, hparams.win_length)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, default='../hparams.yaml',
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('-c', '--checkpoint_path', type=str, 
    default='checkpoints/ALL_RAVDESS2023-01-10-21_47/checkpoint_img_epoch_649.pth',
                        required=False, help='checkpoint path')
    parser.add_argument('--infer_img', type=str, default='./test/000001.jpg')
    parser.add_argument('--infer_lm', type=str, default='./test/000001.txt')
    # parser.add_argument('--hparams', type=str, default='./hparams.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--distributed_run', type=bool, default=False)
    parser.add_argument('--multi_node', type=bool, default=False)
    parser.add_argument('--master_addr', type=str)
    parser.add_argument('--master_port',  type=int)
    parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--dataset_mode', type=str, default=None,
                        help='chooses how datasets are loaded. [None | flist]')
    parser.add_argument('--bfm_folder', type=str, default='/home/songyifei9/code/prosody/StyleProsody/mellotron/deep3d/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    parser.add_argument('--use_last_fc', type=bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='/home/songyifei9/code/prosody/StyleProsody/mellotron/deep3d/checkpoints/resnet50-0676ba61.pth')


    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.add_argument('--use_opengl', type=bool, nargs='?', const=True, default=False, help='use opengl context or not')


    # loss weights
    parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
    parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
    parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
    parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
    parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
    parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
    parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
    parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
    parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')
    parser.add_argument('--isTrain', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.set_defaults(
        focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
    )
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.mode = 'test'
    os.makedirs('./test', exist_ok=True)
    # test(args, hparams, args.checkpoint_path
    torch.cuda.set_device(2)
    inference(args, hparams)
    