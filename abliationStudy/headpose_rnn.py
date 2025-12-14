import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('..')
sys.path.append('.')
from deep3d.models.face_render import Face_render
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
from vqvae.generate import VAE as PoseGEN
from vqvae.pose_sampler import PoseSampler
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
from face_utils.utils import write_video
import argparse
import pdb
import numpy as np
from PIL import Image
import audio_wav2lip
from collections import OrderedDict
import torch.nn.functional as F
import shutil
import scipy.io as scio
import cv2
from utils import read_mats
from yin import compute_yin
from utils import gaussian_kernel
import ffmpeg
from tqdm import tqdm
torch.cuda.set_device(0)
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
pirender_weight = '/remote-home/yfsong/code/prosody/vico_challenge_baseline-main/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
finetun_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD2023-04-13-11_02/checkpoint_epoch_5.pth'
# lstm_init = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/ALL_MEAD2023-02-08-20_07/checkpoint_epoch_129.pth'
# lstm_init_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Init_LSTM/LSTM2023-02-17-18_36/checkpoint_exp_epoch_124.pth'
# audio_exp_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Exp_coeff/Exp_Upper2023-02-18-21_36/checkpoint_exp_epoch_149.pth'
vae_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/vqvae/VQVAE-Res-2023-04-14-19_54/vqvae_epoch_39.pth'
pose_sampler_path= '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/posesampler/PoseSampler2023-04-16-21_35/pose_sampler_epoch_89.pth'
pose_sampler_mfcc_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints/posesampler/PoseSampler2023-04-12-20_40/pose_sampler_epoch_299.pth'
def read_img(img_path):
    im = Image.open(img_path).convert('RGB')
    W, H = im.size
    # if to_tensor:
    im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    im = (im - 0.5)/0.5
    # lm = torch.tensor(lm).unsqueeze(0)
    return im 


def prepare_model():
    pass

def prepare_dataloader(hparams):
    testset = TextMelLoader(hparams.testing_files, hparams, mode='test')
    # collate_fn = TestTextMelCollate(hparams) 
    test_loader = DataLoader(testset, batch_size=1, collate_fn=collate_fn_test, shuffle=False)
    return test_loader

# def write_video(frames, video_file):
#     fps = 29.97  # 视频帧率
#     size = (256, 256)  # 需要转为视频的图片的尺寸
#     video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
#     for img in frames:
#         # image_path = data_path + "%010d_color_labels.png" % (i + 1)
#         # print(image_path)
#         img = img.astype(np.float32)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         img = np.uint8(img)
#         video.write(img)
#     video.release()
#     cv2.destroyAllWindows()

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
    # audio_exp_model = Audio_Exp_Model(hparams)
    # audio_exp_model.load_state_dict(torch.load(audio_exp_path, map_location='cpu')['audio_model_with_exp'])
    ##############################################
    # audio_init_model = AudioEncoder(hparams)
    # dict = dict = torch.load(lstm_init_path, map_location=torch.device('cpu'))
    # new_dict = OrderedDict()
    # for k,v in dict['LSTM_init'].items():
    #         if k.split('.')[0] =='module':
    #             k = '.'.join(k.split('.')[1:])
    #         new_dict[k] = v
    # audio_init_model.load_state_dict(new_dict)
    ####################################################
    pose_sampler = PoseSampler(2, 9, 256)
    pose_sampler.load_state_dict(torch.load('/remote-home/yfsong/code/prosody/StyleProsody/mellotron/checkpoints/PoseSampler_rnn2023-07-31-13_25/pose_sampler_epoch_499.pth'))
    # pose_sampler = PoseGEN(hparams.PoseModel.in_dim, hparams.PoseModel.n_embeddings, 
    #     hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
    #     hparams.PoseModel.beta, vae_weight, pose_sampler_path)
    # pose_sampler.load_state_dict(torch.load(finetun_weight, map_location='cpu')['pose_model'])
    # pose_sampler.load_state_dict(torch.load(pose_sampler_path, map_location='cpu'))
    ######################################################
    pirender = PIRenderFaceGenerator()
    pirender.load_state_dict(torch.load(pirender_weight)['net_G_ema'])
    ###################################################
    # audio_exp_model = audio_exp_model.cuda()
    # audio_init_model = audio_init_model.cuda()
    pose_sampler = pose_sampler.cuda()
    pirender = pirender.cuda()
    pirender.eval()
    # audio_exp_model.eval()
    # audio_init_model.eval()
    pose_sampler.eval()
    preprocessor = PIRenderPreprocessor(semantic_radius=13)
    poseprocessor = PIRenderPostProcessor()
    smooth_kernel = gaussian_kernel(hparams.PoseModel.pose_dim, hparams.PoseModel.pose_dim, 5).cuda()
    dataloader = prepare_dataloader(hparams)
    coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean.npy'))).view(1, 1, -1).float().cuda()
    coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std.npy'))).view(1, 1, -1).float().cuda()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
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
            # if not os.path.exists('test/MEAD_AUDIO/{}_{}.wav'.format(id, base_name)):
            #     shutil.copy(audio_path[0], 'test/MEAD_AUDIO/{}_{}.wav'.format(id, base_name))
            base_pose = coeff_dynamic[:, 0, -6:]
            base_crop = coeff_crop[:, 0, :]
            pose_proso = torch.cat((f0, energy), dim=2)
            coff_pose_pred = pose_sampler(pose_proso, audio_length)
            # coff_pose_pred = pose_sampler(mel[:, :, :80], audio_length)
            ##########################################
            coff_pose_pred_smooth = coff_pose_pred.contiguous().permute(0, 2, 1)
            coff_pose_pred_smooth = F.conv1d(coff_pose_pred_smooth, weight=smooth_kernel, stride=1, padding=0, groups=hparams.PoseModel.pose_dim)
            coff_pose_pred_smooth = coff_pose_pred_smooth.contiguous().permute(0, 2, 1)
            start =  coff_pose_pred_smooth[:, 0, :].repeat([1, 2, 1])
            end = coff_pose_pred_smooth[:, -1, :].repeat([1, 2, 1])
            coff_pose_pred = torch.cat([start, coff_pose_pred_smooth, end], dim=1)
            ###########################################
            # params_pose = coff_pose_pred.cpu().numpy()
            if coeff_dynamic.shape[1]>coff_pose_pred.shape[1]:
                coeff_dynamic = coeff_dynamic[:, :coff_pose_pred.shape[1], :]
            else:
                coff_pose_pred = coff_pose_pred[:, :coeff_dynamic.shape[1] ,:]
            ########################################################
            coff_crop_pred = coff_pose_pred[:, :, -3:]+base_crop
            coeff_pose_pred = coff_pose_pred[:, :, :6]*2 + base_pose
            ####################################################
            coeff_vect_pred = torch.cat((coeff_dynamic[:, :, :64], coeff_pose_pred), dim=2)
            # coeff_3dmm_real = StructureCoeff(coeff_static, coeff_dynamic)*coeff_std+coeff_mean
            coeff_3dmm_pred = StructureCoeff(coeff_static, coeff_vect_pred)*coeff_std+coeff_mean
            # coeff_dict_real = get_coeff_dict(coeff_3dmm_real[0], coeff_crop[0])
            coeff_dict_fake = get_coeff_dict(coeff_3dmm_pred[0], coff_crop_pred[0])
            # semantic_real = preprocessor.prepare_coeffs(coeff_dict_real).cuda().float()
            semantic_fake = preprocessor.prepare_coeffs(coeff_dict_fake).cuda().float()
            # real_result_imgs = []
            pred_result_imgs = []
            for i in tqdm(range(len(semantic_fake))):
                # real_result_img = pirender(ref_imgs, semantic_real[i].unsqueeze(0))
                # real_result_imgs.append(real_result_img.cpu())
                pred_result_img = pirender(ref_imgs, semantic_fake[i].unsqueeze(0))
                pred_result_imgs.append(pred_result_img.cpu())
            # real_result_imgs = torch.stack(real_result_imgs, dim=0)
            pred_result_imgs = torch.stack(pred_result_imgs, dim=0)
            # result_imgs = torch.cat([real_result_imgs, pred_result_imgs], dim=-1)
            # result_imgs = poseprocessor.recover_video(result_imgs)
            # result_imgs = result_imgs.cpu().numpy()
            # real_result_imgs = poseprocessor.recover_video(real_result_imgs).cpu().numpy()
            pred_result_imgs = poseprocessor.recover_video(pred_result_imgs).cpu().numpy()
            os.makedirs('/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia/head_rnn/param', exist_ok=True)
            params_pose_files_name = os.path.join('/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia/head_rnn/param', id+'_'+base_name+'.npy')
            np.save(params_pose_files_name, coeff_3dmm_pred.cpu().numpy())
            os.makedirs('/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia/head_rnn', exist_ok=True)
            video_name = os.path.join('/remote-home/yfsong/code/prosody/StyleProsody/mellotron/new_res/new_ablia/head_rnn', id+'_'+base_name+'.mp4')
            write_video(video_name, pred_result_imgs) 
            # os.makedirs('res/headpose/real', exist_ok=True)
            # video_name = os.path.join('res/headpose/real', id+'_'+base_name+'.mp4')
            # write_video(video_name, real_result_imgs)



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
    parser.add_argument('--bfm_folder', type=str, default='./deep3d/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    parser.add_argument('--use_last_fc', type=bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='deep3d/checkpoints/resnet50-0676ba61.pth')


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
    # os.makedirs('./test', exist_ok=True)
    # test(args, hparams, args.checkpoint_path
    inference(args, hparams)
    