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
# lstm_init_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Init_LSTM/LSTM2023-02-24-16_29/checkpoint_exp_epoch_94.pth'
# audio_exp_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/Exp_coeff/Exp_Upper2023-02-24-18_35/checkpoint_exp_epoch_400.pth'
# finetune_path = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_ATTEN_GST_PRO2023-02-28-13_47/checkpoint_epoch_0.pth'
# style_resnet_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_EXP_STYLE2023-03-03-20_45/checkpoint_epoch_49.pth'
# resnet_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_NO_STYLE2023-03-04-14_31/checkpoint_epoch_49.pth'
# finetune_style = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD2023-03-06-17_01/checkpoint_epoch_1.pth'
# prosody_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_NO_PRO2023-04-27-23_13/checkpoint_epoch_60.pth'
no_prosody_weight = '/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_NO_PRO2023-04-27-23_13/checkpoint_epoch_60.pth'
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
    # audio_exp_model = ProsoResNet(hparams, 244, 2, 64)
    # audio_exp_model.load_state_dict(torch.load(prosody_weight, map_location='cpu')['audio_model'])
    audio_exp_model = ResNet(244, 64)
    audio_exp_model.load_state_dict(torch.load(no_prosody_weight, map_location='cpu')['audio_model'])
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
    audio_exp_model = audio_exp_model.cuda()
    # audio_init_model = audio_init_model.cuda()
    pirender = pirender.cuda()
    pirender.eval()
    audio_exp_model.eval()
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
            crop = coeff_crop[:, :, :]
            # crop = crop.repeat([1, coeff_crop.shape[1], 1])
            ##############################################################################
            # # coeff_exp_pred_lstm_init = audio_init_model(mel, audio_length)
            # coff_vect_pred_lstm_init = torch.cat([coeff_exp, coeff_pose], dim=2)
            # coeff_3dmm_lstm_init = StructureCoeff(coeff_static, coff_vect_pred_lstm_init)*coeff_std+coeff_mean
            # # coeff_3dmm_lstm_init = StructureCoeff(coeff_static, coeff_dynamic)*coeff_std+coeff_mean
            # coeff_dict_lstm_init = get_coeff_dict(coeff_3dmm_lstm_init[0], crop[0])
            # landmarks_init, _, _, _ = mesh_render.forward(coeff_3dmm_lstm_init.contiguous().view(-1, coeff_3dmm_lstm_init.shape[-1]))
            # semantic_init = preprocessor.prepare_coeffs(coeff_dict_lstm_init).cuda().float()
            # init_result_imgs = []
            # for i in tqdm(range(len(semantic_init))):
            #     # pdb.set_trace()
            #     init_result_img = pirender(ref_imgs, semantic_init[i].unsqueeze(0))
            #     init_result_imgs.append(init_result_img.cpu())
            # init_result_imgs = torch.stack(init_result_imgs, dim=0)
            # landmarks_init = landmarks_init.cpu().numpy()
            
            # os.makedirs('res/facial_style/landmarks/GT', exist_ok=True)
            # landmarks_file = os.path.join('res/facial_style/landmarks/GT', id+'_'+base_name+'.npy')
            # np.save(landmarks_file, landmarks_init)

            # init_result_imgs = poseprocessor.recover_video(init_result_imgs)
            # init_result_imgs = init_result_imgs.cpu().numpy()
            # os.makedirs('res/facial_style/GT', exist_ok=True)
            # init_video_name = os.path.join('res/facial_style/GT', id+'_'+base_name+'.mp4')
            # write_video(init_video_name, init_result_imgs)

            #######################################################################################
            # coff_exp_pred_exp = audio_exp_model(mel, pose_proso, audio_length)
            coff_exp_pred_exp = audio_exp_model(mel, audio_length)
            #######################smoooth#################################
            coff_exp_pred_smooth = coff_exp_pred_exp .contiguous().permute(0, 2, 1)
            coff_exp_pred_smooth = F.conv1d(coff_exp_pred_smooth, weight=smooth_kernel, stride=1, padding=0, groups=64)
            coff_exp_pred_smooth = coff_exp_pred_smooth.contiguous().permute(0, 2, 1)
            start = coff_exp_pred_smooth[:, 0, :].repeat([1, 2, 1])
            end = coff_exp_pred_smooth[:, -1, :].repeat([1, 2, 1])
            coff_exp_pred_exp = torch.cat([start, coff_exp_pred_smooth, end], dim=1)
            # coff_exp_pred_exp [:, 2:-2, :] = coff_exp_pred_smooth
            ################################################################
            coff_vect_pred_exp = torch.cat([coff_exp_pred_exp, coeff_pose], dim=2) 
            coeff_3dmm_pred_exp = StructureCoeff(coeff_static, coff_vect_pred_exp)*coeff_std+coeff_mean
            ####################################
            exp_params = coeff_3dmm_pred_exp.cpu().numpy()
            os.makedirs('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/prosody/facial/no_pro/params', exist_ok=True)
            params_file = os.path.join('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/prosody/facial/no_pro/params', id+'_'+base_name+'.npy')
            np.save(params_file, exp_params)
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
            os.makedirs('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/prosody/facial/no_pro', exist_ok=True)
            exp_video_name = os.path.join('/home/songyifei9/code/prosody/StyleProsody/mellotron/ablia__again/prosody/facial/no_pro', id+'_'+base_name+'.mp4')
            write_video(exp_video_name, exp_result_imgs)



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
    # os.makedirs('./test', exist_ok=True)
    # test(args, hparams, args.checkpoint_path
    torch.cuda.set_device(1)
    inference(args, hparams)
    