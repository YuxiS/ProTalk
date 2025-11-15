import torch
import numpy as np
import torch
from torchvision import transforms
from proso_features import Energy
import os
from yin import compute_yin
import audio_wav2lip
from hparams import create_hparams
import cv2
import torchaudio
import librosa
from stylemodel import ProsoResNet
from utils import gaussian_kernel
from vqvae.generate import VAE as PoseGEN
import torch.nn.functional as F
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
from tqdm import tqdm
from face_utils.utils import write_video
import argparse
from multiprocessing import Pool
from deep3d.coeff_detector import CoeffDetector
from deep3d.extract_kp_videos import KeypointExtractor
from itertools import cycle
from PIL import Image
import deep3d
import sys
sys.path.append('.')
import pdb

os.environ['TORCH_HOME']='/remote-home/yfsong/.cache/torch/hub'

hparams = create_hparams('./hparams.yaml')

transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]
        )

energy_processor = Energy(hparams.filter_length, hparams.hop_length, hparams.win_length)

def process_ref_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(img)
    return img 

def getInitCoeff(img_path, keyPointDetector, CoeffDetector):
    # coeff_detector = CoeffDetector(opt)
    # kp_extractor = KeypointExtractor()
    image = Image.open(img_path)
    lm = keyPointDetector.extract_keypoint(image)
    predicted = CoeffDetector(image, lm)
    return predicted

def StructureCoeff(coeff_dynamic):
    _id = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 80), device=coeff_dynamic.get_device())
    _tex = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 80), device=coeff_dynamic.get_device())
    _gamma = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 27), device=coeff_dynamic.get_device())        
    _exp, _angle, _trans = coeff_dynamic[:, :, :64], coeff_dynamic[:, :, 64:67], coeff_dynamic[:, :, 67:] 
    coeff = torch.cat([_id, _exp, _tex,  _angle, _gamma, _trans], dim=2)
    return coeff


def get_f0(audio, sampling_rate=22050, frame_length=1024,
            hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(
        audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad
    f0 = np.array(f0, dtype=np.float32)
    f0 = np.nan_to_num(f0, copy=True, nan=0.)
    return f0

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

def process_ref_audio(audio_path, sampling_rate=22050, fps=30):
    if not os.path.exists(audio_path):
        return None
    # pdb.set_trace()
    print(hparams.mean_std_root)
    mfcc_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mfcc_mean_wild.npy'))).view(1, -1).float()
    mfcc_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mfcc_std_wild.npy'))).view(1, -1).float()
    frame_n_samples =  int(sampling_rate/fps)
    audio = audio_wav2lip.load_wav(audio_path , sampling_rate)
    shifted_n_samples = 0
    n_frames = len(audio)//frame_n_samples
    audio = audio[:n_frames*frame_n_samples]
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        # pdb.set_trace()
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), 
        sample_frequency=sampling_rate, use_energy=True, num_ceps=80, num_mel_bins=80)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        rms = librosa.feature.rms(curr_samples, sampling_rate).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(curr_samples, 
                                                 sampling_rate).reshape(-1)
        curr_feat = np.concatenate((curr_mfccs, rms, zcr))
        curr_feats.append(curr_feat)
    if len(curr_feats)==0:
         print(audio_path)
         return None, None, None
    melspec = np.stack(curr_feats, axis=0)
    melspec = torch.from_numpy(melspec).float()
    melspec = (melspec-mfcc_mean)/mfcc_std 
    f0 = get_f0(np.array(audio), sampling_rate, hparams.filter_length, hparams.hop_length,
             hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
    f0 = torch.from_numpy(f0).unsqueeze(0)
    f0 = (f0-f0.min())/(f0.max()-f0.min())
    energy = energy_processor.get_energy(np.array(audio))
    energy = torch.from_numpy(energy).unsqueeze(0)
    energy = (energy-energy.min())/(energy.max()-energy.min())

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
    f0 = f0.transpose(0, 1)
    energy = energy.transpose(0, 1)
    return melspec, f0, energy

def inference(opts):
    args, ref_data, device_id = opts
    device = torch.device(int(device_id))
    audio_exp_model = ProsoResNet(hparams, 244, 2, 64)
    # audio_exp_model = ProsoLSTM(hparams, 244, 2, 64)
    audio_exp_model.load_state_dict(torch.load('/remote-home/yfsong/code/ProTalk/weights/style/All/ALL_MEAD_WILD2024-01-04-04_19/checkpoint_epoch_79.pth',
                                                map_location='cpu')['audio_model'])
    # pose_sampler = PoseGEN(80, 1024, 
    #     hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
    #     hparams.PoseModel.beta, 
    #     '/remote-home/yfsong/code/ProTalk/weights/vqvae/VQVAE-window-2024-01-09-06_30/vqvae_epoch_999.pth',
    #     '/remote-home/yfsong/code/ProTalk/weights/ablia/headrnn_no_prosody/PoseSampler2024-01-23-09_30/pose_sampler_epoch_399.pth')
    pose_sampler = PoseGEN(hparams.PoseModel.in_dim, 1024, 
        hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
        hparams.PoseModel.beta, 
        '/remote-home/yfsong/code/ProTalk/weights/vqvae/VQVAE-window-2024-01-09-06_30/vqvae_epoch_999.pth',
        '/remote-home/yfsong/code/ProTalk/weights/headrnn/PoseSampler2024-01-11-04_52/pose_sampler_epoch_399.pth') # For Prosody
    # pose_sampler.load_state_dict(torch.load(args.model_weight, map_location='cpu')['pose_model'])
    pirender = PIRenderFaceGenerator()
    pirender.load_state_dict(torch.load(args.pirender_weight)['net_G_ema'])
    ####################################################################
    coeff_detector = CoeffDetector(args)
    keypoint_detector = KeypointExtractor()
    base_coeff = getInitCoeff(ref_data['ref_img'], keypoint_detector, coeff_detector)
    base_3d_coeff, base_crop_coeff = base_coeff['Coeff'], base_coeff['Trans'][None, :]
    base_3d_coeff = torch.from_numpy(base_3d_coeff).to(device)
    base_crop_coeff = base_crop_coeff.to(device)
    # print(base_3d_coeff.shape)
    # base_crop_coeff = torch.from_numpy(base_crop_coeff).to(device)
    ############################################################################
    with torch.no_grad():
    ###############
        audio_exp_model = audio_exp_model.to(device)
        pose_sampler = pose_sampler.to(device)
        pirender = pirender.to(device)
        audio_exp_model.eval()
        pose_sampler.eval()
        pirender.eval()
        ################
        preprocessor = PIRenderPreprocessor(semantic_radius=13)
        poseprocessor = PIRenderPostProcessor()
        pose_smooth_kernel = gaussian_kernel(hparams.PoseModel.pose_dim, hparams.PoseModel.pose_dim, 5, std=3).to(device)
        face_smooth_kernel = gaussian_kernel(64, 64, 5, std=3).to(device)
        coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean_wild.npy'))).view(1, 1, -1).float().to(device)
        coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std_wild.npy'))).view(1, 1, -1).float().to(device)
        ##################################################
        base_3d_coeff = (base_3d_coeff-coeff_mean)/coeff_std
        # print(base_crop_coeff)
        base_crop_coeff[:, 3] = base_crop_coeff[:, 3]/base_crop_coeff[:, 0]
        base_crop_coeff[:, 4] = base_crop_coeff[:, 4]/base_crop_coeff[:, 0]
        base_crop_coeff = base_crop_coeff[:, 2:].to(device)
        ##############################################
        # print(ref_data)
        mel, f0, energy = process_ref_audio(ref_data['ref_audio'])
        if mel is None:
            return 0
        ref_img = process_ref_img(ref_data['ref_img'])
        #################################
        mel = mel.to(device)
        f0 = f0.to(device)
        energy = energy.to(device) 
        ref_img = ref_img.to(device)
        audio_length = [mel.shape[0]]
        mel = mel.unsqueeze(0)
        f0  = f0.unsqueeze(0)
        energy = energy.unsqueeze(0)
        ref_img = ref_img.unsqueeze(0)
        # pdb.set_trace()
        ##################################
        local_proso = torch.cat((f0, energy), dim=2)
        coeff_exp = audio_exp_model(mel, local_proso, audio_length)
        coeff_pose = pose_sampler(local_proso, audio_length, step=4)
        ##############################################
        if coeff_pose.shape[1]<coeff_exp.shape[1]:
            coeff_exp = coeff_exp[:, :coeff_pose.shapee[1], :]
        elif coeff_pose.shape[1]>coeff_exp.shape[1]:
            coeff_pose = coeff_pose[:, :coeff_exp.shape[1], :]
        ##############################################
        coeff_pose_smooth = coeff_pose.contiguous().permute(0, 2, 1)
        coeff_pose_smooth = F.conv1d(coeff_pose_smooth, weight=pose_smooth_kernel, stride=1, padding=0, groups=hparams.PoseModel.pose_dim)
        coeff_pose_smooth = coeff_pose_smooth.contiguous().permute(0, 2, 1)
        # coeff_pose[:, 3:-5, :] = coeff_pose_smooth
        start =  coeff_pose_smooth[:, 0, :].repeat([1, 2, 1])
        end = coeff_pose_smooth[:, -1, :].repeat([1, 2, 1])
        coeff_pose = torch.cat([start, coeff_pose_smooth, end], dim=1)
        # ################################################################
        coeff_exp = coeff_exp.contiguous().permute(0, 2, 1)
        coeff_exp_smooth = F.conv1d(coeff_exp, weight=face_smooth_kernel, stride=1, padding=0, groups=64)
        coeff_exp_smooth = coeff_exp_smooth.permute(0, 2, 1)
        start_exp = coeff_exp_smooth[:, 0, :].repeat([1, 2, 1])
        end_exp = coeff_exp_smooth[:, -1, :].repeat([1, 2, 1])
        coeff_exp = torch.cat([start_exp, coeff_exp_smooth, end_exp], dim=1)
        ################################################################
        ####################################################################
        # pdb.set_trace()
        coeff_exp = coeff_exp #+ base_3d_coeff[:, :, 80:144]
        base_coeff_pose = torch.cat([base_3d_coeff[:, :, 224:227], base_3d_coeff[:, :, 254:257]], dim=2)
        # coeff
        coeff_pose[:, :, :6] = (coeff_pose[:, :, :6]-coeff_pose[:, 0, :6])*0.6  + base_coeff_pose # 让头部动作更明显
        coeff_pose[:, :, -3:] = (coeff_pose[:, :, -3:]-coeff_pose[:, 0, -3:])*0.6 + base_crop_coeff
        # coeff_pose[:, :, :6] = (coeff_pose[:, :, :6]-coeff_pose[:, 0, :6])  + base_coeff_pose # 让头部动作更明显
        # coeff_pose[:, :, -3:] = (coeff_pose[:, :, -3:]-coeff_pose[:, 0, -3:]) + base_crop_coeff
        ####################################################################

        coeff_vect = torch.cat([coeff_exp, coeff_pose[:, :, :6]], dim=2) 
        coeff_crop = coeff_pose[:, :, -3:]
        coeff = StructureCoeff(coeff_vect)*coeff_std+coeff_mean
        coeff_dict = get_coeff_dict(coeff[0], coeff_crop[0])

        semantic_init = preprocessor.prepare_coeffs(coeff_dict).to(device).float()   
        results = []
        for i in range(len(semantic_init)):
            img = pirender(ref_img, semantic_init[i].unsqueeze(0))
            results.append(img.cpu())
        result_video = torch.stack(results, dim=0)
        basename = os.path.basename(ref_data['ref_audio']).replace('.wav', '.mp4')
        result_video = poseprocessor.recover_video(result_video)
        video_path = os.path.join(args.save_dir, basename)
        write_video(video_path, result_video.cpu().numpy())

        os.makedirs(os.path.join(args.save_dir, 'coeff'), exist_ok=True)
        coeff_path = os.path.join(args.save_dir, 'coeff', basename[:-4]+'.npy')
        np.save(coeff_path, coeff.cpu().numpy())
        return 1


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--ref_img', type=str, default='ref_img.jpg')
    args.add_argument('--driven_audio', type=str, default='driven_audio.txt')
    args.add_argument('--save_dir', type=str,  default='/remote-home/yfsong/code/ProTalk/test_res/') 
    args.add_argument('--model_weight', type=str, default='/remote-home/yfsong/code/ProTalk/weights/style/checkpoint_epoch_99.pth')
    args.add_argument('--pirender_weight', type=str, default='/remote-home/yfsong/code/prosody/vico_challenge_baseline-main/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt')
    args.add_argument('--mfcc_mean_std_root',type=str, default='/remote-home/share/yfsong/MEAD_VIDEO/feat_mean_std')
    ################################################################

    # args.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    args.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    args.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')
    args.add_argument('--checkpoints_dir', type=str, default='/remote-home/yfsong/code/prosody/StyleProsody/mellotron/deep3d/checkpoints', help='models are saved here')
    args.add_argument('--bfm_folder', type=str, default='./deep3d/BFM')
    args.add_argument('--isTrain', type=bool, default=False)
    args.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    args.add_argument('--focal', type=float, default=1015.)
    args.add_argument('--center', type=float, default=112.)
    args.add_argument('--camera_d', type=float, default=10.)
    args.add_argument('--z_near', type=float, default=5.)
    args.add_argument('--z_far', type=float, default=15.)
    args.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    args.add_argument('--use_opengl', type=bool, nargs='?', const=True, default=False, help='use opengl context or not')
    args.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    args.add_argument('--init_path', type=str, default='/remote-home/yfsong/code/prosody/StyleProsody/mellotron/deep3d/checkpoints/resnet50-0676ba61.pth')
    args.add_argument('--use_last_fc', type=bool, default=False)
    args.add_argument('--epoch', type=str, default='20', help='which epoch to load? set to latest to use latest cached model')
    args.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    args.add_argument('--use_ddp', type=deep3d.util.str2bool, nargs='?', const=True, default=False, help='whether use distributed data parallel')
    ##################################################################
    parser = args.parse_args()
    torch.multiprocessing.set_start_method('spawn', force=True)
    ##############################################################
    ref_data = {
                'ref_img': '/remote-home/yfsong/code/ProTalk/test_res/test_3.jpg',
                'ref_audio': '/remote-home/yfsong/code/ProTalk/test_res/LJ001-0005.wav'
    }
    opts = parser
    inference([opts, ref_data, 0])
    #################################################################
    #################################CREMA_D##############################################
    # REF_IMGS_ROOT = '/remote-home/share/yfsong/CREMA-D/ref_imgs'
    # WAV_ROOT = '/remote-home/share/yfsong/CREMA-D/wav'
    # # parser.mfcc_mean_std_root = '/home/songyifei9/data/CREMA-D/feat_mean_std'
    # parser.save_dir = os.path.join(parser.save_dir, 'CREMA-D')
    # # hparams.mean_std_root = parser.mfcc_mean_std_root
    # os.makedirs(parser.save_dir, exist_ok=True)
    # with open('/remote-home/yfsong/code/ProTalk/data/CREMA_D.txt', mode='r') as f:
    #     test_videos = f.readlines()
    # test_videos = [v.strip() for v in test_videos]
    # test_videos = test_videos[:200]
    # ref_data = []
    # for v in test_videos:
    #     ref_img_path =os.path.join(REF_IMGS_ROOT, v.replace('.mat', '.jpg'))
    #     speech = v.split('-')[-1].replace('.mat', '.wav')
    #     ref_speech_path = os.path.join(WAV_ROOT, speech)
    #     ref_data.append(
    #         {
    #             'ref_img': ref_img_path,
    #             'ref_audio': ref_speech_path
    #         }
    #     )
    
    # pool = Pool(processes=2)
    # # process_list = []
    # opts = cycle([parser])
    # devices = cycle([0, 1, 2, 3])
    # # inference((parser, ref_data[0], 1))
    # for data in tqdm(pool.imap_unordered(inference, zip(opts, ref_data, devices)), total=len(ref_data)):
    #     None        
    # pool.close()
    # ############################RAVEDSS############################
    # REF_IMGS_ROOT = '/remote-home/share/yfsong/RAVEDSS/ref_imgs'
    # WAV_ROOT = '/remote-home/share/yfsong/RAVEDSS/wav_resample'
    # # parser.save_dir = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/compareStudy/res'
    # parser.save_dir =os.path.join(parser.save_dir, 'RAVEDSS')
    # # parser.mfcc_mean_std_root = '/home/songyifei9/data/RAVEDSS/feat_mean_std'
    # # parser.save_dir = '/remote-home/yfsong/code/prosody/StyleProsody/mellotron/compareStudy/new_res/RAVEDSS-lip-40-overlap4-129-s9-face-s5'
    # # hparams.mean_std_root = parser.mfcc_mean_std_root
    # os.makedirs(parser.save_dir, exist_ok=True)
    # with open('/remote-home/yfsong/code/ProTalk/data/RAVEDSS.txt', mode='r') as f:
    #     test_videos = f.readlines()
    # test_videos = [v.strip() for v in test_videos]
    # test_videos = test_videos[:200]
    # ref_data = []
    # for v in test_videos:
    #     id = v.split('-')[1]
    #     basename = os.path.basename(v)
    #     ref_img_path =os.path.join(REF_IMGS_ROOT, basename.replace('.mp4', '.jpg'))
    #     speech = v.replace('.mp4', '.wav')
    #     # pdb.set_trace()
    #     speech = '-'.join(speech.split('-')[3:])
    #     # pdb.set_trace()
    #     ref_speech_path = os.path.join(WAV_ROOT, id, '03-'+speech)
    #     ref_data.append(
    #         {
    #             'ref_img': ref_img_path,
    #             'ref_audio': ref_speech_path
    #         }
    #     )  
    # pool = Pool(processes=4)
    # opts = cycle([parser])
    # devices = cycle([0, 1, 2, 3])
    # for data in tqdm(pool.imap_unordered(inference, zip(opts, ref_data, devices)), total=len(ref_data)):
    #     None    
    # pool.close()
    


    
    
    



    










    


