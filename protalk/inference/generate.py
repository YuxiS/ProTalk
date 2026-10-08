import torch
import numpy as np
from torchvision import transforms
from protalk.audio.prosody import Energy
import os
from protalk.audio.yin import compute_yin
import protalk.audio.wav2lip as audio_wav2lip
from protalk.config import create_hparams
from protalk.runtime import str2bool, minmax_normalize, align_coefficients
import cv2
import torchaudio
import librosa
from protalk.models.expression import ProsoResNet
from protalk.inference.smoothing import gaussian_kernel
from protalk.models.pose.generate import VAE as PoseGEN
import torch.nn.functional as F
from protalk.third_party.face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
from tqdm import tqdm
from protalk.third_party.face_utils.utils import write_video
import argparse
from multiprocessing import Pool
from protalk.third_party.deep3d.coeff_detector import CoeffDetector
from protalk.third_party.deep3d.extract_kp_videos import KeypointExtractor
from itertools import cycle
from PIL import Image
import protalk.third_party.deep3d as deep3d
import sys
import pdb

hparams = None

transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]
        )

energy_processor = None

def process_ref_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read reference image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(img)
    return img

def getInitCoeff(img_path, keyPointDetector, CoeffDetector):
    image = Image.open(img_path)
    lm = keyPointDetector.extract_keypoint(image)
    if lm is None:
        raise ValueError(f"No face detected in reference image: {img_path}")
    predicted = CoeffDetector(image, lm)
    return predicted

def StructureCoeff(coeff_dynamic):
    _id = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 80), device=coeff_dynamic.device)
    _tex = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 80), device=coeff_dynamic.device)
    _gamma = torch.ones((coeff_dynamic.shape[0], coeff_dynamic.shape[1], 27), device=coeff_dynamic.device)
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
    coeff ={}
    coeff['exp'] = coeff_vect[:, 80:144].cpu().numpy()
    coeff['angle'] = coeff_vect[:, 224:227].cpu().numpy()
    coeff['trans'] = coeff_vect[:, 254:257].cpu().numpy()
    coeff_crop[:, 1:] = coeff_crop[:, 1:]*256
    coeff['crop'] = coeff_crop.cpu().detach().numpy()
    return coeff

def process_ref_audio(audio_path, sampling_rate=22050, fps=30):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    mfcc_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mfcc_mean.npy'))).view(1, -1).float()
    mfcc_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mfcc_std.npy'))).view(1, -1).float()
    frame_n_samples =  int(sampling_rate/fps)
    audio = audio_wav2lip.load_wav(audio_path , sampling_rate)
    shifted_n_samples = 0
    n_frames = len(audio)//frame_n_samples
    audio = audio[:n_frames*frame_n_samples]
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
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
    if len(curr_feats) < 5:
        raise ValueError("Audio must contain at least five complete frames at 30 fps")
    melspec = np.stack(curr_feats, axis=0)
    melspec = torch.from_numpy(melspec).float()
    if not torch.isfinite(mfcc_std).all() or (mfcc_std <= 0).any():
        raise ValueError("MFCC standard deviations must be finite and positive")
    melspec = (melspec-mfcc_mean)/mfcc_std
    f0 = get_f0(np.array(audio), sampling_rate, hparams.filter_length, hparams.hop_length,
             hparams.f0_min, hparams.f0_max, hparams.harm_thresh)
    f0 = torch.from_numpy(f0).unsqueeze(0)
    f0 = minmax_normalize(f0)
    energy = energy_processor.get_energy(np.array(audio))
    energy = torch.from_numpy(energy).unsqueeze(0)
    energy = minmax_normalize(energy)

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
    global hparams, energy_processor
    args, ref_data, device_id = opts
    hparams = create_hparams(args.hparams)
    if args.mfcc_mean_std_root:
        hparams.mean_std_root = args.mfcc_mean_std_root
    energy_processor = Energy(hparams.filter_length, hparams.hop_length, hparams.win_length)
    for path in (ref_data['ref_img'], ref_data['ref_audio'], args.model_weight,
                 args.vae_weight, args.sampling_weight, args.pirender_weight):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required asset missing: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("The full Deep3D/PIRender pipeline requires a CUDA GPU")
    torch.cuda.set_device(int(device_id))
    device = torch.device(f"cuda:{device_id}")
    os.makedirs(args.save_dir, exist_ok=True)
    audio_exp_model = ProsoResNet(hparams, 244, 2, 64, load_gst=False)
    expression_checkpoint = torch.load(args.model_weight, map_location='cpu')
    audio_exp_model.load_state_dict(expression_checkpoint['audio_model'])
    pose_sampler = PoseGEN(hparams.PoseModel.in_dim, hparams.PoseModel.n_embeddings,
        hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
        hparams.PoseModel.beta,
        args.vae_weight,
        args.sampling_weight)
    pirender = PIRenderFaceGenerator()
    pirender.load_state_dict(torch.load(args.pirender_weight, map_location='cpu')['net_G_ema'])
    coeff_detector = CoeffDetector(args)
    keypoint_detector = KeypointExtractor(device=f'cuda:{device_id}')
    base_coeff = getInitCoeff(ref_data['ref_img'], keypoint_detector, coeff_detector)
    base_3d_coeff, base_crop_coeff = base_coeff['Coeff'], base_coeff['Trans'][None, :]
    base_3d_coeff = torch.from_numpy(base_3d_coeff).to(device)
    base_crop_coeff = base_crop_coeff.to(device)
    with torch.no_grad():
        audio_exp_model = audio_exp_model.to(device)
        pose_sampler = pose_sampler.to(device)
        pirender = pirender.to(device)
        audio_exp_model.eval()
        pose_sampler.eval()
        pirender.eval()
        preprocessor = PIRenderPreprocessor(semantic_radius=13)
        poseprocessor = PIRenderPostProcessor()
        pose_smooth_kernel = gaussian_kernel(hparams.PoseModel.pose_dim, hparams.PoseModel.pose_dim, 5, std=3).to(device)
        face_smooth_kernel = gaussian_kernel(64, 64, 5, std=3).to(device)
        coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean.npy'))).view(1, 1, -1).float().to(device)
        coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std.npy'))).view(1, 1, -1).float().to(device)
        if not torch.isfinite(coeff_std).all() or (coeff_std <= 0).any():
            raise ValueError("Coefficient standard deviations must be finite and positive")
        base_3d_coeff = (base_3d_coeff-coeff_mean)/coeff_std
        base_crop_coeff[:, 3] = base_crop_coeff[:, 3]/base_crop_coeff[:, 0]
        base_crop_coeff[:, 4] = base_crop_coeff[:, 4]/base_crop_coeff[:, 0]
        base_crop_coeff = base_crop_coeff[:, 2:].to(device)
        mel, f0, energy = process_ref_audio(ref_data['ref_audio'])
        if mel is None:
            return 0
        ref_img = process_ref_img(ref_data['ref_img'])
        mel = mel.to(device)
        f0 = f0.to(device)
        energy = energy.to(device)
        ref_img = ref_img.to(device)
        audio_length = [mel.shape[0]]
        mel = mel.unsqueeze(0)
        f0  = f0.unsqueeze(0)
        energy = energy.unsqueeze(0)
        ref_img = ref_img.unsqueeze(0)
        local_proso = torch.cat((f0, energy), dim=2)
        coeff_exp = audio_exp_model(mel, local_proso, audio_length)
        coeff_pose = pose_sampler(local_proso, audio_length, step=4)
        coeff_exp, coeff_pose = align_coefficients(coeff_exp, coeff_pose)

        coeff_exp = coeff_exp.contiguous().permute(0, 2, 1)
        coeff_exp_smooth = F.conv1d(coeff_exp, weight=face_smooth_kernel, stride=1, padding=0, groups=64)
        coeff_exp_smooth = coeff_exp_smooth.permute(0, 2, 1)
        start_exp = coeff_exp_smooth[:, 0, :].repeat([1, 2, 1])
        end_exp = coeff_exp_smooth[:, -1, :].repeat([1, 2, 1])
        coeff_exp = torch.cat([start_exp, coeff_exp_smooth, end_exp], dim=1)
        coeff_exp = coeff_exp + base_3d_coeff[:, :, 80:144]
        base_coeff_pose = torch.cat([base_3d_coeff[:, :, 224:227], base_3d_coeff[:, :, 254:257]], dim=2)
        pose_scale = args.pose_scale if args.pose_scale is not None else expression_checkpoint.get('metadata', {}).get('pose_scale', 2.)
        coeff_pose[:, :, :6] = coeff_pose[:, :, :6]*pose_scale + base_coeff_pose # 让头部动作更明显
        coeff_pose[:, :, -3:] = coeff_pose[:, :, -3:] + base_crop_coeff

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
        result_video = poseprocessor.recover_video(result_video)
        video_path = os.path.join(args.save_dir, 'temp.mp4')
        write_video(video_path, result_video.cpu().numpy())

