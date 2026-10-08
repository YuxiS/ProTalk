import random
import os
import re
import numpy as np
import torch
import torch.utils.data
# import librosa
# import layers
from utils import load_wav_to_torch, load_filepaths_and_text, read_mats
from yin import compute_yin
# from PIL import Image
from utils import to_gpu
from proso_features import Energy
from runtime_utils import minmax_normalize
from torchvision import transforms
import audio_wav2lip
from glob import glob
import scipy.io as scio
import cv2
import random
import pdb
import tqdm
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

img_transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


class AudioDataset(torch.utils.data.Dataset):
    """
        1) loads audio, text
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, mode, dataset='MEAD'):
        self.dataset = dataset
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.coff_root = hparams.coff_root
        self.mode = mode
        self.frame_per_video = hparams.frames_per_video
        self.hparams = hparams
        self.id_name = hparams.id_name
        self.energy = Energy(self.filter_length, self.hop_length, hparams.win_length)
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])]
        )
        self.get_std_mean()

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

    def get_energy(self, audio):
        e = self.energy.get_energy(audio)
        return e
        
    def get_data(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        ids = audiopath.split(os.sep)[-2]
        mel, f0, energy = self.get_mel_and_f0(audiopath)
        coeff_dynamic, coeff_static, coeff_crop = self.get_coff3d(audiopath)
        
        # if coeff_dynamic.shape[0] < mel.shape[1]//3:
        #     mel = mel[:, :coeff_dynamic.shape[0]*3]
        # elif coeff_dynamic.shape[0] > mel.shape[1]//3:
        #     coeff_dynamic = coeff_dynamic[:mel.shape[1]//3, :]
        #     coeff_static = coeff_static[:mel.shape[1]//3, :]
        # video_length = coff_3d['id'].shape[0]
        # mel = mel.transpose(0, 1)
        f0 = f0.transpose(0, 1)
        energy = energy.transpose(0, 1)
        frames, frames_index = self.get_imgs(audiopath, coeff_dynamic.shape[0])
        # ref_img is the first image
        ###############################
        if f0.isnan().any().item():
            f0[torch.isnan(f0)]=0.
        if energy.isnan().any().item():
            energy[torch.isnan(energy)]=0.
        ################################
        if coeff_dynamic.shape[0]>mel.shape[0]:
            coeff_dynamic = coeff_dynamic[:]
        return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, frames, frames_index
    
    def get_test_data(self, audiopath_and_text):
        # pdb.set_trace()
        audiopath, text = audiopath_and_text
        ids = audiopath.split(os.sep)[-2]
        mel, f0, energy = self.get_mel_and_f0(audiopath)
        f0 = f0.transpose(0, 1)
        energy = energy.transpose(0, 1)
        coff_d, coff_s, coff_c = self.get_coff3d(audiopath)
        ref_img = self.get_ref_img(audiopath)
        return mel, f0, energy, coff_d, coff_s, coff_c, ref_img, audiopath

    def get_ref_img(self, audio_path):
        if self.dataset =='MEAD':
            ids = audio_path.split(os.sep)[-2]
            sub_dir = os.path.split(audio_path)[-1][:-4]
            video_path = os.path.join(self.hparams.video_root, ids, sub_dir+'.mp4')
        elif self.dataset =='Wild':
            basename = os.path.basename(audio_path).replace('.wav', '.mp4')
            video_path = os.path.join(self.hparams.video_root, basename)
        # pdb.set_trace()
        ref_img = read_video(video_path)[0]
        ref_img = self.transforms(ref_img)
        return ref_img


    def get_imgs(self, audio_path, video_length):
        if self.dataset =='Wild':
        #################Wild###################
            basename = os.path.basename(audio_path).replace('.wav', '.mp4')
            video_path = os.path.join(self.hparams.video_root, basename)
        #############MEAD#######################
        elif self.dataset =='MEAD':
            ids = audio_path.split(os.sep)[-2]
            sub_dir = os.path.split(audio_path)[-1][:-4]
            video_path = os.path.join(self.hparams.video_root, ids, sub_dir+'.mp4')
        ###################RAVEDSS###################################
        # if os.path.exists(os.path.join(self.img_root, ids+'_aligned', '01-'+sub_dir[3:])):
        #     video_path =  os.path.join(self.img_root, ids+'_aligned', '01-'+sub_dir[3:])
        # elif os.path.exists(os.path.join(self.img_root, ids+'_aligned', '02-'+sub_dir[3:])):
        #     video_path = os.path.join(self.img_root, ids+'_aligned', '02-'+sub_dir[3:])
        # imgfile_list = sorted(glob(os.path.join(video_path, '*.jpg')))
        ####################CREMA-D#################################
        # basename = os.path.basename(audio_path).replace('.wav', '.mp4')
        # video_path = os.path.join(self.hparams.video_root, 'CREMA-D-VideoFlash-'+basename)
        frames = read_video(video_path)
        assert len(frames)==video_length, "{} : Video length doesn't match audio length!".format(audio_path)
        # imgfile_list = sorted(glob(os.path.join(self.img_root, ids+'_aligned', sub_dir, "*.jpg")))
        frames_index = np.arange(1, video_length, step=(video_length//self.hparams.frames_per_video))[:self.hparams.frames_per_video]
        if video_length%self.hparams.frames_per_video == 0:
            frames_index[-1] = frames_index[-1]-1
        img_list = [self.transforms(frames[0])]
        for index in frames_index:
            # img = cv2.imread(imgfile_list[index])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = frames[index]
            img = self.transforms(img)
            img_list.append(img)
        imgs = torch.stack(img_list)
        return imgs, frames_index


    def get_coff3d(self, audio_path):
        if self.dataset =='MEAD':
            ids = audio_path.split(os.sep)[-2]
            sub_dir = os.path.split(audio_path)[-1][:-4]
            coeff_path = os.path.join(self.coff_root, ids, sub_dir+'.mat')
        #####################################################
        elif self.dataset =='Wild':
            basename = os.path.basename(audio_path).replace('.wav', '.mat')
            coeff_path = os.path.join(self.hparams.coff_root, basename)
        coeff = scio.loadmat(coeff_path)['coeff']
        coeff_crop = scio.loadmat(coeff_path)['transform_params']
        coeff_crop = torch.from_numpy(coeff_crop)
        coeff = torch.from_numpy(coeff)
        coeff = (coeff - self.coeff_mean)/self.coeff_std
        coeff_dynamic = torch.cat((coeff[:, 80:144], coeff[:, 224:227], coeff[:, 254:]), dim=1)
        coeff_static = torch.cat((coeff[:, :80], coeff[:, 144:224], coeff[:, 227:254]), dim=1)
        coeff_crop[:, 3] = coeff_crop[:, 3]/coeff_crop[:, 0]
        coeff_crop[:, 4] = coeff_crop[:, 4]/coeff_crop[:, 1] 
        coeff_crop = coeff_crop[:, 2:]
        return coeff_dynamic, coeff_static, coeff_crop

        ##############################################
#         if os.path.exists(os.path.join(self.coff_root, ids, '01-'+sub_dir[3:])):
#             sub_dir = '01-'+sub_dir[3:]
#         elif os.path.exists(os.path.join(self.cof=f_root, ids, '02-'+sub_dir[3:])):
#             sub_dir = '02-'+sub_dir[3:]
        ##############################################
        # pdb.set_trace()
        # if os.path.exists(os.path.join(self.coff_root, ids, '01-'+sub_dir[3:])):
        #     video_path =  os.path.join(self.coff_root, ids, '01-'+sub_dir[3:])
        # elif os.path.exists(os.path.join(self.coff_root, ids, '02-'+sub_dir[3:])):
        #     video_path = os.path.join(self.coff_root, ids, '02-'+sub_dir[3:])
        # coff = read_mats(os.path.join(video_path, 'epoch_20_000000'))
        # ##########################################
        # # coff = read_mats(os.path.join(self.coff_root, ids, sub_dir, 'epoch_20_000000'))
        # for key, value in coff.items():
        #     coff[key]=torch.from_numpy(value)
        # coff['exp'] = (coff['exp'] - self.coeff_mean[:, 80:144])/self.coeff_std[:, 80:144]
        # coff['angle'] = (coff['angle'] - self.coeff_mean[:, 224:227])/self.coeff_std[:, 224:227]
        # coff['trans'] = (coff['trans'] - self.coeff_mean[:, 254:])/self.coeff_std[:, 254:]
        # coeff_dynamic = torch.cat([coff['exp'], coff['angle'], coff['trans']], dim=1)
        # coeff_static = torch.cat([coff['id'], coff['tex'], coff['gamma']], dim=1)
        # return coeff_dynamic, coeff_static


    def get_mel_and_f0(self, filepath):
        if self.dataset == 'MEAD':
        ##################################################
            id = filepath.split(os.sep)[-2]
            basename = os.path.basename(filepath)[:-4]
            melspec = torch.from_numpy(np.load(os.path.join(self.hparams.mfcc_dir, id, basename+'.npy'))).float()
        #################################################
        elif self.dataset =='Wild':
            basename = os.path.basename(filepath).replace('.wav', '.npy')
            melspec = torch.from_numpy(np.load(os.path.join(self.hparams.mfcc_dir,  basename))).float()

        wav = audio_wav2lip.load_wav(filepath, self.sampling_rate)
        melspec = (melspec-self.mfcc_mean)/self.mfcc_std  ### normalize
        f0 = self.get_f0(np.array(wav), self.sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0).unsqueeze(0)
        # f0 = f0[:, :melspec.size(0)]
        # pdb.set_trace()
        energy = self.get_energy(np.array(wav))
        energy = torch.from_numpy(energy).unsqueeze(0)
        #############
        f0 = minmax_normalize(f0)
        energy = minmax_normalize(energy)
        ##############

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

        # energy = energy[:, :melspec.size(1)]
        return melspec, f0, energy



    def __getitem__(self, index):
        if self.mode == 'train':
            return self.get_data(self.audiopaths_and_text[index])
        else:
            return self.get_test_data(self.audiopaths_and_text[index])
        
    def __len__(self):
        return int(len(self.audiopaths_and_text))

    def get_std_mean(self):
        root = self.hparams.mean_std_root
        if self.dataset =='MEAD':
            self.coeff_mean = torch.from_numpy(np.load(os.path.join(root,  'mean.npy'))).view(1, -1).float()
            self.coeff_std = torch.from_numpy(np.load(os.path.join(root,  'std.npy'))).view(1, -1).float()
            self.mfcc_mean = torch.from_numpy(np.load(os.path.join(root,  'mfcc_mean.npy'))).view(1, -1).float()
            self.mfcc_std = torch.from_numpy(np.load(os.path.join(root,  'mfcc_std.npy'))).view(1, -1).float()
        elif self.dataset =='Wild':
            self.coeff_mean = torch.from_numpy(np.load(os.path.join(root,  'mean_wild.npy'))).view(1, -1).float()
            self.coeff_std = torch.from_numpy(np.load(os.path.join(root,  'std_wild.npy'))).view(1, -1).float()
            self.mfcc_mean = torch.from_numpy(np.load(os.path.join(root,  'mfcc_mean_wild.npy'))).view(1, -1).float()
            self.mfcc_std = torch.from_numpy(np.load(os.path.join(root,  'mfcc_std_wild.npy'))).view(1, -1).float()
            
def collate_fn(batch):
    mel = [b[0] for b in batch]
    f0 = [b[1] for b in batch]
    energy = [b[2] for b in batch]
    coeff_dynamic = [b[3] for b in batch]
    coeff_static = [b[4] for b in batch]
    coeff_crop = [b[5] for b in batch]
    frames = [b[6] for b in batch]
    frames_index =[b[7] for b in batch]
    audio_length = [m.shape[0] for m in mel]
    
    mel = torch.nn.utils.rnn.pad_sequence(mel, batch_first=True)
    f0 = torch.nn.utils.rnn.pad_sequence(f0, batch_first=True)
    energy = torch.nn.utils.rnn.pad_sequence(energy, batch_first=True)
    coeff_dynamic = torch.nn.utils.rnn.pad_sequence(coeff_dynamic, batch_first=True)
    coeff_static = torch.nn.utils.rnn.pad_sequence(coeff_static, batch_first=True)
    coeff_crop = torch.nn.utils.rnn.pad_sequence(coeff_crop, batch_first=True)
    # pdb.set_trace()
    frames = torch.stack(frames)
    return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, frames, audio_length, frames_index

def collate_fn_test(batch): 
    mel = [b[0] for b in batch]
    f0 = [b[1] for b in batch]
    energy = [b[2] for b in batch]
    coeff_dynamic = [b[3] for b in batch]
    coeff_static = [b[4] for b in batch]
    coeff_crop = [b[5] for b in batch] 
    ref_imgs = [b[6] for b in batch]
    audio_names = [b[7] for b in batch]
    audio_length = [m.shape[0] for m in mel]
    
    mel = torch.nn.utils.rnn.pad_sequence(mel, batch_first=True)
    f0 = torch.nn.utils.rnn.pad_sequence(f0, batch_first=True)
    energy = torch.nn.utils.rnn.pad_sequence(energy, batch_first=True)
    coeff_dynamic = torch.nn.utils.rnn.pad_sequence(coeff_dynamic, batch_first=True)
    coeff_static = torch.nn.utils.rnn.pad_sequence(coeff_static, batch_first=True)
    coeff_crop = torch.nn.utils.rnn.pad_sequence(coeff_crop, batch_first=True)
    ref_imgs = torch.stack(ref_imgs)

    return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, audio_length, ref_imgs, audio_names

def collate_vq(batch):
    overlap = 2
    windows = 8
    mel = [b[0] for b in batch]
    f0 = [b[1] for b in batch]
    energy = [b[2] for b in batch]
    coeff_dynamic = []
    coeff_static = []
    coeff_crop = []
    for b in batch:
        # pdb.set_trace()
        # temp_dyna = b[3]-b[3][0]
        # coeff_dynamic.append(temp_dyna[1:, :])
        # temp_stat = b[4]-b[4][0]
        # coeff_static.append(temp_stat[1:, :])
        # temp_crop = b[5]-b[5][0]
        # coeff_crop.append(temp_crop[1:, :])
        ############################################
        coeff_dynamic.append(b[3])
        coeff_static.append(b[4])
        coeff_crop.append(b[5])
        ############################################
    audio_length = [m.shape[0] for m in mel]

    mel_list = []
    f0_list = []
    energy_list = []
    coeff_d_list = []
    coeff_s_list = []
    coeff_crop_list = []
    for i in range(len(mel)):
        for index in range(0, mel[i].shape[0]-windows, overlap):
            # pdb.set_trace()
            mel_list.append(mel[i][index:index+windows, :])
            f0_list.append(f0[i][index*3:index*3+windows*3, :])
            energy_list.append(energy[i][index*3:index*3+windows*3, :])
            coeff_d_list.append(coeff_dynamic[i][index:index+windows, :])
            coeff_s_list.append(coeff_static[i][index:index+windows, :])
            coeff_crop_list.append(coeff_crop[i][index:index+windows, :])
    # pdb.set_trace()
    mel_tensor = torch.stack(mel_list, dim=0)
    f0_tensor = torch.stack(f0_list, dim=0)
    energy_tensor = torch.stack(energy_list, dim=0)
    coeff_dynamic_tensor = torch.stack(coeff_d_list, dim=0)
    coeff_static_tensor = torch.stack(coeff_s_list, dim=0)
    coeff_crop_tensor = torch.stack(coeff_crop_list, dim=0)
    return mel_tensor, f0_tensor, energy_tensor, coeff_static_tensor, coeff_dynamic_tensor, coeff_crop_tensor


    # pdb.set_trace()
    # frames = torch.stack(frames)
    # return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, frames, audio_length, frames_index

def parse_facial_batch(batch):
    mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, audio_length, ref_imgs, audio_names = batch
    # mel, f0, coff_padded, mel_window, energy, frame_index, video_lengths, coff_init, frames_window, ids = batch
    # input_length = input_length.cuda(non_blocking=False)
    mel = mel.cuda(non_blocking=False)
    f0 = f0.cuda(non_blocking=False)
    coeff_dynamic = coeff_dynamic.cuda(non_blocking=False)
    coeff_static = coeff_static.cuda(non_blocking=False)
    energy = energy.cuda(non_blocking=False)
    coeff_crop = coeff_crop.cuda(non_blocking=False)
    ref_imgs = ref_imgs.cuda(non_blocking=False)
    return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, audio_length, ref_imgs, audio_names

def parse_head_batch(batch):
    mel_tensor, f0_tensor, energy_tensor, coeff_static_tensor, coeff_dynamic_tensor, coeff_crop_tensor = batch
    mel_tensor = mel_tensor.cuda(non_blocking=False)
    f0_tensor = f0_tensor.cuda(non_blocking=False)
    energy_tensor = energy_tensor.cuda(non_blocking=False)
    coeff_static_tensor = coeff_static_tensor.cuda(non_blocking=False)
    coeff_dynamic_tensor = coeff_dynamic_tensor.cuda(non_blocking=False)
    coeff_crop_tensor = coeff_crop_tensor.cuda(non_blocking=False)
    return mel_tensor, f0_tensor, energy_tensor, coeff_static_tensor, coeff_dynamic_tensor, coeff_crop_tensor
    
    

class DataPrefetcher():
    def __init__(self, loader, facial=True):
        self.loader = iter(loader)
        self.facial_flag = facial
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            if self.facial_flag:
                self.batch = parse_facial_batch(self.batch)
            else:
                self.batch = parse_head_batch(self.batch)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



def read_video(video_fn):
    reader = cv2.VideoCapture(video_fn)
    frames = []
    ret, frame = reader.read()
    while ret:
        img = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(img)
        ret, frame = reader.read()
    return frames

if __name__=='__main__':
    from hparams import create_hparams
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    hparams = create_hparams(yaml_file='/remote-home/yfsong/code/ProTalk/hparams.yaml')
    trainset = AudioDataset(hparams.training_files, hparams, mode='train')     
    loader = DataLoader(trainset, collate_fn=collate_fn, batch_size=128, drop_last=True)
    for batch in tqdm(loader):
        for i in batch:
            if isinstance(i, torch.Tensor):
                if torch.isnan(i).any():
                    print('kkkkkk')




    
