import numpy as np
from scipy.io.wavfile import read
import torch
from glob import glob
import os
import random
import scipy.io as scio
import pdb
from tqdm import tqdm
import torch.nn as nn
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont
np.random.seed(1234)
import audio_wav2lip
IDS = ['M003', 'M005', 'M007', 'M009', 'M013',
     'W009', 'W014', 'W018', 'W023', 'W029']#, 'M019','M022', 'W016', 'W037']
# import torchaudio
# IDS = ['M019','M022', 'W016', 'W037']

# IDS = ['M005','M007', 'M009','M011','M013','M026', 'M027',
#        'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034']
# IDS = ['Actor_01','Actor_02','Actor_03','Actor_04','Actor_05',
#         'Actor_07','Actor_08','Actor_09','Actor_10']


def get_mel(wav_root, mel_root, sample_rate=22050, coeff_root = None):
    wav_files = glob(os.path.join(wav_root, '*.wav'))
    # wav_files = glob(os.path.join(wav_root, '*.wav'))
    os.makedirs(os.path.join(mel_root), exist_ok=True)
    for f in tqdm(wav_files):
        # wav = audio_wav2lip.load_wav(f, sample_rate)
        # melspec = audio_wav2lip.melspectrogram(wav)
        # mfcc = torchaudio.compliance.kaldi.mfcc()
        # melspec = audio_wav2lip.get_mels(f, coeff_root='/home/songyifei9/data/MEAD_VIDEO/Coeff_3D')
        if not os.path.exists(os.path.join(coeff_root, os.path.basename(f).replace('.wav', '.mat'))): # type: ignore
            print("Coeff file not exist!")
            continue
        melspec = audio_wav2lip.get_wild_mels(f, coeff_root=coeff_root)
        if melspec is None:
            print('{} is not exist!'.format(f))
            continue
        basename= os.path.basename(f)[:-4]
        np.save(os.path.join(mel_root, basename+'.npy'), melspec)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def audio_file_list(audio_dataroot, 
                    train_record_path='./data/data_train_wild.txt',
                    val_record_path='./data/data_val_wild.txt',
                    test_record_path='./data/data_test_wild.txt',
                   aligned_root='/home/songyifei9/data/MEAD/new_3d_coeff'):
    file_train = open(train_record_path, mode='w', encoding='utf8')
    file_val = open(val_record_path, mode='w', encoding='utf8')
    file_test = open(test_record_path, mode='w', encoding='utf8')
    
    audio_files = []
    # audio_files = glob(os.path.join(audio_dataroot, '*.wav'))
    ###########################################################
    temp_audio_files = glob(os.path.join(audio_dataroot, '*.wav'))
    for a in temp_audio_files:
        basename = os.path.splitext(os.path.basename(a))[0]+'.mat'
        v = os.path.join('/remote-home/share/yfsong/TalkingHeadDatasetsClean/coeff', basename)
        if os.path.exists(v):
            audio_files.append(a)
    print(len(audio_files))
    ############################################################
    # for ids in IDS:
    #     audio_files += glob(os.path.join(audio_dataroot, ids ,"*.wav"))
    # pdb.set_trace()
    random.shuffle(audio_files)
    # print(len(audio_files))
    audio_files = remove_empty_audio(audio_files, aligned_root)
    # print(len(audio_files))
    # pdb.set_trace()
    num_records = len(audio_files)
    for a in tqdm(audio_files[:int(0.9*num_records)]):
        ids = a.split(os.sep)[-2]
        _, name = os.path.split(a)
        # with open(os.path.join(txt_dataroot, ids+'_txt', name[:-3]+'txt'), mode='r') as f:
        #     txt = f.readline()
        txt = ' '
        file_train.write(a+'|'+txt+'\n')
    file_train.close()
    
    for a in tqdm(audio_files[int(0.9*num_records):int(0.95*num_records)]):
        ids = a.split(os.sep)[-2]
        _, name = os.path.split(a)
        # with open(os.path.join(txt_dataroot, ids+'_txt', name[:-3] + 'txt'), mode='r') as f:
        #     txt = f.readline()
        txt = ' '
        file_val.write(a + '|' + txt + '\n')
    file_val.close()

    for a in tqdm(audio_files[int(0.95*num_records):]):
        ids = a.split(os.sep)[-2]
        _, name = os.path.split(a)
        # with open(os.path.join(txt_dataroot, ids+'_txt', name[:-3] + 'txt'), mode='r') as f:
        #     txt = f.readline()
        txt = ' '
        file_test.write(a + '|' + txt + '\n')
    file_test.close()


def m4a_to_wav(source_root, target_root):
    files = glob(os.path.join(source_root, '*.m4a'))
    command = 'ffmpeg -i {} -acodec pcm_s16le -ac 1 -ar 23040 {}'
    for f in files:
        _, name = os.path.split(f)
        target_file = os.path.join(target_root, name[:-3]+'wav')
        os.system(command.format(f, target_file))


# def read_styles(data_root):
#     style_files = sorted(glob(os.path.join(data_root, '*.npy')))
#     styles = np.load(style_files[0])
#     styles = np.expand_dims(styles, axis=0)
#     for s in style_files[1:]:
#         data = np.load(s)
#         data = np.expand_dims(data, axis=0)
#         styles = np.concatenate([styles, data], axis=0)
#     return styles
# #
# def read_frames(video_path, frames_index):
#
#     for id in frames_index:
#         name = str(id).ljust(6, '0')
def read_mats(data_root):
    # print(data_root)
    mat_files = sorted(glob(os.path.join(data_root, '*.mat')))
    id = []
    exp = []
    tex = []
    angle = []
    gamma = []
    trans = []
    landmarks = []
    
    for m in mat_files:
        coff = scio.loadmat(m)
        id_temp = np.array(coff['id'])
        exp_temp = np.array(coff['exp'])
        tex_temp = np.array(coff['tex'])
        angle_temp = np.array(coff['angle'])
        gamma_temp = np.array(coff['gamma'])
        trans_temp = np.array(coff['trans'])
        landmarks_temp = np.array(coff['lm68']) 
        
        id.append(id_temp)
        exp.append(exp_temp)
        tex.append(tex_temp)
        angle.append(angle_temp)
        gamma.append(gamma_temp)
        trans.append(trans_temp)
        landmarks.append(landmarks_temp)
    # pdb.set_trace()
    id_series = np.concatenate(id, axis=0)
    exp_series = np.concatenate(exp, axis=0)
    tex_series = np.concatenate(tex, axis=0)
    angle_series = np.concatenate(angle, axis=0)
    gamma_series = np.concatenate(gamma, axis=0) #[T, gamma]
    trans_series = np.concatenate(trans, axis=0) #[T, d_tran]
    landmarks_series = np.concatenate(landmarks, axis=0)
    
    exp_series = exp_series
    angle_series = angle_series
    trans_series = trans_series
    return {'id':id_series,
            'exp': exp_series,
            'tex': tex_series,
            'gamma': gamma_series,
            'angle': angle_series,
            'trans': trans_series,
            'landmarks': landmarks_series}

def remove_empty_audio(audio_path_list, aligned_root):
    new_list = []
    for audio_path in audio_path_list:
        ###############Wild##################
        sub_dir = os.path.split(audio_path)[-1][:-4]
        frames = scio.loadmat(os.path.join(aligned_root, sub_dir+'.mat'))['coeff']
        #################MEAD#################
        # sub_dir = os.path.split(audio_path)[-1][:-4]
        # ids = audio_path.split(os.sep)[-2]
        # frames = scio.loadmat(os.path.join(aligned_root, ids, sub_dir+'.mat'))['coeff']
        ###################CREMA-D##################
        # basename = os.path.basename(audio_path).replace('.wav', '.mat')
        # if not os.path.exists(os.path.join(aligned_root, 'CREMA-D-VideoFlash-'+basename)):
        #     continue
        # frames = scio.loadmat(os.path.join(aligned_root, 'CREMA-D-VideoFlash-'+basename))['coeff']
        #####################################
        # if os.path.exists(os.path.join(aligned_root, ids, '01-'+sub_dir[3:])):
        #     frames_dir = os.path.join(aligned_root, ids, '01-'+sub_dir[3:])
        # elif os.path.exists(os.path.join(aligned_root, ids, '02-'+sub_dir[3:])):
        #     frames_dir = os.path.join(aligned_root, ids, '02-'+sub_dir[3:])
        # frames = glob(os.path.join(frames_dir, '**/*.mat'))
        # else:
        #     print(audio_path)
        
        ###################################
        # frames_dir = os.path.join(aligned_root, ids, sub_dir, 'epoch_20_000000')
        # frames = glob(os.path.join(frames_dir, '*.mat'))
        # pdb.set_trace()
        if len(frames) >= 20:
            new_list.append(audio_path)
        # else:
        #     print(audio_path)
    return new_list

def read_img_data_list(dataroot):
    ids = os.listdir(dataroot)
    images = []
    for id in tqdm(ids, leave=False):
        dirs = os.listdir(os.path.join(dataroot, id))
        for d in tqdm(dirs, leave=False):
            if os.path.isdir(os.path.join(dataroot, id, d)):
                img_temp = glob(os.path.join(dataroot, id, d,'*.jpg'))
                images += img_temp
    random.shuffle(images)
    images_train = images[:int(0.9*len(images))]
    images_val = images[int(0.9*len(images)):int(0.95*len(images))]
    images_test = images[int(0.95*len(images)):]
    with open('data/img_train.txt', mode='w') as f:
        for img in tqdm(images_train, leave=False):
            if img_coeff_is_exist(img): 
                f.write(img+'\n')
    with open('data/img_val.txt', mode='w') as f:
        for img in tqdm(images_val, leave=False):
            if img_coeff_is_exist(img):
                f.write(img+'\n')
    with open('data/img_test.txt', mode='w') as f:
        for img in tqdm(images_test, leave=False):
            if img_coeff_is_exist(img):
                f.write(img+'\n')

def img_coeff_is_exist(img_path, coff_dir='/home/songyifei9/data/MEAD/new_3d_coeff'):
    id = img_path.split(os.sep)[-3][:4]
    dirs = img_path.split(os.sep)[-2]
    coff_name = img_path.split(os.sep)[-1][:-3]+'mat'
    # pdb.set_trace()
    if os.path.exists(os.path.join(coff_dir, id, dirs, 'epoch_20_000000', coff_name)):
        return True
    # print(img_path)
    return False

def normalizeCoeff(coff_root, mean_std_root):
    coff_list = []
    # for id in tqdm(IDS):
        # pdb.set_trace()
    # files = glob(os.path.join(coff_root, id, '*.mat'))
    # files = glob(os.path.join(coff_root, '*.mat'))
    #####################################
    files = []
    temp_files = glob(os.path.join(coff_root, '*.mat'))
    for a in temp_files:
        basename = os.path.splitext(os.path.basename(a))[0]+'.mp4'
        v = os.path.join('/remote-home/share/yfsong/TalkingHeadDatasetsClean/videos_256', basename)
        if os.path.exists(v):
            files.append(a)
    ###############################################
    for f in tqdm(files):
        coff = np.array(scio.loadmat(f)['coeff'])
        coff_list.append(coff)

    
    coff_list = np.concatenate(coff_list, axis=0)
    result = {
        'mean': np.mean(coff_list, axis=0),
        'std': np.std(coff_list, axis=0)
    }
    # os.makedirs(os.path.join(mean_std_root, ids), exist_ok=True)
    np.save(os.path.join(mean_std_root,  'mean_wild.npy'), result['mean'])
    np.save(os.path.join(mean_std_root,  'std_wild.npy'), result['std'])

def normalizeMFCC(mfcc_root, mean_std_root):
    mfcc_list = []
    # for id in tqdm(IDS):
    # files = glob(os.path.join(mfcc_root, id, '*.npy'))
    # files = glob(os.path.join(mfcc_root, '*.npy'))
    #################################################################
    files = []
    temp_files = glob(os.path.join(mfcc_root, '*.npy'))
    for a in temp_files:
        basename = os.path.splitext(os.path.basename(a))[0]+'.mp4'
        v = os.path.join('/remote-home/share/yfsong/TalkingHeadDatasetsClean/videos_256', basename)
        if os.path.exists(v):
            files.append(a)
    ###################################################################
    
    for f in tqdm(files):
        mfcc= np.load(f)
        assert mfcc.shape[1]==244, print(f, mfcc.shape)
        mfcc_list.append(mfcc)
    mfcc_list = np.concatenate(mfcc_list, axis=0)
    result = {
        'mean': np.mean(mfcc_list, axis=0),
        'std': np.std(mfcc_list, axis=0)
    }
    # os.makedirs(os.path.join(mean_std_root, ids), exist_ok=True)
    np.save(os.path.join(mean_std_root, 'mfcc_mean_wild.npy'), result['mean'])
    np.save(os.path.join(mean_std_root, 'mfcc_std_wild.npy'), result['std'])


def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(draw_keypoints)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

     
def gaussian_kernel(in_channel, out_channel, kernel_size=5, std=1, mean=0):
     # 标准高斯分布
    radius = kernel_size//2
    f = lambda x: np.exp(-((x-mean)**2)/ (2*std**2))
    weight = np.array([f(i) for i in range(-radius, radius+1)])
    weight = weight/sum(weight)
    kernel = torch.FloatTensor(weight).view(1, 1, kernel_size)
    kernel = kernel.repeat(out_channel, 1, 1)
    kernel = nn.parameter.Parameter(kernel, requires_grad=False)
    return kernel

class GaussianFilter(nn.Module):
    def __init__(self, in_channles, out_channels ,kernel_size, sigma=1):
        super().__init__()
        self.kernel_size= kernel_size
        self.sigma = sigma
        self.padding = (kernel_size-1)//2
        self.conv = nn.Conv1d(in_channles, out_channels, kernel_size, padding=self.padding, groups=in_channles)

def scale_function(x):
    x = torch.where(x>=0, torch.exp(x), 2-torch.exp(-x))
    return x
def unscale_function(x):
    x = x = torch.where(x>=1, torch.log(x), -torch.log(2-x))
    return x

if __name__ == '__main__':
    # a = torch.randn(1, 5)
    # b = a+0.01
    # print(a)
    # print(b)
    # print(scale_function(a))
    # print(scale_function(b))
    # print(unscale_function(scale_function(a)))
    # print(unscale_function(scale_function(b)))

    # audio_root = '/home/songyifei9/data/CREMA-D/wav'
    # mel_root = '/home/songyifei9/data/CREMA-D/MFCC'
    # mean_root= '/home/songyifei9/data/CREMA-D/feat_mean_std'
    # coeff_root = '/home/songyifei9/data/CREMA-D/coeff_3d/video_256'
    # audio_file_list(
    #     audio_root, train_record_path='./data/CREMA-D-traini.txt',
    #     val_record_path='./data/data_CREMA-D-val.txt',
    #     test_record_path='./data/data_CREMA-D-test.txt',
    #     aligned_root=coeff_root
    # )
    # normalizeMFCC(mel_root, mean_root)
    # coeff_root = '/home/songyifei9/data/CREMA-D/coeff_3d/video_256'
    # normalizeCoeff(coeff_root, mean_root)
    # get_mel(123, audio_root, mel_root)

    # audio_root = '/remote-home/share/yfsong/MEAD_VIDEO/wav'
    # # wav_root = '/home/songyifei9/data/MEAD/wav'
    # # wav_root= '/home/songyifei9/data/RAVDESS/wav_resample'
    # # aligned_root= '/home/songyifei9/data/MEAD/re_aligned'
    # coeff_root = '/remote-home/share/yfsong/MEAD_VIDEO/Coeff_3D'
    # # coeff_root = '/home/songyifei9/data/RAVDESS/3dmm_coeff'
    # mean_std_root = '/remote-home/share/yfsong/MEAD_VIDEO/feat_mean_std'
    # mel_root ='/remote-home/share/yfsong/MEAD_VIDEO/MFCC'
    # normalizeCoeff(coeff_root, mean_std_root)
    # normalizeMFCC(mel_root, mean_std_root)
    # audio_file_list(
    #     audio_dataroot=audio_root,
    #     aligned_root=coeff_root)
    
    # mel_root= '/home/songyifei9/data/RAVDESS/MFCC'
    # os.makedirs(wav_root, exist_ok=True)
    # # m4a_to_wav(audio_root, wav_root)
    # audio_file_list(audio_root, aligned_root=coeff_root)
    # get_mel('M028', wav_root,  mel_root)
    # for id in tqdm(IDS):
    #     get_mel(id, audio_root,  mel_root)
    # normalizeCoeff(coeff_root, mean_std_root)
    # os.makedirs(mean_std_root, exist_ok=True)
    # normalizeMFCC(mel_root, mean_std_root)
    # IDS = os.listdir(wav_root)
    # for id in tqdm(IDS):
    #     # normalizeCoeff(id, coeff_root, mean_std_root)
        # get_mel(id, wav_root,  mel_root)
    #     normalizeMFCC(id, mel_root, mean_std_root)

    # read_img_data_list(aligned_root)
    # style_dir = r'G:\MEAD-Talking_face\MEAD_Frames\M003_style\angry_level_1_001'
    # styles = read_styles(style_dir)
    # print(styles.shape)
    # masks = get_mask_from_lengths(torch.from_numpy(np.array([24, 56])).cuda())
    # print(masks.shape)
    # import lpips

    # loss = lpips.LPIPS(net='alex', model_path=r'E:\Torch_weight\hub\checkpoints\alexnet-owt-4df8aa71.pth')
#     p = r'E:\Projects\Deep3DFaceRecon_pytorch-master\checkpoints\face_recon\results\datasets\examples\epoch_20_000000'
#     batch = read_mats(p)
#     for i in batch:
#         print(i.shape)
###############################################################
    wild_wav_root = '/remote-home/share/yfsong/TalkingHeadDatasetsClean/wav'
    coeff_root = '/remote-home/share/yfsong/TalkingHeadDatasetsClean/coeff'
    mel_root = '/remote-home/share/yfsong/TalkingHeadDatasetsClean/mfcc'
    mean_std_root = '/remote-home/yfsong/code/ProTalk/mean_std'

    # get_mel(wild_wav_root, mel_root=mel_root, coeff_root=coeff_root) # 计算mfcc声学特征
    audio_file_list(wild_wav_root, aligned_root=coeff_root) #写入txt文件
    normalizeCoeff(coeff_root, mean_std_root)
    normalizeMFCC(mel_root, mean_std_root)