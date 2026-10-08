import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from protalk.config import create_hparams
from glob import glob
import os.path as osp
import os
import torchaudio
import torch
import scipy.io as scio
import pdb
hp = create_hparams()

def get_mels(audio_name, coeff_root='/home/songyifei9/data/MEAD_VIDEO/Coeff_3D'):
    video_id = osp.basename(audio_name)[:-4]
    ids = audio_name.split(os.sep)[-2]
    fps = 30
    sr = hp.sampling_rate
    audio = load_wav(audio_name, hp.sampling_rate)
    # pdb.set_trace()
    frame_n_samples = int(sr / fps)
    ######################################################
    # pdb.set_trace()
    # if os.path.exists(os.path.join(hp.coff_root, ids, '01-'+video_id[3:])):
    #     video_path =  os.path.join(hp.coff_root, ids, '01-'+video_id[3:])
    # elif os.path.exists(os.path.join(hp.coff_root, ids, '02-'+video_id[3:])):
    #     video_path = os.path.join(hp.coff_root, ids, '02-'+video_id[3:])
    # n_frames = len(glob(os.path.join(video_path, '**/*.mat')))
    # n_frames = len(glob(os.path.join('/home/songyifei9/data/MEAD_VIDEO/Coeff_3D', ids, video_id, "**/*.mat")))
    basename = osp.splitext(osp.basename(audio_name))[0]
    # pdb.set_trace()
    coeff = scio.loadmat(os.path.join(coeff_root, ids, basename+'.mat'))
    # if not os.path.exists(os.path.join(coeff_root, 'CREMA-D-VideoFlash-'+basename+'.mat')):
    #     print('{} is not exist!'.format(audio_name))
    #     return None
    # coeff = scio.loadmat(os.path.join(coeff_root, 'CREMA-D-VideoFlash-'+basename+'.mat'))
    n_frames = len(coeff['coeff'])
    assert n_frames!=0, print(audio_name)
    #########################################
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0

    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        # pdb.set_trace()
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True, num_ceps=80, num_mel_bins=80)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)

        rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)

        curr_feat = np.concatenate((curr_mfccs, rms, zcr))
        curr_feats.append(curr_feat)
    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


def get_wild_mels(audio_name, coeff_root='/home/songyifei9/data/MEAD_VIDEO/Coeff_3D'):
    # video_id = osp.basename(audio_name)[:-4]
    # ids = audio_name.split(os.sep)[-2]
    fps = 30
    sr = hp.sampling_rate
    audio = load_wav(audio_name, hp.sampling_rate)
    # pdb.set_trace()
    frame_n_samples = int(sr / fps)
    ######################################################
    # pdb.set_trace()
    # if os.path.exists(os.path.join(hp.coff_root, ids, '01-'+video_id[3:])):
    #     video_path =  os.path.join(hp.coff_root, ids, '01-'+video_id[3:])
    # elif os.path.exists(os.path.join(hp.coff_root, ids, '02-'+video_id[3:])):
    #     video_path = os.path.join(hp.coff_root, ids, '02-'+video_id[3:])
    # n_frames = len(glob(os.path.join(video_path, '**/*.mat')))
    # n_frames = len(glob(os.path.join('/home/songyifei9/data/MEAD_VIDEO/Coeff_3D', ids, video_id, "**/*.mat")))
    basename = osp.splitext(osp.basename(audio_name))[0]
    # pdb.set_trace()
    coeff = scio.loadmat(os.path.join(coeff_root, basename+'.mat'))
    # if not os.path.exists(os.path.join(coeff_root, 'CREMA-D-VideoFlash-'+basename+'.mat')):
    #     print('{} is not exist!'.format(audio_name))
    #     return None
    # coeff = scio.loadmat(os.path.join(coeff_root, 'CREMA-D-VideoFlash-'+basename+'.mat'))
    n_frames = len(coeff['coeff'])
    assert n_frames!=0, print(audio_name)
    #########################################
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0

    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        # pdb.set_trace()
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True, num_ceps=80, num_mel_bins=80)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)

        rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)

        curr_feat = np.concatenate((curr_mfccs, rms, zcr))
        curr_feats.append(curr_feat)
    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_length
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sampling_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S



def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_length, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_length)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sampling_rate // 2
    return librosa.filters.mel(hp.sampling_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)




if __name__=='__main__':
    mel = get_mels('/home/songyifei9/data/MEAD/wav/M007/angry_level_1_005.wav')
    # mel = melspectrogram(wav)
    print(mel.shape)