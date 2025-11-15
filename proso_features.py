import librosa
import numpy as np
# import pyworld
from scipy.interpolate import interp1d
from audio_wav2lip import load_wav

class Energy():
    def __init__(self,
                 n_fft: int=2048,
                 hop_length: int=300,
                 win_length: int=None,
                 window: str="hann",
                 center: bool=True,
                 pad_mode: str="reflect"):

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def _stft(self, wav: np.ndarray):
        D = librosa.core.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode)
        return D

    def _calculate_energy(self, input: np.ndarray):
        input = input.astype(np.float32)
        input_stft = self._stft(input)
        input_power = np.abs(input_stft)**2
        energy = np.sqrt(
            np.clip(
                np.sum(input_power, axis=0), a_min=1.0e-10, a_max=float('inf')))
        return energy

    def _average_by_duration(self, input: np.ndarray,
                             d: np.ndarray) -> np.ndarray:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        # shape (T,1)
        arr_list = np.expand_dims(np.array(arr_list), 0).T
        return arr_list

    def get_energy(self,
                   wav: np.ndarray,
                   use_token_averaged_energy: bool=True,
                   duration: np.ndarray=None):
        energy = self._calculate_energy(wav)
        if use_token_averaged_energy and duration is not None:
            energy = self._average_by_duration(energy, duration)
        return energy





if __name__=='__main__':
    wav = load_wav('/home/songyifei9/data/MEAD/wav/M027/contempt_level_1_027.wav', 22050)
    engry = Energy(n_fft=1024, hop_length=256, win_length=1024)
    e = engry.get_energy(wav=wav)
    print(e.shape)
    print(e)