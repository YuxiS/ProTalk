import torch
import numpy as np

def parse_batch(batch):
    mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, _, audio_length, _ = batch
    # input_length = input_length.cuda(non_blocking=False)
    mel = mel.cuda()
    f0 = f0.cuda()
    energy = energy.cuda()
    coeff_dynamic = coeff_dynamic.cuda()
    coeff_static = coeff_static.cuda()
    coeff_crop = coeff_crop.cuda()
    return mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, audio_length    

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = parse_batch(self.batch)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch