import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def read_manifest(path):
    path = Path(path).resolve()
    records = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ('sample', 'audio', 'coeff', 'mfcc', 'video'):
            if item.get(key):
                value = Path(item[key]).expanduser()
                item[key] = str(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
        records.append(item)
    if not records:
        raise ValueError(f'Empty manifest: {path}')
    return records


def validate_sample(sample, label='sample'):
    expected = {'mfcc': 244, 'prosody': 2, 'coeff': 257, 'crop': 3}
    frames = len(sample['coeff'])
    if frames < 8:
        raise ValueError(f'{label}: at least 8 frames are required')
    for key, width in expected.items():
        value = sample[key]
        length = 3 * frames if key == 'prosody' else frames
        if value.shape != (length, width) or not np.isfinite(value).all():
            raise ValueError(f'{label}: invalid {key}, expected finite {(length, width)}, got {value.shape}')
    return frames


class Moments:
    """Streaming population moments; no test/validation data are included."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        mean = values.mean(0)
        m2 = ((values - mean) ** 2).sum(0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + delta ** 2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def result(self):
        std = np.sqrt(self.m2 / self.count)
        # Constant channels stay finite and contribute zero after normalization.
        std[std < 1e-6] = 1.
        return self.mean.astype(np.float32), std.astype(np.float32)


class ClipDataset(Dataset):
    def __init__(self, manifest, statistics, max_frames=256, train=True):
        self.records = read_manifest(manifest)
        self.max_frames, self.train = max_frames, train
        if not 8 <= max_frames <= 2048:
            raise ValueError('max_frames must be between 8 and 2048')
        self.stats = {key: np.load(Path(statistics) / (key + '.npy'))
                      for key in ('mean', 'std', 'mfcc_mean', 'mfcc_std')}
        for key in ('std', 'mfcc_std'):
            if not np.isfinite(self.stats[key]).all() or (self.stats[key] <= 0).any():
                raise ValueError(f'Invalid normalization statistic: {key}')
        for item in self.records:
            with np.load(item['sample'], allow_pickle=False) as sample:
                item['length'] = validate_sample(sample, item['sample'])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        with np.load(item['sample'], allow_pickle=False) as stored:
            sample = {key: stored[key].astype(np.float32) for key in ('mfcc', 'prosody', 'coeff', 'crop')}
        total = len(sample['coeff'])
        start = random.randint(0, max(total - self.max_frames, 0)) if self.train else 0
        end = min(start + self.max_frames, total)
        coeff = (sample['coeff'] - self.stats['mean']) / self.stats['std']
        mfcc = (sample['mfcc'] - self.stats['mfcc_mean']) / self.stats['mfcc_std']
        pose = np.concatenate([coeff[:, 224:227], coeff[:, 254:257], sample['crop']], axis=1)
        # Both branches predict displacement from the clip's first reference frame.
        return {'mfcc': torch.from_numpy(mfcc[start:end]),
                'prosody': torch.from_numpy(sample['prosody'][3 * start:3 * end]),
                'expression': torch.from_numpy((coeff[:, 80:144] - coeff[0, 80:144])[start:end]),
                'pose': torch.from_numpy((pose - pose[0])[start:end]),
                'coeff': torch.from_numpy(sample['coeff'][start:end]),
                'base_coeff': torch.from_numpy(sample['coeff'][0]),
                'crop': torch.from_numpy(sample['crop'][start:end]),
                'start': start, 'video': item.get('video'), 'path': item['sample']}


class PoseWindows(Dataset):
    def __init__(self, clips):
        self.clips = clips
        self.index = []
        for i, item in enumerate(clips.records):
            # Include the last full window (the historical range excluded it).
            self.index.extend((i, t) for t in range(0, item['length'] - 8 + 1, 4))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        clip, start = self.index[index]
        item = self.clips.records[clip]
        with np.load(item['sample'], allow_pickle=False) as sample:
            coeff = (sample['coeff'] - self.clips.stats['mean']) / self.clips.stats['std']
            pose = np.concatenate([coeff[:, 224:227], coeff[:, 254:257], sample['crop']], axis=1)
        return torch.from_numpy((pose[start:start + 8] - pose[0]).astype(np.float32))


def collate_clips(samples):
    batch = {key: pad_sequence([s[key] for s in samples], batch_first=True)
             for key in ('mfcc', 'prosody', 'expression', 'pose', 'coeff', 'crop')}
    batch['base_coeff'] = torch.stack([s['base_coeff'] for s in samples])
    batch['lengths'] = [len(s['mfcc']) for s in samples]
    batch['samples'] = samples
    return batch
