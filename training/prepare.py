"""Cache raw MFCC/prosody/3DMM arrays and fit training-only statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from hparams import create_hparams
from training.data import read_manifest, validate_sample, Moments


def audio_features(path, frames, hp, precomputed=None):
    import librosa
    import torch
    import torchaudio
    from yin import compute_yin
    # Reuse the historical Energy implementation.
    from proso_features import Energy
    audio, _ = librosa.load(path, sr=hp.sampling_rate, mono=True)
    per_frame = hp.sampling_rate // 30
    audio = np.pad(audio[:frames * per_frame], (0, max(0, frames * per_frame - len(audio))))
    if precomputed:
        mfcc = np.load(precomputed).astype(np.float32)
    else:
        features = []
        for i in range(frames):
            chunk = audio[i * per_frame:(i + 1) * per_frame]
            value = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(chunk).view(1, -1),
                sample_frequency=hp.sampling_rate, use_energy=True, num_ceps=80, num_mel_bins=80).T
            delta = torchaudio.functional.compute_deltas(value)
            delta2 = torchaudio.functional.compute_deltas(delta)
            features.append(np.concatenate([np.stack([value.numpy(), delta.numpy(), delta2.numpy()]).reshape(-1),
                librosa.feature.rms(y=chunk, frame_length=hp.sampling_rate).reshape(-1),
                librosa.feature.zero_crossing_rate(y=chunk, frame_length=hp.sampling_rate).reshape(-1)]))
        mfcc = np.asarray(features, dtype=np.float32)
    pitch, _, _, _ = compute_yin(audio, hp.sampling_rate, hp.filter_length, hp.hop_length,
                                hp.f0_min, hp.f0_max, hp.harm_thresh)
    pad = int(hp.filter_length / hp.hop_length / 2)
    pitch = np.nan_to_num(np.asarray([0.] * pad + pitch + [0.] * pad, dtype=np.float32))
    energy = Energy(hp.filter_length, hp.hop_length, hp.win_length).get_energy(audio)
    normalized = []
    for value in (pitch, energy):
        span = value.max() - value.min()
        value = (value - value.min()) / span if span else value * 0
        value = np.pad(value[:3 * frames], (0, max(0, 3 * frames - len(value))))
        normalized.append(value)
    return mfcc, np.stack(normalized, axis=1).astype(np.float32)


def prepare(train, val, out, hparams):
    out = Path(out).resolve()
    # Avoid silently replacing a previous training cache/statistics set.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output must be empty; choose a new cache directory: {out}')
    training, validation = read_manifest(train), read_manifest(val)
    identifiers = lambda records: {r.get('sample', r.get('audio')) for r in records}
    if identifiers(training) & identifiers(validation):
        raise ValueError('Training and validation manifests overlap')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir()
    hp = create_hparams(hparams)
    moments = {'coeff': Moments(), 'mfcc': Moments()}
    for split, records in [('train', training), ('val', validation)]:
        converted = []
        for item in records:
            if item.get('sample'):
                with np.load(item['sample'], allow_pickle=False) as data:
                    sample = {k: data[k].astype(np.float32) for k in ('mfcc', 'prosody', 'coeff', 'crop')}
            else:
                data = loadmat(item['coeff'])
                coeff = np.asarray(data['coeff'], dtype=np.float32)
                transforms = np.asarray(data['transform_params'], dtype=np.float32)
                if transforms.shape != (len(coeff), 5) or (transforms[:, :2] <= 0).any():
                    raise ValueError(f"Invalid transform_params in {item['coeff']}")
                crop = transforms[:, 2:].copy()
                crop[:, 1] /= transforms[:, 0]
                crop[:, 2] /= transforms[:, 1]
                mfcc, prosody = audio_features(item['audio'], len(coeff), hp, item.get('mfcc'))
                sample = dict(mfcc=mfcc, prosody=prosody, coeff=coeff, crop=crop)
            frames = validate_sample(sample, item.get('sample', item.get('audio')))
            identity = item.get('sample', item.get('audio'))
            name = hashlib.sha256(identity.encode()).hexdigest()[:24] + '.npz'
            np.savez_compressed(out / 'samples' / name, **sample)
            record = {'sample': 'samples/' + name, 'frames': frames}
            if item.get('video'):
                record['video'] = item['video']
            converted.append(record)
            if split == 'train':
                for key in moments:
                    moments[key].update(sample[key])
            print(f'{split}: {identity} ({frames} frames)', flush=True)
        (out / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in converted))
    stats = out / 'mean_std'
    stats.mkdir()
    for key, accumulator in moments.items():
        mean, std = accumulator.result()
        prefix = '' if key == 'coeff' else 'mfcc_'
        for suffix in ('', '_wild'):
            np.save(stats / (prefix + 'mean' + suffix + '.npy'), mean)
            np.save(stats / (prefix + 'std' + suffix + '.npy'), std)
    (stats / 'provenance.json').write_text(json.dumps({'train_manifest': str(Path(train).resolve()),
        'train_sha256': hashlib.sha256(Path(train).read_bytes()).hexdigest(), 'clips': len(training),
        'coefficient_frames': moments['coeff'].count, 'fps': 30, 'sampling_rate': hp.sampling_rate}, indent=2))
    print(f'Cache and training-only statistics saved to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--hparams', default='hparams.yaml')
    args = parser.parse_args()
    prepare(args.train, args.val, args.out, args.hparams)


if __name__ == '__main__':
    main()
