"""Exercise real ProTalk models on synthetic data; these are not release weights."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from training.prepare import prepare
from training.run import resolve_config, run


def fixture(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Synthetic fixture output must be empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    for split, count in [('train', 4), ('val', 2)]:
        records = []
        for i in range(count):
            frames = 16 if i % 2 == 0 else 12
            name = split + str(i) + '.npz'
            np.savez(directory / name, mfcc=rng.normal(size=(frames, 244)).astype('float32'),
                prosody=rng.uniform(size=(3 * frames, 2)).astype('float32'),
                coeff=rng.normal(size=(frames, 257)).astype('float32'),
                crop=rng.normal(size=(frames, 3)).astype('float32'))
            records.append({'sample': name})
        (directory / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
    root = Path(__file__).resolve().parents[1]
    hp = yaml.safe_load((root / 'hparams.yaml').read_text())
    hp['PoseModel'].update(n_embeddings=8, embedding_dim=16, n_hiddens=16)
    (directory / 'hparams.yaml').write_text(yaml.safe_dump(hp))
    prepare(directory / 'train.jsonl', directory / 'val.jsonl', directory / 'prepared', directory / 'hparams.yaml')
    config = yaml.safe_load((root / 'configs/train.yaml').read_text())
    config.update(hparams=str(directory / 'hparams.yaml'), train_manifest=str(directory / 'prepared/train.jsonl'),
        val_manifest=str(directory / 'prepared/val.jsonl'), statistics=str(directory / 'prepared/mean_std'),
        output=str(directory / 'weights'), workers=0, max_frames=32)
    for stage in ('expression', 'vqvae', 'sampler'):
        config[stage].update(epochs=1, batch_size=2)
    config['sampler']['vqvae_weight'] = str(directory / 'weights/vqvae/best-inference.pth')
    path = directory / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path


def smoke(directory):
    torch.set_num_threads(1)
    path = fixture(directory)
    config = resolve_config(path)
    outputs = {}
    for stage in ('expression', 'vqvae', 'sampler'):
        outputs[stage] = run(stage, config, 'cpu')
        assert (outputs[stage] / 'best-inference.pth').is_file()
    original = torch.load(outputs['sampler'] / 'last-training.pth', map_location='cpu')
    config['sampler']['epochs'] = 2
    run('sampler', config, 'cpu', resume=outputs['sampler'] / 'last-training.pth')
    resumed = torch.load(outputs['sampler'] / 'last-training.pth', map_location='cpu')
    assert resumed['epoch'] == 1 and resumed['step'] > original['step']
    from vqvae.generate import VAE
    from stylemodel import ProsoResNet
    from hparams import create_hparams
    hp = create_hparams(config['hparams'])
    expression = ProsoResNet(hp, 244, 2, 64, load_gst=False).eval()
    expression.load_state_dict(torch.load(outputs['expression'] / 'best-inference.pth')['audio_model'])
    pose = VAE(2, 8, 16, 16, 9, .25, str(outputs['vqvae'] / 'best-inference.pth'),
               str(outputs['sampler'] / 'best-inference.pth')).eval()
    with torch.no_grad():
        expr = expression(torch.randn(2, 16, 244), torch.randn(2, 48, 2), [16, 12])
        head = pose(torch.randn(2, 48, 2), [16, 12], step=4)
    assert expr.shape == (2, 16, 64) and head.shape[0] == 2 and head.shape[-1] == 9
    assert torch.isfinite(expr).all() and torch.isfinite(head).all()
    print('PASS: real-model CPU training, validation, resume and inference-export loading. Synthetic data only.')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Keep the synthetic fixture in a new directory; otherwise use a temporary directory')
    args = parser.parse_args()
    if args.output:
        smoke(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='protalk-smoke-') as directory:
            smoke(directory)


if __name__ == '__main__':
    main()
