"""Report dependencies and required local files before a long run."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['training', 'inference', 'visual'], default='training')
    parser.add_argument('--config', default='configs/train.yaml')
    args = parser.parse_args()
    packages = ['torch', 'numpy', 'scipy', 'yaml', 'librosa', 'torchvision', 'torchaudio']
    if args.profile in ('inference', 'visual'):
        packages += ['cv2', 'kornia', 'einops']
    if args.profile == 'inference':
        packages += ['face_alignment', 'av', 'nvdiffrast']
    missing = []
    for module in packages:
        found = importlib.util.find_spec(module) is not None
        print(f'{"OK" if found else "MISSING"} package: {module}')
        if not found: missing.append(module)
    if missing:
        raise SystemExit(1)
    from training.run import resolve_config
    config = resolve_config(args.config)
    files = [config['hparams']]
    if args.profile != 'inference':
        files += [config['train_manifest'], config['val_manifest']]
        if config['expression']['gst_init'] == 'pretrained':
            from hparams import create_hparams
            gst = Path(create_hparams(config['hparams']).gst_weight)
            files.append(gst if gst.is_absolute() else Path(__file__).resolve().parents[1] / gst)
    stats = Path(config['statistics'])
    files += [stats / (key + '.npy') for key in ('mean', 'std', 'mfcc_mean', 'mfcc_std')]
    if args.profile == 'visual':
        files += [config['expression']['pirender_weight'], config['expression']['emotion_weight'],
                  Path(config['expression']['bfm_folder']) / 'BFM_model_front.mat']
    if args.profile == 'inference':
        files += [Path(config['output']) / stage / 'best-inference.pth' for stage in ('expression', 'vqvae', 'sampler')]
        files += [config['expression']['pirender_weight'], Path(config['expression']['bfm_folder']) / 'BFM_model_front.mat',
            Path(config['expression']['bfm_folder']) / 'similarity_Lm3D_all.mat',
            'deep3d/checkpoints/face_recon/epoch_20.pth']
    for path in files:
        found = Path(path).is_file()
        print(f'{"OK" if found else "MISSING"} file: {path}')
        if not found: missing.append(str(path))
    if missing:
        raise SystemExit(1)
    print('Required packages/files found. This check does not establish model quality or CUDA compatibility.')


if __name__ == '__main__':
    main()
