"""Extract raw 257-D coefficients and five alignment parameters from aligned videos."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True, help='Aligned 256x256, 30-fps video root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='assets/reconstruction/face_recon/epoch_20.pth')
    parser.add_argument('--bfm', default='assets/bfm')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from scipy.io import savemat
    from protalk.third_party.deep3d.coeff_detector import CoeffDetector
    from protalk.third_party.deep3d.extract_kp_videos import KeypointExtractor
    if not torch.cuda.is_available():
        raise RuntimeError('Coefficient extraction requires CUDA and the Deep3D dependencies')
    torch.cuda.set_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or not checkpoint.name.startswith('epoch_'):
        raise ValueError('Expected an existing reconstruction checkpoint named epoch_NUMBER.pth')
    options = SimpleNamespace(name=checkpoint.parent.name, checkpoints_dir=str(checkpoint.parent.parent),
        epoch=checkpoint.stem.removeprefix('epoch_'), model='facerecon', isTrain=False, phase='test',
        bfm_folder=str(Path(args.bfm).resolve()), bfm_model='BFM_model_front.mat', net_recon='resnet50',
        use_last_fc=False, init_path='', use_ddp=False, focal=1015., center=112., camera_d=10.,
        z_near=5., z_far=15., use_opengl=False)
    detector, landmarks = CoeffDetector(options), KeypointExtractor(device=f'cuda:{args.device}')
    root, output = Path(args.videos).resolve(), Path(args.output).resolve()
    videos = sorted(root.rglob('*.mp4'))
    if not videos:
        raise ValueError('No videos found')
    for video in videos:
        destination = output / video.relative_to(root).with_suffix('.mat')
        if destination.exists():
            print(f'Skip existing {destination}', flush=True)
            continue
        reader = cv2.VideoCapture(str(video))
        if not np.isclose(reader.get(cv2.CAP_PROP_FPS), 30., atol=.05):
            reader.release()
            raise ValueError(f'Expected 30-fps video: {video}')
        coeffs, transforms = [], []
        try:
            with torch.no_grad():
                while True:
                    ok, frame = reader.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (256, 256):
                        raise ValueError(f'Expected aligned 256x256 video: {video}')
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    points = landmarks.extract_keypoint(image)
                    if points is None:
                        raise ValueError(f'No face detected in {video}, frame {len(coeffs)}')
                    result = detector(image, points)
                    coeffs.append(result['Coeff'][0])
                    transforms.append(result['Trans'].cpu().numpy())
        finally:
            reader.release()
        if len(coeffs) < 8:
            raise ValueError(f'Video contains fewer than eight readable frames: {video}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.mat.tmp')
        savemat(str(temporary), {'coeff': np.asarray(coeffs), 'transform_params': np.asarray(transforms)}, appendmat=False)
        temporary.replace(destination)
        print(f'{video}: {len(coeffs)} frames -> {destination}', flush=True)


if __name__ == '__main__':
    main()
