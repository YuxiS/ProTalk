"""Download selected public auxiliary assets; ProTalk-trained weights are not bundled."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'reconstruction': ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/epoch_20.pth', 'deep3d/checkpoints/face_recon/epoch_20.pth'),
    'landmarks': ('https://raw.githubusercontent.com/OpenTalker/SadTalker/main/src/config/similarity_Lm3D_all.mat', 'deep3d/BFM/similarity_Lm3D_all.mat'),
    'bfm-fitting': ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip', 'weights/BFM_Fitting.zip'),
    'wav2lip': ('https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/wav2lip.pth', 'weights/wav2lip.pth'),
    'gfpgan': ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 'gfpgan/weights/GFPGANv1.4.pth'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', nargs='+', choices=ASSETS)
    args = parser.parse_args()
    for name in args.assets:
        url, relative = ASSETS[name]
        destination = ROOT / relative
        if destination.exists():
            print(f'Skip existing {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.part')
        checksum = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'ProTalk-assets'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    raise ValueError('Download returned an HTML page instead of the asset')
                expected = int(response.headers.get('Content-Length', 0))
                size = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block); checksum.update(block); size += len(block)
                    print(f'\r{name}: {size // 1048576} MiB', end='', flush=True)
                if not size or (expected and expected != size):
                    raise ValueError('Incomplete asset download')
            temporary.replace(destination)
            destination.with_suffix(destination.suffix + '.source.json').write_text(json.dumps(
                {'source': url, 'sha256': checksum.hexdigest(), 'bytes': size}, indent=2))
            print(f'\nSaved {destination}')
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
