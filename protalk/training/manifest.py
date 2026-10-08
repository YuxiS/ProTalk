"""Build explicit manifests from aligned videos and extracted coefficient MAT files."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--coefficients', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extract-audio', action='store_true', help='Use ffmpeg to extract missing WAV files')
    args = parser.parse_args()
    videos, coeffs, audio = [Path(v).resolve() for v in (args.videos, args.coefficients, args.audio)]
    records = []
    for video in sorted(videos.rglob('*.mp4')):
        relative = video.relative_to(videos)
        coefficient = coeffs / relative.with_suffix('.mat')
        wav = audio / relative.with_suffix('.wav')
        if not coefficient.is_file():
            raise FileNotFoundError(coefficient)
        if not wav.is_file() and args.extract_audio:
            wav.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-nostdin', '-n', '-i', str(video), '-vn', '-ac', '1', '-ar', '22050', str(wav)], check=True)
        if not wav.is_file():
            raise FileNotFoundError(f'{wav}; use --extract-audio or supply audio')
        records.append(dict(audio=str(wav), coeff=str(coefficient), video=str(video)))
    if not records:
        raise ValueError('No MP4 videos found')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new manifest path: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(record) + '\n' for record in records))
    print(f'{len(records)} records saved to {output}')


if __name__ == '__main__':
    main()
