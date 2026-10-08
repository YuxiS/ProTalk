# ProTalk

One-shot speech-driven talking-head generation with speech prosody. This repository
contains the expression model, pose VQ-VAE, prosody-conditioned pose sampler, and
integration with Deep3D and PIRender.

**Original ProTalk training checkpoints and statistics are unavailable.** The
supported workflow below lets you prepare your own data and train the existing
networks again. It is a new training recipe, not a verified reproduction of the
paper's numbers. CPU tests exercise real networks, checkpoint resume/export, audio
features and renderer gradients. Full training and video generation with external
assets have not been verified on a CUDA machine. Paper video results are supplied
separately as supplementary material.

## 1. Install and check the code

Run commands from the repository root in a Python 3.9 environment. Use Linux and
CUDA for coefficient extraction, full training and video generation. Install
matching builds of PyTorch 2.2.2, torchvision 0.17.2 and torchaudio 2.2.2 for your
CUDA runtime, then:

```bash
python -m pip install -r requirements-training.txt
python -m training.smoke
python -m unittest discover -s tests -v
```

The smoke check creates synthetic data in a temporary folder, trains all three
stages briefly, resumes training and reloads the inference exports. It does not
need a dataset or downloaded checkpoints and does not produce useful release
weights. For extraction/inference also install `requirements.txt`, FFmpeg and
[nvdiffrast](https://github.com/NVlabs/nvdiffrast). See
[required assets and sources](docs/ASSETS.md) for Deep3D, BFM and PIRender.

## 2. Prepare data

Use synchronized, face-aligned **256×256, 30-fps MP4** clips with speech audio.
Split training and validation subjects before preprocessing. This repository
expects aligned videos; resizing a full scene is not face alignment. Preserve
relative filenames across video, coefficient and audio trees.

```bash
python -m training.extract --videos /dataset/train/videos --output /dataset/train/coefficients
python -m training.manifest --videos /dataset/train/videos \
  --coefficients /dataset/train/coefficients --audio /dataset/train/audio \
  --extract-audio --output data/train-raw.jsonl
python -m training.extract --videos /dataset/val/videos --output /dataset/val/coefficients
python -m training.manifest --videos /dataset/val/videos \
  --coefficients /dataset/val/coefficients --audio /dataset/val/audio \
  --extract-audio --output data/val-raw.jsonl
python -m training.prepare --train data/train-raw.jsonl --val data/val-raw.jsonl \
  --out data/prepared --hparams hparams.yaml
```

Extraction requires a reconstruction checkpoint, face-alignment and BFM assets.
If you already have compatible coefficient MAT files, skip extraction.
Preparation caches features and computes normalization **only on the training
split**. It refuses to overwrite a nonempty cache. Explicit JSONL/NPZ formats and
preprocessing assumptions are documented in [training instructions](docs/TRAINING.md).

## 3. Train

Copy/edit `configs/train.yaml` for your paths, batch sizes and GPU memory. Default
GST initialization is from scratch, so Mellotron is optional. The basic expression
stage uses coefficient supervision; optional visual fine-tuning is described in
[training instructions](docs/TRAINING.md).

```bash
python -m training.doctor --profile training --config configs/train.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all.sh configs/train.yaml
```

Or run each stage separately, in this order:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_style.sh
CUDA_VISIBLE_DEVICES=0 bash train_vqvae.sh
CUDA_VISIBLE_DEVICES=0 bash train_pose_sampler.sh
```

Set `TRAIN_CONFIG` to your config and `PYTHON` to your interpreter if needed.
Wrappers accept `--resume /path/to/last-training.pth`; set `NPROC_PER_NODE` and
`CUDA_VISIBLE_DEVICES` for multiple GPUs. Each stage writes validation logs,
`best/last-training.pth` (optimizer/scheduler/RNG for resume) and
`best/last-inference.pth` (model export) under `weights/retrained/<stage>/`.
After changing the output directory, also update `sampler.vqvae_weight` in the
config. The sampler must use the VQ-VAE trained with the same normalization.

## 4. Generate a video

Download the external assets described in [ASSETS.md](docs/ASSETS.md).
`generate.sh` defaults to the three new exports and `data/prepared/mean_std`.
It uses `reference.py`, whose reference-relative coefficients match new training.

```bash
python -m training.doctor --profile inference --config configs/train.yaml
bash generate.sh /path/to/portrait.jpg /path/to/speech.wav \
  --pirender_weight /path/to/pirender.pt --save_dir results/example --device 0
ffmpeg -i results/example/temp.mp4 -i /path/to/speech.wav \
  -c:v copy -c:a aac -shortest results/example/result.mp4
```

Use a tightly cropped, face-aligned square portrait as the reference.
The raw output `temp.mp4` is silent, at 30 fps; input audio must cover at least five
complete frames. Override `--model_weight`, `--vae_weight`, `--sampling_weight`,
`--mfcc_mean_std_root`, `--hparams` and reconstruction options for nondefault paths.
Use the exact statistics/hparams accompanying the trained weights. `--pose_scale`
controls motion strength; new expression exports default to 1, historical exports
to 2. `inference.py` retains a different historical preset and is not the supported
entry for these new exports.

Optional Wav2Lip/GFPGAN postprocessing uses separate installations:

```bash
export WAV2LIP_ROOT=/path/to/Wav2Lip
export WAV2LIP_CHECKPOINT=/path/to/wav2lip.pth
export WAV2LIP_PYTHON=/path/to/wav2lip/environment/bin/python
export GFPGAN_PYTHON=/path/to/gfpgan/environment/bin/python
bash scripts/postprocess.sh results/example/temp.mp4 /path/to/speech.wav results/example/postprocessed
```

This optional pipeline has not been run in the cleanup environment.

## Repository guide

| Path | Purpose |
| --- | --- |
| `training/`, `configs/train.yaml` | Supported preparation, training, validation, resume and export |
| `reference.py`, `generate.sh` | Supported inference for newly trained exports |
| `stylemodel.py`, `modules.py` | Existing expression/prosody/GST architecture |
| `vqvae/` | Existing pose codebook and sampler architecture |
| `deep3d/`, `face_utils/` | Reconstruction and PIRender integration |
| `scripts/download_assets.py` | Explicit downloads of selected public third-party assets |
| `tests/`, `.github/workflows/` | CPU checks and continuous integration |
| `train_style.py`, `train_script/`, `compare/`, `abliationStudy/`, `metrics/` | Historical experiments/evaluation; may contain dataset-specific paths |

The historical Python training loops, `train_rnn.sh` and `train_head_rnn.sh` are
archival entries. Use `training.run` for the maintained three-stage workflow.
The spelling `abliationStudy` is retained. See [reproducibility notes](docs/REPRODUCIBILITY.md)
for historical limitations and [third-party notices](docs/THIRD_PARTY.md) for
component terms. No repository-wide license is assigned by this cleanup.
