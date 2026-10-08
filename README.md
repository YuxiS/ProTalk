# ProTalk

Research code for one-shot speech-driven talking-head generation with speech prosody.
The repository contains expression prediction, VQ-VAE pose modeling, pose sampling,
rendering, and historical evaluation/ablation scripts. Video results will accompany
the manuscript as supplementary material.

## Status

The source has been organized and checked with CPU regression tests. End-to-end
GPU training and video generation have **not** been verified during this cleanup:
model checkpoints, normalization statistics, face-model assets and datasets are
not included. This checkout alone does not reproduce the paper's tables. See
[reproducibility notes](docs/REPRODUCIBILITY.md) for implementation details and
remaining issues, and [asset requirements](docs/ASSETS.md) before running inference.

## Layout

| Path | Purpose |
| --- | --- |
| `reference.py` | Single-image generation entry, original reference preset |
| `inference.py` | Later inference preset with different checkpoints and motion processing |
| `stylemodel.py`, `modules.py` | Expression model and global style tokens |
| `vqvae/` | Pose VQ-VAE and sampling modules |
| `train_style.py`, `train_script/` | Training and historical ablation entries |
| `deep3d/`, `face_utils/` | Face reconstruction and PIRender integration |
| `metrics/` | Evaluation utilities, shared BAS/SBAS formulas |
| `compare/`, `abliationStudy/`, `Visual/`, `facial_exp/` | Historical comparisons, ablations and visual losses |
| `scripts/`, `tests/`, `docs/` | Optional postprocessing, regression checks and documentation |

The spelling `abliationStudy` is retained for compatibility. Historical scripts in
these folders still contain experiment-specific paths; they are not portable CLIs.

## Environment

Use a Linux CUDA environment for the complete pipeline. `requirements.txt` gives
a **candidate Python 3.9 environment**, retaining librosa 0.9.2 and NumPy 1.23.5 for
the historical audio feature APIs. Its complete GPU compatibility is unverified.
Install matching CUDA builds of PyTorch, torchvision and torchaudio, then the
requirements, and separately install NVIDIA nvdiffrast for Deep3D rendering.
Wav2Lip/GFPGAN and extra evaluation models need their own dependencies and assets.
Activate your environment before using the shell scripts; they do not activate a
personal conda environment or overwrite `TORCH_HOME`.

## Generate a video

Run from the repository root. Set `gst_weight` and `mean_std_root` in
`hparams.yaml`, and place the assets listed in `docs/ASSETS.md` locally.

```bash
bash generate.sh /path/to/portrait.jpg /path/to/speech.wav \
  --hparams hparams.yaml \
  --model_weight /path/to/expression.pth \
  --vae_weight /path/to/vqvae.pth \
  --sampling_weight /path/to/pose_sampler.pth \
  --pirender_weight /path/to/pirender.pt \
  --save_dir results/example --device 0
```

This writes the raw **silent** 30-fps video `results/example/temp.mp4`. A short
clip must provide at least five complete frames. `inference.py` accepts the same
arguments but preserves its own later preset; these entries are not numerically
interchangeable. Checkpoint CLI options now actually control the loaded files.

For optional Wav2Lip and GFPGAN processing:

```bash
export WAV2LIP_ROOT=/path/to/Wav2Lip
export WAV2LIP_CHECKPOINT=/path/to/wav2lip.pth
export WAV2LIP_PYTHON=/path/to/wav2lip/environment/bin/python
export GFPGAN_PYTHON=/path/to/gfpgan/environment/bin/python
bash scripts/postprocess.sh results/example/temp.mp4 /path/to/speech.wav results/example/postprocessed
```

FFmpeg is required. The script creates `result.mp4` with audio after lip
synchronization and restoration. External tools and weights are not bundled;
this optional pipeline has not been run in the cleanup environment.

## Training entries

Configure the dataset roots, list files, pretrained models and statistics before
training. List files are `audio_path|text` per line; the loader derives subject
and coefficient/video filenames from audio paths. Preserve the original data
split and training statistics when reproducing a result.

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash train_style.sh
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash train_vqvae.sh
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash train_pose_sampler.sh --vae_weight /path/to/vqvae.pth
```

Wrappers use `torch.distributed.run`, accept extra CLI options, and default to
one GPU and offline W&B logging. For multiple GPUs, set `CUDA_VISIBLE_DEVICES`
and `NPROC_PER_NODE` explicitly. `HPARAMS`, `PYTHON` and `MASTER_PORT` can be set.
`train_style.sh` enables the existing ProsoResNet branch through `--prosody`;
direct `train_style.py` keeps its historical no-prosody default. `train_rnn.sh`
selects the MFCC ablation; `train_head_rnn.sh` is a direct head-RNN baseline.
These are source-level entry points, not validated reproduction recipes. See
known training limitations in `docs/REPRODUCIBILITY.md`.

## Verification

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests cover constant-signal normalization, coefficient alignment, CPU quantizer
and sampler operations, the original Gaussian smoothing, legacy BAS/SBAS math,
Python syntax and shell syntax. They do not establish end-to-end GPU correctness
or reproduce quantitative experiments.

## Third-party components

Existing copyright notices are retained. See [third-party notes](docs/THIRD_PARTY.md).
No repository-wide license is assigned by this cleanup.
