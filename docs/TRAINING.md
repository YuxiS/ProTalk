# Training the existing ProTalk networks again

The original learned weights, normalization arrays and exact experiment protocol
are not available. This workflow makes the existing architectures trainable with
explicit data paths, validation, stage-specific exports and resume. Hyperparameters
in `configs/train.yaml` are starting values, not a claim of paper reproduction.
Keep the historical results separate from newly trained checkpoints.

## Environment and a quick functional check

Python 3.9, torch/torchvision/torchaudio 2.2.2/0.17.2/2.2.2 and NumPy 1.23.5 are
used by CPU checks. Install `requirements-training.txt`. Run
`python -m training.smoke` without downloading weights; add `--output /new/folder`
to retain synthetic fixtures. These fixtures are for program checks only.
For reconstruction and generation install `requirements.txt` and build nvdiffrast
with your CUDA toolchain. FFmpeg extracts audio and muxes the result.

## Data contracts

Each raw manifest is UTF-8 JSONL, one synchronized clip per line:

```json
{"audio":"/dataset/train/audio/person/clip.wav","coeff":"/dataset/train/coefficients/person/clip.mat","video":"/dataset/train/videos/person/clip.mp4"}
```

Relative paths resolve against the manifest directory. `video` is optional for
coefficient-only training and required for visual fine-tuning. An optional `mfcc`
path supplies a precomputed raw NumPy `(T,244)` array. MAT files must have:

| Field | Shape and interpretation |
| --- | --- |
| `coeff` | `(T,257)`: identity 80, expression 64, texture 80, angle 3, lighting 27, translation 3 |
| `transform_params` | `(T,5)`: original width, height, alignment scale, x translation, y translation |

`training.extract` saves this format from aligned 256×256, 30-fps MP4 videos.
Use `--checkpoint`, `--bfm` and `--device` for nondefault reconstruction assets.
It skips existing MAT files; remove a particular cached file deliberately to redo
it. It stops on missing faces or invalid resolution/frame rate. It does not align
arbitrary input scenes; start from a consistently aligned dataset.

`training.manifest` matches relative MP4 filenames with MAT and WAV filenames.
`--extract-audio` extracts missing mono 22050-Hz WAVs using FFmpeg and leaves
existing audio files intact. Choose a new manifest output path on reruns.

You may also provide an already prepared raw NPZ:

```json
{"sample":"/dataset/raw/clip.npz","video":"/dataset/aligned/clip.mp4"}
```

The NPZ fields are `mfcc(T,244)`, `prosody(3T,2)`, `coeff(T,257)` and `crop(T,3)`;
all values must be finite float arrays and `T >= 8`. `crop` is alignment scale,
x/width, y/height. MFCC/coefficient arrays must be **unnormalized**.
Prosody columns are pitch and energy, each min/max normalized per clip.
Do not mix prepared normalized arrays with this raw-data interface.

Create separate training/validation manifests with independent subjects and
sessions as appropriate for your evaluation. The code rejects overlapping file
paths; it cannot detect the same speaker saved under different filenames.

## Preparation

```bash
python -m training.prepare --train data/train-raw.jsonl --val data/val-raw.jsonl \
  --out data/prepared --hparams hparams.yaml
```

This creates `samples/*.npz`, `train.jsonl`, `val.jsonl`, `mean_std/*.npy` and
`mean_std/provenance.json`. It refuses nonempty output directories to avoid mixing
caches. After a failed preparation choose a new empty output or inspect and clear
the incomplete cache yourself.

Audio uses the historical 22050-Hz feature convention: each 30-fps frame consumes
734 samples, Kaldi 80 MFCCs plus first/second deltas and RMS/ZCR produce 244
features. YIN pitch and energy are padded/truncated to three samples per video
frame. Waveforms are trimmed/padded to coefficient duration. This convention
assumes synchronized clips; preprocessing cannot correct audio/video offsets.

Population mean/std are fitted on **training frames only**. Constant channels use
std 1. `mean/std.npy` hold 257 coefficient statistics; `mfcc_mean/std.npy` hold
244 feature statistics. `_wild` aliases contain the same training statistics for
legacy loaders; they are not a separate wild-data estimate. Preserve these arrays
with the checkpoint. Changing the dataset requires recomputing its statistics and
retraining compatible stages.

## Stage objectives and defaults

All supported stages use the original network classes. Expression and pose targets
are displacement from the **first frame of the complete clip**, even when random
subclips are sampled. Expression coefficients and the six angle/translation values
use training mean/std; crop displacement is in the alignment units above.
Expression inference adds the reference portrait's coefficients back.

| Stage | Objective / behavior | Default schedule |
| --- | --- | --- |
| expression | Masked L2 norm of 64 coefficients ×0.5 plus adjacent-frame delta L2 ×1 | Adam, lr 0.001, 100 epochs, batch 8; StepLR 20 epochs ×0.7 |
| vqvae | MSE on eight-frame, nine-value pose windows plus EMA commitment loss ×5 | AdamW, lr 0.0001, 500 epochs, batch 256; StepLR 100 epochs ×0.5 |
| sampler | Cross-entropy for frozen VQ-VAE codes at stride four; report code accuracy | AdamW, lr 0.0004, 500 epochs, batch 8; StepLR 30 epochs ×0.8 |

New VQ-VAE training initializes EMA sums/counts consistently and synchronizes EMA
updates across distributed ranks. Windows have length eight and stride four.
Sampler targets pad the final partial window with zeros and mask invalid time
positions. The old decoded-argmax pose penalty does not provide a gradient through
code selection and is not included in the maintained sampler objective.

Default `expression.gst_init: scratch` trains the existing GST and projection
jointly. `pretrained` loads the Mellotron `gst_weight` in hparams, then freezes
those parts. A saved expression export contains its GST, so inference needs no
separate Mellotron checkpoint. Scratch GST and coefficient-only supervision are
practical retraining settings; they differ from the historical training recipe.
The existing InstanceNorm layout is retained and may emit a shape warning with
current torch; network axes were not silently changed during cleanup.

## Launch, validation and resume

All config paths resolve against the repository root, not the config directory.
Manifest record paths resolve against their manifest. Copy `configs/train.yaml`,
edit paths and batch sizes, then check it. New runs refuse an output stage folder
that already contains checkpoints; choose a fresh output or use `--resume`:

```bash
python -m training.doctor --profile training --config configs/my-data.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all.sh configs/my-data.yaml
```

For staged/distributed training:

```bash
TRAIN_CONFIG=configs/my-data.yaml CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash train_style.sh
TRAIN_CONFIG=configs/my-data.yaml CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train_vqvae.sh
TRAIN_CONFIG=configs/my-data.yaml CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash train_pose_sampler.sh
```

The batch size is **per rank**, must be at least two, and training drops incomplete
batches. Reduce it if the dataset has too few clips/windows. `max_frames` controls
expression/sampler clip length (8–2048); VQ-VAE uses all full windows.
`workers: 0` helps when debugging data loading. `PYTHON`, `MASTER_ADDR` and
`MASTER_PORT` can override wrapper defaults. `train_all.sh` runs one process per
stage; use the stage wrappers for torchrun. Only single-node wrappers are supplied.

Validation runs each epoch with fixed seeds and padding masks; it never updates
the codebook or optimization state. Validation summaries append to `metrics.jsonl`.
The selected best loss is a training diagnostic, not a manuscript metric.

```bash
python -m training.run --stage expression --config configs/my-data.yaml \
  --resume weights/retrained/expression/last-training.pth
```

Full checkpoints store model/optimizer/scheduler/RNG states, epoch, configuration,
architecture and normalization hash. Resume requires the same world size, data
paths and stage settings; you may increase `epochs`. Keep source data and manifest
contents unchanged as well. `--initialize` warm-starts an expression/sampler
inference export with a fresh optimizer for a new run. `--max-steps` is only for
functional smoke checks; its partial-epoch checkpoints are intentionally rejected
by resume. GPU algorithms can remain nondeterministic despite recorded seeds.

## Optional visual fine-tuning

The base workflow requires no emotion checkpoint or renderer during training.
For image/landmark/emotion supervision, set `expression.loss_profile: visual` and
supply `pirender_weight` (`net_G_ema`), `bfm_folder` (`BFM_model_front.mat`) and a
compatible `emotion_weight` (`state_dict`) used by `Visual/expression_loss.py`.
The original emotion checkpoint is not bundled; this option requires you to
obtain a compatible asset and check its terms. Every manifest record must retain
its aligned video.

Use a new output directory and warm-start your coefficient-trained model:

```bash
python -m training.doctor --profile visual --config configs/visual.yaml
python -m training.run --stage expression --config configs/visual.yaml \
  --initialize weights/retrained/expression/best-inference.pth
```

PIRender, BFM geometry and the emotion network are frozen. Gradients pass through
rendering/projection to predicted expression coefficients. The reference is the
video's first frame, with ground-truth pose/static coefficients. Selected image
frames and all adjacent lip-landmark deltas contribute visual losses. Defaults
are landmark 0.0001, lip delta 0.5, emotion 100 and image SmoothL1 0; adjust them
explicitly and record your settings. Renderer input gradients were tested with
random weights on CPU; real assets, visual quality and full GPU optimization have
not been validated.

## Exports, inference and sharing

Each stage exports `best-inference.pth` and `last-inference.pth`:
expression contains `audio_model` plus metadata, VQ-VAE contains
`Encoder/Decoder/CodeBook` plus metadata, and sampler is a bare state dict for
legacy PoseGEN. Exports exclude optimizer state. New expression metadata sets
`pose_scale: 1` for `reference.py`.

Run `generate.sh` as described in the README. For a reusable trained release,
package all three matching inference exports, the exact hparams/config and the
four normalization arrays, along with dataset/split provenance, training
environment and instructions for external face/renderer assets. Do not distribute
synthetic smoke-test weights as a trained release. See `docs/ASSETS.md` for
public source links and compatibility limitations.
