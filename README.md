# ProTalk

One-shot speech-driven talking-head generation with speech prosody.

**The original ProTalk checkpoints and normalization statistics are unavailable.**
This repository provides a maintained workflow to prepare data and train the
existing networks again. The new recipe is not a verified reproduction of the
paper's numbers. Paper video results are provided separately as supplementary
material. Full CUDA training and video generation still require external assets
and validation on real data.

## Repository layout

```text
protalk/                   Maintained Python package
  audio/                   Audio features, prosody and STFT
  models/                  Expression/GST and pose VQ-VAE/sampler
  training/                Data preparation, training, validation and checkpoints
  inference/               Generation and Gaussian smoothing
  evaluation/              Shared beat-score formulas
  third_party/             Deep3D, PIRender/face utilities and EMOCA-derived code
configs/                   Model parameters and training settings
scripts/                   Shell launchers, downloads and optional restoration
requirements/              Training, testing and inference dependencies
assets/                    Instructions; downloaded face assets are ignored
archive/                   Local historical backup; its contents are excluded from Git
tests/                    CPU regression and workflow checks
docs/                     Training, assets, structure and reproducibility notes
```

Use `protalk/` for the supported workflow. Historical scripts in the local `archive/` retain
experiment-specific assumptions and are not uploaded. Only its explanatory README
is tracked; remote users can inspect previous versions through Git history. See
[structure and migration map](docs/STRUCTURE.md) for the old-to-new paths.

## Install and check

Use Python 3.9. Run commands from the repository root in an activated environment.
Install matching PyTorch 2.2.2, torchvision 0.17.2 and torchaudio 2.2.2 builds for
your CUDA runtime, then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/training.txt
python -m pip install -e . --no-deps
python -m protalk --help
python -m protalk smoke
python -m unittest discover -s tests -v
```

The editable installation supplies the `protalk` command as an alternative to
`python -m protalk`. Run workflow commands from the repository root so relative
paths in CLI arguments resolve consistently. Synthetic smoke data exercise all
three real networks, validation, resume and export loading; they do not yield
usable trained weights.

For coefficient extraction and generation, also install `requirements.txt`,
FFmpeg and [nvdiffrast](https://github.com/NVlabs/nvdiffrast) in a Linux CUDA
environment. Follow [asset instructions](docs/ASSETS.md) for Deep3D, BFM and PIRender.

## Prepare, train and generate

Start with synchronized, face-aligned **256×256, 30-fps MP4** clips, split by
subject/session into training and validation sets. The code expects aligned clips;
resizing a full scene does not align a face.

```bash
# Repeat extraction/manifest creation for a separate validation split.
python -m protalk extract --videos /dataset/train/videos --output /dataset/train/coefficients
python -m protalk manifest --videos /dataset/train/videos \
  --coefficients /dataset/train/coefficients --audio /dataset/train/audio \
  --extract-audio --output data/train-raw.jsonl
python -m protalk extract --videos /dataset/val/videos --output /dataset/val/coefficients
python -m protalk manifest --videos /dataset/val/videos \
  --coefficients /dataset/val/coefficients --audio /dataset/val/audio \
  --extract-audio --output data/val-raw.jsonl
python -m protalk prepare --train data/train-raw.jsonl --val data/val-raw.jsonl \
  --out data/prepared --hparams configs/model.yaml
```

Preparation fits statistics on training data only. Compatible existing MAT files
can be used without extraction. Copy/edit `configs/train.yaml` for your paths,
batch sizes and GPU memory. Default GST training starts from scratch; optional
visual fine-tuning is described in [TRAINING.md](docs/TRAINING.md).

```bash
python -m protalk doctor --profile training --config configs/train.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all.sh configs/train.yaml
```

For individual stages or multiple GPUs use `scripts/train_expression.sh`,
`scripts/train_vqvae.sh` and `scripts/train_sampler.sh`, in that order. Set
`TRAIN_CONFIG`, `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` as needed.
Resume with `--resume /path/to/last-training.pth`. Exports are written to
`weights/retrained/<stage>/best-inference.pth`; the sampler must load the matching
VQ-VAE specified in its config.

After training and acquiring external rendering/reconstruction assets:

```bash
python -m protalk doctor --profile inference --config configs/train.yaml
bash scripts/generate.sh /path/to/aligned-portrait.jpg /path/to/speech.wav \
  --pirender_weight weights/pirender.pt --save_dir results/example --device 0
ffmpeg -i results/example/temp.mp4 -i /path/to/speech.wav \
  -c:v copy -c:a aac -shortest results/example/result.mp4
```

The initial output is silent at 30 fps. Use a tightly cropped, aligned square
reference portrait and audio with at least five complete frames. Generation
options are listed by `python -m protalk generate --help`. Use the exact model
config/statistics accompanying all three matching checkpoints. Optional
Wav2Lip/GFPGAN restoration is described in [TRAINING.md](docs/TRAINING.md).

## Further documentation

- [Training and data formats](docs/TRAINING.md)
- [External assets and download sources](docs/ASSETS.md)
- [Structure and migration map](docs/STRUCTURE.md)
- [Reproducibility and known historical limitations](docs/REPRODUCIBILITY.md)
- [Third-party notices](docs/THIRD_PARTY.md)

Existing component notices remain in place. No repository-wide license is
assigned by this cleanup; third-party code, weights and datasets have their own
terms.
