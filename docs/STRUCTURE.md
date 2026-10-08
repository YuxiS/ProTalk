# Repository structure and migration map

The public workflow uses one Python package, `protalk`, with a command dispatcher
(`python -m protalk`, or `protalk` after editable installation). Root-level model
files and obsolete experiment entries have been relocated to make supported code
and historical scripts easy to distinguish. The archived files remain on the
author's machine but are excluded from Git; only `archive/README.md` is uploaded. Model class names, module attributes
and state-dict tensor keys are retained, so this reorganization does not change
checkpoint formats or introduce a new model.

## Maintained code

| Location | Responsibility |
| --- | --- |
| `protalk/config.py`, `paths.py`, `runtime.py` | YAML loading, repository paths and runtime helpers |
| `protalk/audio/` | Wav2Lip-derived audio helpers, MFCC/prosody, STFT and YIN |
| `protalk/models/expression.py`, `gst.py`, `layers.py`, `attention/` | Existing expression, style-token and attention components |
| `protalk/models/pose/vqvae/`, `sampler/`, `generate.py` | Existing pose VQ-VAE, code sampler and generation |
| `protalk/training/` | Preparation, datasets, objectives, training/resume/exports and environment checks |
| `protalk/inference/` | Supported reference-relative video generation; lightweight argument parser |
| `protalk/evaluation/beat_scores.py` | Historical BAS/SBAS formulas, shared by archived evaluation |
| `protalk/third_party/` | Integrated face reconstruction/rendering and optional emotion loss |
| `configs/model.yaml`, `configs/train.yaml` | Architecture/audio parameters and training settings |
| `scripts/` | Launchers, explicit downloads and optional restoration |
| `assets/`, `weights/`, `data/` | External face assets, learned checkpoints and prepared data |

The generation argument parser does not import CUDA reconstruction dependencies
before processing `--help`. Basic CPU training tests require no face weights.
Third-party imports are namespaced and the Deep3D dynamic model loader uses its
own package rather than a root-level `models` package.

## Old to new paths

| Previous path/command | Current path/command |
| --- | --- |
| `hparams.py`, `hparams.yaml` | `protalk/config.py`, `configs/model.yaml` |
| `stylemodel.py`, `modules.py`, `layers.py` | `protalk/models/expression.py`, `gst.py`, `layers.py` |
| `multi_head_attention/` | `protalk/models/attention/` |
| `audio_processing.py`, `audio_wav2lip.py`, `proso_features.py`, `stft.py`, `yin.py` | `protalk/audio/processing.py`, `wav2lip.py`, `prosody.py`, `stft.py`, `yin.py` |
| `vqvae/models/`, `vqvae/pose_sampler/`, `vqvae/generate.py` | `protalk/models/pose/vqvae/`, `sampler/`, `generate.py` |
| `training/` / `python -m training.run` | `protalk/training/` / `python -m protalk train` |
| `reference.py` | `protalk/inference/generate.py`; run `python -m protalk generate` |
| `generate.sh` | `scripts/generate.sh` |
| `train_style.sh`, `train_vqvae.sh`, `train_pose_sampler.sh` | `scripts/train_expression.sh`, `scripts/train_vqvae.sh`, `scripts/train_sampler.sh` |
| `deep3d/`, `face_utils/`, `Visual/` | `protalk/third_party/deep3d/`, `face_utils/`, `emoca/` |
| `deep3d/BFM/`, `deep3d/checkpoints/` | `assets/bfm/`, `assets/reconstruction/` |
| `inference_gfpgan.py` | `scripts/restore_video.py` |
| `inference.py` | `archive/inference/later_preset.py` |
| `train_style.py`, `train_script/`, `CoeffDataset.py` | `archive/training/train_style.py`, `train_script/`, `dataset.py` |
| `model.py`, `facial_exp/` | `archive/models/model.py`, `facial_exp/` |
| `abliationStudy/`, `compare/`, `metrics/` | `archive/ablation/`, `comparison/`, `metrics/` |
| `metrics/beat_scores.py` | `protalk/evaluation/beat_scores.py` |
| `requirements-training.txt`, `requirements-test.txt` | `requirements/training.txt`, `requirements/test.txt` |

Shell scripts now switch to the repository root. Update local scripts/imports
using the table; old root-level names are not duplicated as compatibility copies.
Full training checkpoints from an old layout may contain old config paths; use
matching inference exports and point to their exact model config/statistics, or
adapt the recorded training setup before resuming. No weights are converted by
this migration.

## Dependencies and installation

`requirements/training.txt` supplies CPU-test/core training dependencies;
`requirements/inference.txt` adds full reconstruction/inference dependencies.
Root `requirements.txt` includes the inference list for familiar installation.
Use an editable install (`python -m pip install -e . --no-deps`) with Python 3.9.
Config and asset paths belong to the checkout, so a standalone wheel deployment
is not supported by this repository layout.

`archive/` is excluded from both package discovery and version control, apart
from its README. Old training/ablation/evaluation scripts, result arrays, notebooks
and unused third-party demos are not part of the remote release. Downloaded assets, datasets,
checkpoints, build metadata and Python bytecode are excluded from Git. Existing
component attribution is preserved; historical saved metric arrays remain local.
