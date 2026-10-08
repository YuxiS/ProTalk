# Code review and reproducibility notes

The initial review below describes the historical loops. For current supported
training, use `protalk/training/` and [TRAINING.md](TRAINING.md), not the archival Python
entries. Original result checkpoints remain unavailable.

## Scope of this cleanup

The baseline is commit `23ddaf3`. The cleanup does not add a model, run a new
experiment, regenerate videos or change manuscript results. Existing checkpoints
retain their architecture and state-dict keys. The following fixes affect future
runs and therefore must not be presented as retroactively validated experiments.

## Fixed issues

- `protalk/config.py` imported a missing, unused Tacotron `text.symbols` package. The
  loader now reads the YAML mapping directly, with a repo-relative default.
- Inference previously ignored expression checkpoint CLI options and embedded
  server paths for all checkpoints. All three learned components now take the
  supplied paths. `archive/inference/later_preset.py` now uses the supplied image/audio instead of a
  hardcoded example. Output directories are created before writing.
- `shapee` in coefficient length alignment caused an exception for a shorter
  pose sequence. Both sequences now trim to their shared length, with a minimum
  five-frame check before the original valid Gaussian convolution.
- Constant pitch/energy gave zero-division NaNs. These signals now map to zeros
  in both inference and the dataset loader. Nonconstant min/max scaling is
  unchanged. Inference rejects invalid standard deviations and missing faces.
- CPU/device assumptions in vector quantization, code lookup, the sampler and
  GST lengths were replaced with tensor devices. Full face rendering still
  requires CUDA; this is not a CPU inference implementation.
- EMA accumulated weights were allocated from uninitialized memory. They now
  start from the embedding weights, and EMA updates preserve Parameter objects
  rather than replacing them each step. State-dict names and shapes are retained.
  This fixes new training; saved codebooks load their existing state unchanged.
- Boolean CLI parsing treated `False` as true. Training now reads `LOCAL_RANK`
  from torchrun, sets the distributed sampler epoch, advances schedulers on all
  ranks, and supports saving an unwrapped single-GPU model where applicable.
- Expression and pose training variants are explicit. The style script exposes
  the existing ProsoResNet with `--prosody` while direct Python invocation retains
  the original ResNet default. Pose input selection distinguishes prosody from
  MFCC; training and validation use the same selection.
- The unused expression validation function used a removed batch parser and an
  incompatible forward signature. It was removed; no replacement validation or
  claimed evaluation has been added.
- `archive/training/loss_function.py` no longer imports an unused ESPnet lipreading model.
- Metrics load optional FVD dependencies lazily. BAS/SBAS formulas now have a
  shared implementation, retaining original orientation, sigma and sum scaling.
- Shell commands quote paths, stop on errors, use repo-relative entries and avoid
  personal conda activation. Generation and optional postprocessing are separate.
  GFPGAN's folder input/output contract is respected, and FFmpeg muxes audio.
- Generated notebook checkpoints, `.DS_Store` and personal editor settings are
  removed from the Git index only. Their local originals are retained.

## Details requiring care when interpreting the paper

1. **Historical inference presets differ.** Original `reference.py` used epochs
   99/499/99 and mean/std; the later preset used 79/999/399 and mean_wild/std_wild.
   Maintained generation now defaults to the newly trained best exports. Pose scaling, reference
   offsets and smoothing differ as listed in `ASSETS.md`. The correct preset and
   released checkpoints behind each result remain to be identified by the author.
2. **The stored BAS implementation is directional.** It averages over audio beats
   and finds nearest motion beats. The symmetric implementation in `archive/metrics/test.py`
   is the sum of both directions, with range [0, 2]. It has not been divided by 2.
   Existing comparison/ablation wrappers call the directional implementation.
   None of these files proves which implementation generated a manuscript table.
3. **Beat indices have a time-base issue to verify.** Librosa beat extraction uses
   its default hop of 512 at 22050 Hz; motion coefficients use video-frame indices
   (normally 30 fps). Historical scripts compare these indices directly. The
   formulas have been preserved, not silently rescaled: check the original
   experiment protocol before claiming correct physical-time alignment or changing
   reported scores. Empty beat sets return -1 in legacy wrappers and are filtered
   by comparison scripts; the new pure functions instead raise ValueError.
4. **Gaussian smoothing** uses a normalized size-5 kernel, std=3, valid grouped
   convolution and two repeated boundary values at each end. Expression smoothing
   is active in both entries; pose smoothing is active only in `archive/inference/later_preset.py`.
5. **GST position embedding** is computed and masked in `cal_gst_feature`, but the
   returned tensor is the repeated global embedding. The positional tensor is not
   consumed. This existing behavior remains unchanged and should be reconciled
   with the method description before submission.
6. **VQ loss conventions differ by class.** The ordinary quantizer computes the
   two squared mean losses; the active VQVAE uses the EMA quantizer, with a
   beta-weighted commitment term and EMA codebook updates. Do not conflate these
   paths when describing the training implementation.
7. **Training is not yet certified reproducible.** The archival VQ-VAE loader hardcodes batch
   1024 while its CLI default is 256; this was left unchanged. Validation is not
   invoked by the archival main training loops. The maintained `protalk.training.run`
   entry validates every epoch and honors configured batch sizes. The MFCC pose ablation still uses
   a stride-3 fuse layer designed for prosody (three samples per frame), so its
   time-length protocol needs author verification before running that ablation.
   Old `archive/training/train_script/train_sampler.py` references missing `models_easy` and `TextMelLoader`;
   it is an archival experiment, not the supported pose-sampler entry.
8. **Stochastic sampling** initializes the head LSTM state with random tensors,
   and the sampler module sets a time-based CUDA seed at import. These historical
   choices remain unchanged; exact repeated-run reproducibility requires a defined
   seed protocol and recording the original environment.
9. **Release assets and terms remain outstanding.** No weights, normalization
   statistics, dataset split lists or tested CUDA environment are supplied here.
   The EMOCA-derived source has a proprietary license notice referencing a missing
   license file. Resolve this in the public release; see `THIRD_PARTY.md`.

## Validation performed

CPU tests were run with Python 3.9.6, torch 2.2.2, NumPy 1.23.5, SciPy 1.10.1
and PyYAML 6.0.2 on macOS ARM64. The nine regression tests cover pure metric
math, normalization, length alignment, Gaussian smoothing, CPU quantization,
EMA parameter identity, codebook lookup and pose-sampler forward shapes, plus
Python and shell syntax checks. Full GPU training, rendering, restoration and
paper-table reproduction were not run. Candidate requirements for the complete
pipeline are not a tested environment lockfile.


## Maintained retraining workflow (2026-10-08)

A new portable preparation/training package supports JSONL paths, raw MAT/NPZ
inputs, cached MFCC/prosody arrays and training-only normalization statistics.
It trains the existing expression/VQ-VAE/sampler network classes, validates each
epoch and exports checkpoints accepted by `protalk.inference.generate`/PoseGEN. It saves
optimizer, scheduler and per-rank RNG states for resume and synchronizes EMA
codebook updates across distributed ranks.

This is a new training recipe because original training assets are lost. Default
scratch GST and coefficient supervision, reference-relative targets, masked losses
and cross-entropy sampler training are explicit in TRAINING.md. The old decoded
argmax penalty does not propagate gradients through code selection; it is not
included in the maintained sampler objective. Optional visual fine-tuning freezes
third-party networks but permits gradients to expression coefficients. No new
model architecture or regenerated paper metric is claimed.

CPU verification includes all three real network stages, validation, resumed
sampler training, exported expression/GST/codebook/sampler loading, raw waveform
feature dimensions and gradients through frozen PIRender. Two-process CPU Gloo
runs of all three stages also completed training, validation and checkpoint saving. These checks
use synthetic data and reduced pose dimensions. Full CUDA training, external
checkpoint compatibility, coefficient extraction and video quality remain
unverified. CI runs the CPU regression suite; no trained weights are supplied.


## Package reorganization (2026-10-08)

Supported code now lives in `protalk/`; configuration, launchers and dependencies
live in `configs/`, `scripts/` and `requirements/`. Historical experiments are
preserved locally in `archive/` and excluded from the remote release. Third-party source and external assets are separate.
Generated bytecode and the unrelated downloaded CIFAR demo archive have been
removed from version control while retaining local files. See STRUCTURE.md for
path changes. The new CLI supports help without loading CUDA reconstruction,
and the package is installed in editable mode.


The current remote tree contains the maintained workflow and its dependencies.
Historical scripts/results, unused upstream training/demos, notebooks and caches
are local-only. To inspect their original source remotely, use commit `2df0b4a`
or earlier history. Repository history was subsequently rewritten to remove saved results, demo
data and generated caches from every reachable commit.


After reorganization, 18 CPU tests passed, including all maintained CLI help
commands outside the checkout, default config lookup, Deep3D's namespaced dynamic
loader and the three-stage training/resume/export check. Core computations in
14 model/audio modules and the generation functions were compared with the prior
commit and were unchanged apart from imports. Pre-reorganization synthetic exports
also loaded in the new package and produced finite outputs. A two-process VQ-VAE
run through the reorganized module completed training, validation and export.


## Historical artifact removal (2026-10-08)

The branch history was rewritten to remove 14 saved comparison-result arrays,
three plot images, the historical notebook/CIFAR demo archive, Python bytecode,
notebook checkpoints and personal editor/cache files (74 historical paths total).
The maintained source tree was verified identical before documentation updates.
Original history is backed up outside the repository and is not uploaded.
Existing clones from before this rewrite should be replaced with a fresh clone
or deliberately reconciled; merging old history would reintroduce removed files.
