# Code review and reproducibility notes

## Scope of this cleanup

The baseline is commit `7840024`. The cleanup does not add a model, run a new
experiment, regenerate videos or change manuscript results. Existing checkpoints
retain their architecture and state-dict keys. The following fixes affect future
runs and therefore must not be presented as retroactively validated experiments.

## Fixed issues

- `hparams.py` imported a missing, unused Tacotron `text.symbols` package. The
  loader now reads the YAML mapping directly, with a repo-relative default.
- Inference previously ignored expression checkpoint CLI options and embedded
  server paths for all checkpoints. All three learned components now take the
  supplied paths. `inference.py` now uses the supplied image/audio instead of a
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
- `loss_function.py` no longer imports an unused ESPnet lipreading model.
- Metrics load optional FVD dependencies lazily. BAS/SBAS formulas now have a
  shared implementation, retaining original orientation, sigma and sum scaling.
- Shell commands quote paths, stop on errors, use repo-relative entries and avoid
  personal conda activation. Generation and optional postprocessing are separate.
  GFPGAN's folder input/output contract is respected, and FFmpeg muxes audio.
- Generated notebook checkpoints, `.DS_Store` and personal editor settings are
  removed from the Git index only. Their local originals are retained.

## Details requiring care when interpreting the paper

1. **Inference presets differ.** `reference.py` uses epochs 99/499/99 and mean/std;
   `inference.py` uses 79/999/399 and mean_wild/std_wild. Pose scaling, reference
   offsets and smoothing differ as listed in `ASSETS.md`. The correct preset and
   released checkpoints behind each result remain to be identified by the author.
2. **The stored BAS implementation is directional.** It averages over audio beats
   and finds nearest motion beats. The symmetric implementation in `metrics/test.py`
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
   is active in both entries; pose smoothing is active only in `inference.py`.
5. **GST position embedding** is computed and masked in `cal_gst_feature`, but the
   returned tensor is the repeated global embedding. The positional tensor is not
   consumed. This existing behavior remains unchanged and should be reconciled
   with the method description before submission.
6. **VQ loss conventions differ by class.** The ordinary quantizer computes the
   two squared mean losses; the active VQVAE uses the EMA quantizer, with a
   beta-weighted commitment term and EMA codebook updates. Do not conflate these
   paths when describing the training implementation.
7. **Training is not yet certified reproducible.** VQ-VAE's loader hardcodes batch
   1024 while its CLI default is 256; this was left unchanged. Validation is not
   invoked by the current main training loops. The MFCC pose ablation still uses
   a stride-3 fuse layer designed for prosody (three samples per frame), so its
   time-length protocol needs author verification before running that ablation.
   Old `train_sampler.py` references missing `models_easy` and `TextMelLoader`;
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
