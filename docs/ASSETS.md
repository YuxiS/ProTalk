# Required local assets

Public third-party sources are listed below. Supply the exact
checkpoints/statistics used for the selected experiment. Paths below are defaults
or configurable through the shown option.

| Asset | Configuration/location | Expected contents |
| --- | --- | --- |
| Expression checkpoint | `--model_weight` | Dictionary containing `audio_model` |
| Pose VQ-VAE | `--vae_weight` | `CodeBook`, `Decoder` (training also uses `Encoder`) |
| Pose sampler | `--sampling_weight` | PoseSampler state dict |
| PIRender | `--pirender_weight` | Dictionary containing `net_G_ema` |
| Mellotron GST pretraining | YAML `gst_weight` | Optional for `gst_init: pretrained` training only; `state_dict` containing `gst.*`. Inference loads GST from the expression export. |
| Normalization statistics | YAML `mean_std_root` or `--mfcc_mean_std_root` | `mfcc_mean.npy`, `mfcc_std.npy`; `mean.npy`, `std.npy` for reference; `mean_wild.npy`, `std_wild.npy` for inference |
| Basel face model assets | `--bfm_folder` (default `assets/bfm`) | `BFM_model_front.mat`, `similarity_Lm3D_all.mat`, and upstream auxiliary files if conversion is needed |
| Reconstruction checkpoint | `--checkpoints_dir`, `--name`, `--epoch` | default `assets/reconstruction/face_recon/epoch_20.pth`, containing `net_recon` |
| Reconstruction initialization | `--init_path` | Optional when a complete `net_recon` checkpoint is loaded; no separate ImageNet file required |
| Landmark detector weights | face-alignment cache | Managed by face-alignment |
| Expression-loss network | optional visual fine-tuning only | `data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt` |

Standard deviations must be positive and finite. Use training-set statistics,
not statistics computed on the test set. Feature extraction produces 244 values
per video frame and the assembled 3DMM vector has 257 coefficients. These assets
and their provenance must be supplied before claiming reproducibility.

## Presets retained from the original source

| Entry | Expression | VQ-VAE | Sampler | Coefficient statistics | Motion processing |
| --- | --- | --- | --- | --- | --- |
| Original `reference.py` (math retained in `protalk.inference.generate`) | epoch 99 | epoch 499 | epoch 99 | mean/std | Expression smoothing; pose multiplier 2 with reference offsets |
| `archive/inference/later_preset.py` | epoch 79 | epoch 999 | epoch 399 | mean_wild/std_wild | Expression and pose smoothing; pose re-centering and multiplier 0.6 |

Epoch numbers are historical defaults, not evidence that these are the released
weights or the weights behind a specific table. Use explicit paths to disambiguate.


## SadTalker asset sources and compatibility

Checked on 2026-10-08 against the official [download script](https://github.com/OpenTalker/SadTalker/blob/main/scripts/download_models.sh), [reconstruction loader](https://github.com/OpenTalker/SadTalker/blob/main/src/utils/preprocess.py), and [legacy release](https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2).

| Asset | Source and local destination | Assessment |
| --- | --- | --- |
| epoch_20.pth | [Download](https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/epoch_20.pth); assets/reconstruction/face_recon/epoch_20.pth | Candidate reconstruction checkpoint: the ASTs of ReconNetWrapper, ResNet, BasicBlock and Bottleneck match this checkout, and the upstream loader uses net_recon. Actual loading is untested. |
| similarity_Lm3D_all.mat | [Download](https://raw.githubusercontent.com/OpenTalker/SadTalker/main/src/config/similarity_Lm3D_all.mat); assets/bfm/similarity_Lm3D_all.mat | Shared alignment asset; does not replace the full BFM face model. |
| BFM_Fitting.zip | [Download](https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip) | Candidate fitting assets. Inspect archive contents and MAT fields before using; complete renderer compatibility is unverified. |
| wav2lip.pth | [Download](https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/wav2lip.pth); weights/wav2lip.pth | Optional postprocessing; requires a matching Wav2Lip checkout and environment. |
| GFPGANv1.4.pth | [Download](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth); gfpgan/weights/GFPGANv1.4.pth | Optional restoration; source is the official GFPGAN project as linked by SadTalker. |
| auido2exp, auido2pose, complete SadTalker safetensors | SadTalker-specific models | Cannot directly replace ProTalk expression, VQ-VAE or pose-sampler checkpoints. |
| facevid2vid and mapping checkpoints | SadTalker facevid2vid/SPADE renderer | Cannot directly replace ProTalk PIRender, which expects net_G_ema. |

No weights were downloaded in this inspection. The reconstruction checkpoint is approximately 289 MB, BFM fitting archive 404 MB, and Wav2Lip 436 MB. A structurally compatible reconstruction checkpoint is not proof that it was used for ProTalk's original results.

SadTalker's modern safetensors bundle contains reconstruction tensors, but the current ProTalk loader does not extract that format. The legacy standalone checkpoint fits the existing loading contract. Original ProTalk-trained expression/VQ-VAE/sampler weights and normalization statistics are unavailable. Train them and fit training statistics with `protalk/training/` instead; PIRender remains a separate dependency. Mellotron is optional when training GST from scratch. Asset terms remain separate from the repository license.


## Assets for the maintained workflow

Basic training on prepared arrays needs no third-party model checkpoint. For
raw coefficient extraction and inference, acquire the face assets above and a
PIRender checkpoint. The helper downloads explicitly selected public assets:

```bash
python scripts/download_assets.py reconstruction landmarks
# Optional; this does not unpack or validate the archive as a complete BFM setup:
python scripts/download_assets.py bfm-fitting
```

Downloads are skipped if the destination exists. Partial files use `.part` until
complete; a sidecar records the source URL, size and calculated SHA256. This hash
records what was received, not independent authentication of the published asset.
Large asset downloads have not been exercised during cleanup.

PIRender's [official README](https://github.com/RenYurui/PIRender) provides its
[pretrained model archive](https://drive.google.com/file/d/1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7/view?usp=sharing).
Unpack it, locate the face generator checkpoint (historically
`epoch_00190_iteration_000400000_checkpoint.pt`) and copy it to `weights/pirender.pt`
or configure `pirender_weight`/`--pirender_weight`. ProTalk's loader requires
`net_G_ema` for its 73-channel face generator; actual downloaded checkpoint loading
has not been tested in this environment.

For BFM conversion follow the [Deep3DFaceRecon instructions](https://github.com/sicxu/Deep3DFaceRecon_pytorch).
Obtain the original [BFM09 assets](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads)
and upstream expression basis according to their terms, place the required
conversion inputs in `assets/bfm`, and produce `BFM_model_front.mat` with the
upstream-compatible converter in this checkout:

```bash
python -c "from protalk.third_party.deep3d.util.load_mats import transferBFM09; transferBFM09('assets/bfm')"
```

The converter also requires `std_exp.txt`, `BFM_front_idx.mat`, `BFM_exp_idx.mat`
and `facemodel_info.mat` from the upstream fitting assets. The landmark MAT alone is
insufficient. Inspect any SadTalker fitting archive before using it as this input.

For optional frozen-GST training, Mellotron's [official README](https://github.com/NVIDIA/mellotron)
links its [LibriTTS checkpoint](https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI).
Point `gst_weight` in hparams to the downloaded file and select
`expression.gst_init: pretrained`. Newly trained expression exports already
contain GST parameters.

For new exports, `scripts/generate.sh` uses `weights/retrained/<stage>/best-inference.pth`
and `data/prepared/mean_std`. The historical preset table above describes old
source defaults, not the maintained training recipe. New expression metadata
selects a pose multiplier of 1 unless explicitly overridden.
