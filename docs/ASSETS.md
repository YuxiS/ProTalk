# Required local assets

No download link or published checkpoint is asserted here. Supply the exact
checkpoints/statistics used for the selected experiment. Paths below are defaults
or configurable through the shown option.

| Asset | Configuration/location | Expected contents |
| --- | --- | --- |
| Expression checkpoint | `--model_weight` | Dictionary containing `audio_model` |
| Pose VQ-VAE | `--vae_weight` | `CodeBook`, `Decoder` (training also uses `Encoder`) |
| Pose sampler | `--sampling_weight` | PoseSampler state dict |
| PIRender | `--pirender_weight` | Dictionary containing `net_G_ema` |
| Mellotron GST pretraining | YAML `gst_weight` | `state_dict` containing `gst.*` |
| Normalization statistics | YAML `mean_std_root` or `--mfcc_mean_std_root` | `mfcc_mean.npy`, `mfcc_std.npy`; `mean.npy`, `std.npy` for reference; `mean_wild.npy`, `std_wild.npy` for inference |
| Basel face model assets | `--bfm_folder` (default `deep3d/BFM`) | `BFM_model_front.mat`, `similarity_Lm3D_all.mat`, and upstream auxiliary files if conversion is needed |
| Reconstruction checkpoint | `--checkpoints_dir`, `--name`, `--epoch` | default `deep3d/checkpoints/face_recon/epoch_20.pth`, containing `net_recon` |
| Reconstruction initialization | `--init_path` | default `deep3d/checkpoints/resnet50-0676ba61.pth` |
| Landmark detector weights | face-alignment cache | Managed by face-alignment |
| Expression-loss network | training only | `data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt` |

Standard deviations must be positive and finite. Use training-set statistics,
not statistics computed on the test set. Feature extraction produces 244 values
per video frame and the assembled 3DMM vector has 257 coefficients. These assets
and their provenance must be supplied before claiming reproducibility.

## Presets retained from the original source

| Entry | Expression | VQ-VAE | Sampler | Coefficient statistics | Motion processing |
| --- | --- | --- | --- | --- | --- |
| `reference.py` | epoch 99 | epoch 499 | epoch 99 | mean/std | Expression smoothing; pose multiplier 2 with reference offsets |
| `inference.py` | epoch 79 | epoch 999 | epoch 399 | mean_wild/std_wild | Expression and pose smoothing; pose re-centering and multiplier 0.6 |

Epoch numbers are historical defaults, not evidence that these are the released
weights or the weights behind a specific table. Use explicit paths to disambiguate.
