# External assets

Downloaded assets are separate from Python source and are ignored by Git.

```text
assets/
  bfm/                     BFM_model_front.mat, similarity_Lm3D_all.mat and conversion inputs
  reconstruction/
    face_recon/epoch_20.pth Deep3D reconstruction checkpoint
weights/
  pirender.pt              External PIRender generator (net_G_ema)
  retrained/
    expression/            Newly trained expression exports/checkpoints
    vqvae/                 Newly trained pose codebook exports/checkpoints
    sampler/               Newly trained pose sampler exports/checkpoints
data/prepared/mean_std/     Statistics fitted to your training split
```

See [asset sources and compatibility](../docs/ASSETS.md). The helper
`python scripts/download_assets.py reconstruction landmarks` downloads only the
selected public auxiliary files. It does not provide ProTalk-trained checkpoints,
PIRender or a complete licensed BFM installation. Override CLI/config paths if
using existing files in another location.
