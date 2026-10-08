# Historical experiments

On the author's machine, this directory retains the original research and
experiment variants. **Only this README is tracked; the archived files are not
committed or uploaded.** Remote users can inspect the pre-reorganization code in
commit `2df0b4a` and earlier Git history. They are outside the maintained package and excluded from its editable
installation. They may require missing weights, dataset-specific paths, optional
dependencies or historical modules such as `models_easy`. They have not been
validated as portable training/evaluation commands.

| Folder | Original material |
| --- | --- |
| `training/` | Old expression/pose loops, data loader, loss/logger and RNN launchers |
| `models/` | Historical model and facial-expression variants |
| `ablation/` | Former `abliationStudy/` scripts |
| `comparison/` | Former `compare/` scripts |
| `metrics/` | Historical metric scripts, FID/FVD integration and saved result arrays |
| `inference/` | Later inference preset, differing from supported reference-relative generation |
| `data_tools/` | Historical crop/data helper |
| `vendor_unused/` | Unused third-party training, demos and utilities |
| `pose/` | Upstream VQ-VAE demo/training helpers, notebook and original attribution |
| `utils.py` | Experiment-specific utilities |

Use `python -m protalk` and the root README for the supported workflow.
The old numerical result arrays are retained unchanged. Downloaded CIFAR demo
archives and compiled Python cache files are kept locally but excluded from Git.
Old paths and model architecture names are documented in [STRUCTURE.md](../docs/STRUCTURE.md).
