# Third-party code and assets

The checkout includes or adapts code from several projects. Existing notices
are preserved. This list describes identifiable components; it is not a complete
license audit or a grant of redistribution rights.

| Component | Evidence in this checkout |
| --- | --- |
| Deep3DFaceRecon_pytorch | Headers in `deep3d/models/base_model.py`, reconstruction modules |
| PIRender | `face_utils/renders/PIRender/` |
| Mellotron / GST | `modules.py`, `hparams.yaml`, GST loading in `stylemodel.py` |
| PyTorch STFT | BSD 3-Clause notice in `stft.py` (Prem Seetharaman) |
| EMOCA expression loss | `Visual/expression_loss.py` explicitly references EMOCA and MPG proprietary terms |
| VQ-VAE implementation | Original attribution/readme retained under `vqvae/README.md` |
| I3D / FVD | `metrics/FVD/pytorch_i3d_model/README.mb` and associated modules |
| Wav2Lip / GFPGAN / Real-ESRGAN | Optional external postprocessing; not bundled model weights |

The EMOCA-derived header refers to a LICENSE file that is absent in this checkout.
The owner must resolve the applicable redistribution terms before a public release.
Face-model assets, datasets and pretrained checkpoints have separate terms.
Do not infer a license for these from the availability of ProTalk source code.
No top-level license has been added because its choice and compatibility require
an owner decision. Notebook cache copies and local editor settings are excluded
from version control without deleting local originals.
