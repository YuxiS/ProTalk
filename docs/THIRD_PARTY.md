# Third-party code and assets

The checkout includes or adapts code from several projects. Existing notices
are preserved. This list describes identifiable components; it is not a complete
license audit or a grant of redistribution rights.

| Component | Evidence in this checkout |
| --- | --- |
| Deep3DFaceRecon_pytorch | Headers in `protalk/third_party/deep3d/models/base_model.py`, reconstruction modules |
| PIRender | `protalk/third_party/face_utils/renders/PIRender/` |
| Mellotron / GST | `protalk/models/gst.py`, `configs/model.yaml`, GST loading in `protalk/models/expression.py` |
| PyTorch STFT | BSD 3-Clause notice in `protalk/audio/stft.py` (Prem Seetharaman) |
| EMOCA expression loss | `protalk/third_party/emoca/expression_loss.py` explicitly references EMOCA and MPG proprietary terms |
| VQ-VAE implementation | Original source references retained under `protalk/models/pose/UPSTREAM.md` |
| I3D / FVD | Historical evaluation only; archived locally, excluded from the maintained remote tree |
| Wav2Lip / GFPGAN / Real-ESRGAN | Optional external postprocessing; not bundled model weights |

The EMOCA-derived header refers to a LICENSE file that is absent in this checkout.
The owner must resolve the applicable redistribution terms before a public release.
Face-model assets, datasets and pretrained checkpoints have separate terms.
Do not infer a license for these from the availability of ProTalk source code.
No top-level license has been added because its choice and compatibility require
an owner decision. Notebook cache copies and local editor settings are excluded
from version control without deleting local originals.
