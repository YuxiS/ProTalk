# Integrated third-party code

- `deep3d/`: Deep3DFaceRecon reconstruction, BFM geometry and its vendored ArcFace modules.
- `face_utils/`: PIRender and video I/O utilities.
- `emoca/`: EMOCA-derived expression-loss network and visual-loss helpers (formerly `Visual/`).

Source notices are preserved. Integration imports use the `protalk.third_party`
namespace; trained weights and face-model data live outside the source tree.
Unused standalone upstream demo/training scripts are preserved in the local
archive and excluded from the remote release. See [third-party notes](../../../docs/THIRD_PARTY.md)
and [asset requirements](../../../docs/ASSETS.md) for provenance and terms.
