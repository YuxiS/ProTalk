import os
import random
from pathlib import Path
import numpy as np
import torch


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def inference_export(stage, model, metadata):
    if stage == 'expression':
        return {'audio_model': model.state_dict(), 'metadata': metadata}
    if stage == 'vqvae':
        return {'Encoder': model.encoder.state_dict(), 'Decoder': model.decoder.state_dict(),
                'CodeBook': model.vector_quantization.state_dict(), 'metadata': metadata}
    # Legacy PoseGEN expects a bare state dict for its sampler.
    return model.state_dict()
