"""Beat matching formulas used by the historical evaluation scripts.

Inputs must share the same time units. Sigma is 8 in those units. SBAS is
an unhalved sum, as in metrics/test.py; its maximum is 2, not 1.
"""
import numpy as np


def _directional(source, target, sigma):
    source = np.asarray(source, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if not source.size or not target.size:
        raise ValueError("Both beat sequences must contain at least one beat")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Beat positions must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    return float(np.mean([np.exp(-np.min((target - beat) ** 2) / (2 * sigma ** 2))
                          for beat in source]))


def bas(audio_beats, motion_beats, sigma=8):
    """Audio-to-motion matching, preserving beat_align_score.BA orientation."""
    return _directional(audio_beats, motion_beats, sigma)


def sbas(audio_beats, motion_beats, sigma=8):
    """Sum audio-to-motion and motion-to-audio matching."""
    return bas(audio_beats, motion_beats, sigma) + bas(motion_beats, audio_beats, sigma)
