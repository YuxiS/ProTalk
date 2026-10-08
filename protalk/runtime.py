"""Small shared helpers for command-line parsing and inference edge cases."""
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def minmax_normalize(values):
    """Preserve min/max scaling; a constant signal maps to zeros."""
    span = values.max() - values.min()
    if span.item() == 0:
        return values * 0
    return (values - values.min()) / span


def align_coefficients(expression, pose, min_frames=5):
    """Trim both sequences to their shared length before valid convolution."""
    frames = min(expression.shape[1], pose.shape[1])
    if frames < min_frames:
        raise ValueError(f"Audio must produce at least {min_frames} video frames; got {frames}")
    return expression[:, :frames, :], pose[:, :frames, :]
