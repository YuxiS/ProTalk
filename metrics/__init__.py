"""Evaluation utilities; load the optional FVD dependencies only when requested."""

def frechet_video_distance(*args, **kwargs):
    from .FVD.frechet_video_distance import frechet_video_distance as compute
    return compute(*args, **kwargs)
