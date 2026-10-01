"""
utils/reid.py — REMOVED (torch/torchvision/YOLO dropped for 1 GB RAM budget).

This module previously loaded a ResNet50 backbone for body re-identification.
That pipeline consumed ~500 MB and was the primary OOM cause on Community Cloud.

Face-only 512-D embeddings from InsightFace buffalo_s are now sufficient for
person clustering.  This stub is kept so any stale imports fail gracefully
instead of silently.
"""


def get_reid_extractor():
    """No-op stub — ReID has been removed to fit within 1 GB RAM."""
    raise RuntimeError(
        "ReID module was removed.  Use InsightFace face embeddings only.  "
        "If you see this error, a caller still references the old pipeline."
    )


def extract_body_embedding(*_args, **_kwargs):
    """No-op stub — see get_reid_extractor docstring."""
    raise RuntimeError("ReID module was removed.")