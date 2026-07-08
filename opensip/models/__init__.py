"""Bundled model assets for openSIP."""

from __future__ import annotations

from importlib.resources import files


def get_yunet_model_path():
    """Return the packaged YuNet face detector ONNX resource."""
    return files(__package__) / "face_detection_yunet_2023mar.onnx"


__all__ = ["get_yunet_model_path"]
