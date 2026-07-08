"""Top-level package for openSIP."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__author__ = "Xiaohao Yang"
__email__ = "xiaohaoy111@gmail.com"

try:
    __version__ = _pkg_version("openSIP")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .dataset import (
    GeoTaggedData,
    enrich_aporee_catalog,
    fetch_aporee_catalog,
    fix_aporee_urls,
    getPhoto,
    getSound,
    getSoundAporee,
    getSV,
)
from .pairing import AudioImagePairDataset
from .placepulse import PlacePulsePredictor

__all__ = [
    "__version__",
    "GeoTaggedData",
    "AudioImagePairDataset",
    "PlacePulsePredictor",
    "getSV",
    "getPhoto",
    "getSound",
    "getSoundAporee",
    "fetch_aporee_catalog",
    "enrich_aporee_catalog",
    "fix_aporee_urls",
]
