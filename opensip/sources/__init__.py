"""Per-API source modules. Each re-exports the corresponding free function
defined in :mod:`opensip.dataset`. Provided for a more discoverable layout:

    from opensip.sources.mapillary import getSV
    from opensip.sources.flickr import getPhoto
    from opensip.sources.freesound import getSound
    from opensip.sources.aporee import getSoundAporee
"""
from __future__ import annotations

from . import aporee, flickr, freesound, mapillary  # noqa: F401

__all__ = ["mapillary", "flickr", "freesound", "aporee"]
