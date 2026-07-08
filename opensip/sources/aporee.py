"""Radio Aporee audio source.

Re-exports the helpers that live in :mod:`opensip.dataset`:

* :func:`getSoundAporee` — filter a catalog by spatial proximity
* :func:`fetch_aporee_catalog` — fetch the catalog from Internet Archive
* :func:`enrich_aporee_catalog` — probe URLs for ``duration_s``
* :func:`fix_aporee_urls` — repair wrong ``{identifier}.mp3`` URLs in an
  existing catalog built before ``verify_urls=True`` became the default
"""
from __future__ import annotations

from ..dataset import (
    enrich_aporee_catalog,
    fetch_aporee_catalog,
    fix_aporee_urls,
    getSoundAporee,
)

__all__ = [
    "getSoundAporee",
    "fetch_aporee_catalog",
    "enrich_aporee_catalog",
    "fix_aporee_urls",
]
