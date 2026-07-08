"""opensip/pairing.py — Audio–Image pair builder with ImageBind similarity.

Typical usage
-------------
>>> apd = AudioImagePairDataset(locations=[[-83.073, 42.348]])
>>> apd.get_sound_from_location(source="aporee", distance=100)
>>> apd.get_svi_from_locations(source="google", key="...")
>>> apd.build_pairs(image_source="svi")
>>> apd.compute_embedding_similarity(output_dir="pairs/audio")
>>> apd.filter_pairs_by_similarity(min_similarity=0.2, top_k_per_audio=1)
>>> apd.export("pairs/")
>>> apd.plot()
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import base64
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset import GeoTaggedData

__all__ = ["AudioImagePairDataset"]

# ---------------------------------------------------------------------------
# Metadata column order
# ---------------------------------------------------------------------------

_PAIR_COLS = [
    "loc_id",
    "audio_source",
    "audio_id",
    "clip_index",
    "clip_start_ms",
    "clip_end_ms",
    "audio_url",
    "audio_ts",
    "image_source",
    "image_id",
    "image_url",
    "image_ts",
    "ts_precision",
    "time_diff_days",
    "n_sub_clips",
    "similarity",
]

_FINAL_PAIR_BASE_COLS = [
    "pair_id",
    "audio_node_id",
    "image_node_id",
    "loc_id",
    "audio_source",
    "audio_id",
    "clip_index",
    "clip_start_ms",
    "clip_end_ms",
    "audio_url",
    "audio_path",
    "audio_lat",
    "audio_lon",
    "audio_ts",
    "image_source",
    "image_id",
    "image_url",
    "image_path",
    "image_lat",
    "image_lon",
    "image_ts",
    "ts_precision",
    "distance_m",
    "time_diff_days",
    "n_sub_clips",
    "embedding_similarity",
    "semantic_similarity",
    "aq_similarity",
    "aq_similarity_type",
    "filter_status",
    "filter_reason",
    "selected",
]

DEFAULT_IMAGE_MAX_EDGE = 1024

_LAT_COLS = ("latitude", "lat", "y")
_LON_COLS = ("longitude", "lon", "lng", "x")

SOUND_EVENT_CLASSES = [
    "Natural_sounds",
    "Music",
    "Animal",
    "Environment_and_background",
    "Water",
    "Outside_rural_or_natural",
    "Wind",
    "Human_sounds",
    "Traffic",
    "Noise",
    "Silence",
    "Vehicle",
    "Bird",
    "Sounds_of_things",
    "Speech",
]

ZERO_SHOT_EVENT_PROMPTS = {
    "Natural_sounds": [
        "an outdoor place where natural sounds are likely to be heard",
        "a natural outdoor soundscape",
    ],
    "Music": [
        "a place where music is likely to be heard",
        "an outdoor scene with music or performers",
    ],
    "Animal": [
        "a place where animal sounds are likely to be heard",
        "an outdoor scene with animals",
    ],
    "Environment_and_background": [
        "a place with ambient environmental background sound",
        "an outdoor scene with background ambience",
    ],
    "Water": [
        "a place where water sounds are likely to be heard",
        "a scene with a river lake fountain or water",
    ],
    "Outside_rural_or_natural": [
        "a rural or natural outdoor place",
        "an outdoor countryside or nature scene",
    ],
    "Wind": [
        "a place where wind sounds are likely to be heard",
        "an open outdoor scene with wind",
    ],
    "Human_sounds": [
        "a place where human sounds are likely to be heard",
        "an outdoor scene with people nearby",
    ],
    "Traffic": [
        "a place where traffic noise is likely to be heard",
        "a street scene with road traffic sounds",
    ],
    "Noise": [
        "a noisy outdoor urban place",
        "a place where loud noise is likely to be heard",
    ],
    "Silence": [
        "a quiet place with very little sound",
        "a silent calm outdoor scene",
    ],
    "Vehicle": [
        "a place where vehicle sounds are likely to be heard",
        "a scene with cars buses trucks or vehicles",
    ],
    "Bird": [
        "a place where birds are likely to be heard",
        "an outdoor scene with bird sounds",
    ],
    "Sounds_of_things": [
        "a place where objects and things make sounds",
        "an outdoor scene with machines tools or physical objects",
    ],
    "Speech": [
        "a place where speech is likely to be heard",
        "a scene with people talking",
    ],
}

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_captured_at(s: str) -> tuple[pd.Timestamp | None, str]:
    """Parse a *captured_at* string into ``(timestamp, ts_precision)``.

    Two formats are recognised:

    * **GSV**       ``"YYYY-MM-1-1"``  → day=1, hour=1 are nominal placeholders;
                    ``ts_precision = "year-month"``
    * **Mapillary** ``"YYYY-M-D-H"``   → full datetime;
                    ``ts_precision = "datetime"``

    All other formats are tried via :func:`pandas.Timestamp`; on failure
    ``(None, "unknown")`` is returned.
    """
    if not s or not isinstance(s, str):
        return None, "unknown"
    parts = s.split("-")
    if len(parts) == 4:
        try:
            year, month, day, hour = (int(p) for p in parts)
        except ValueError:
            return None, "unknown"
        # GSV sentinel: day == 1 and hour == 1 → year-month resolution only
        if day == 1 and hour == 1:
            ts = pd.Timestamp(year=year, month=month, day=1)
            return ts, "year-month"
        # Mapillary: full datetime
        ts = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        return ts, "datetime"
    # ISO-like fallback
    try:
        ts = pd.Timestamp(s)
        return ts, "datetime"
    except Exception:
        return None, "unknown"


def _infer_audio_source(url: str | None) -> str:
    """Infer the audio data provider from its URL."""
    if not url:
        return "unknown"
    u = str(url).lower()
    if "freesound" in u:
        return "freesound"
    if "aporee" in u:
        return "aporee"
    return "unknown"


def _infer_image_source(ts_precision: str) -> str:
    """Map ``ts_precision`` to a human-readable image provider name."""
    if ts_precision == "year-month":
        return "google"
    if ts_precision == "datetime":
        return "mapillary"
    return "unknown"


def _time_diff_days(
    audio_ts: pd.Timestamp | None,
    image_ts: pd.Timestamp | None,
    ts_precision: str,
) -> float:
    """Return |audio_ts − image_ts| in days, or NaN when not comparable.

    For ``ts_precision="year-month"`` the difference is computed in whole
    months (approximated as 30 days each) because GSV only stores year+month.
    """
    if audio_ts is None or image_ts is None:
        return float("nan")
    if ts_precision == "year-month":
        # Month-level difference only (GSV resolution)
        diff_months = abs(
            (audio_ts.to_period("M") - image_ts.to_period("M")).n
        )
        return float(diff_months * 30)
    # Normalise tz-awareness: strip tzinfo from whichever side has it so
    # pandas can subtract them.  Both sources use UTC implicitly, so this
    # is safe — we only care about the magnitude of the gap.
    a = audio_ts.tz_localize(None) if audio_ts.tzinfo is not None else audio_ts
    b = image_ts.tz_localize(None) if image_ts.tzinfo is not None else image_ts
    return float(abs((a - b).total_seconds()) / 86_400)


def _resolve_device(device: str | None) -> str:
    """Auto-detect the best available torch device."""
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_IMAGEBIND_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
_IMAGEBIND_WEIGHTS_FILENAME = "imagebind_huge.pth"
_IMAGEBIND_CACHE_DIR = Path.home() / ".cache" / "opensip"


def _load_imagebind(device: str):
    """Load the ImageBind model, downloading weights to ~/.cache/opensip/ if needed.

    Weights are downloaded once from Meta's CDN and cached locally.

    Parameters
    ----------
    device:
        Torch device string.
    """
    try:
        import torch
        from opensip.imagebind.models import imagebind_model
    except ImportError as exc:
        raise ImportError(
            "ImageBind similarity requires 'torch'.\n"
            "Install or repair the core package with: pip install openSIP"
        ) from exc

    dest = _IMAGEBIND_CACHE_DIR / _IMAGEBIND_WEIGHTS_FILENAME
    if not dest.exists():
        import requests as _req
        from tqdm.auto import tqdm as _tqdm
        _IMAGEBIND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        print(f"Downloading ImageBind weights to {dest} …")
        with _req.get(_IMAGEBIND_WEIGHTS_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            with open(part, "wb") as fh, _tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=_IMAGEBIND_WEIGHTS_FILENAME, ncols=80,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        part.rename(dest)

    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(str(dest), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def _ensure_local(
    url: str | None,
    dest_dir: Path,
    stem: str,
) -> Path | None:
    """Resolve *url* to a local :class:`~pathlib.Path`.

    Handles four cases:

    1. *url* is already a local path that exists → returned as-is.
    2. A file named ``<stem>.<any-ext>`` already exists in *dest_dir*
       (downloaded by a previous run) → returned as-is without re-downloading.
    3. ``http://`` / ``https://`` → downloaded to ``dest_dir/<stem>.<ext>``.
    4. ``data:<mime>;base64,…`` → decoded to ``dest_dir/<stem>.<ext>``.

    Returns ``None`` on any failure.
    """
    if not url:
        return None
    p = Path(url)
    if p.exists():
        return p
    # Already downloaded in dest_dir from a previous run?
    existing = sorted(dest_dir.glob(f"{stem}.*"))
    if existing:
        return existing[0]
    # base64 data-URI
    if url.startswith("data:"):
        import base64

        try:
            header, b64data = url.split(",", 1)
            ext = header.split("/")[1].split(";")[0] if "/" in header else "bin"
            raw = base64.b64decode(b64data)
            out = dest_dir / f"{stem}.{ext}"
            out.write_bytes(raw)
            return out
        except Exception:
            return None
    # HTTP(S)
    if url.startswith("http://") or url.startswith("https://"):
        import requests

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            ext_map = {
                "audio/mpeg": "mp3",
                "audio/wav": "wav",
                "audio/ogg": "ogg",
                "audio/flac": "flac",
                "audio/x-wav": "wav",
                "image/jpeg": "jpg",
                "image/png": "png",
                "image/webp": "webp",
            }
            ext = ext_map.get(ct.split(";")[0].strip(), "bin")
            out = dest_dir / f"{stem}.{ext}"
            out.write_bytes(resp.content)
            return out
        except Exception:
            return None
    return None


def _audio_duration(path: Path) -> float | None:
    """Return audio duration in seconds using *pydub*, or ``None`` on failure."""
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(str(path))
        return len(seg) / 1000.0
    except Exception:
        return None


def _split_audio_to_subclips(
    audio_path: Path,
    sub_clip_sec: float,
    start_ms: int | None,
    end_ms: int | None,
    work_dir: Path,
) -> list[Path]:
    """Split ``[start_ms, end_ms]`` of *audio_path* into *sub_clip_sec*-second chunks.

    The trailing partial chunk (if any) is discarded so every returned file
    is exactly *sub_clip_sec* long.  Returns an empty list when the segment is
    shorter than one sub-clip or on any error.
    """
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(str(audio_path))
        if start_ms is not None and end_ms is not None:
            seg = seg[start_ms:end_ms]
        total_ms = len(seg)
        sc_ms = int(sub_clip_sec * 1000)
        if total_ms < sc_ms:
            return []
        paths: list[Path] = []
        for idx, offset in enumerate(range(0, total_ms - sc_ms + 1, sc_ms)):
            chunk = seg[offset : offset + sc_ms]
            p = work_dir / f"sub_{idx:04d}.wav"
            chunk.export(str(p), format="wav")
            paths.append(p)
        return paths
    except Exception:
        return []


def _trim_audio_segment(
    audio_path: Path,
    start_ms: int | None,
    end_ms: int | None,
    out_path: Path,
) -> Path:
    """Write the requested audio segment to ``out_path``.

    If no segment bounds are provided, the original path is returned so callers
    can avoid an unnecessary transcode.
    """
    if start_ms is None or end_ms is None:
        return audio_path

    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(audio_path))
    seg = seg[start_ms:end_ms]
    seg.export(str(out_path), format="wav")
    return out_path


def _pick_time_unit(series: pd.Series, time_unit: str) -> tuple[str, float]:
    """Return ``(label, divisor)`` for the most readable time unit.

    *series* values are in **days**.  Dividing by *divisor* converts to the
    returned label's unit for display.
    """
    if time_unit != "auto":
        # All divisors expressed as days-per-display-unit so that
        # ``value_days / divisor = value_in_display_unit``.
        divisors: dict[str, float] = {
            "s":     1.0 / 86_400,   # 1 s  = 1/86400 d
            "min":   1.0 / 1_440,    # 1 min = 1/1440 d
            "h":     1.0 / 24,       # 1 h  = 1/24 d
            "d":     1.0,            # 1 d  = 1 d
            "month": 30.0,           # 1 mo ≈ 30 d
        }
        return time_unit, divisors.get(time_unit, 1.0)
    median = series.dropna().median()
    if math.isnan(median) or median < 2.0 / 1_440:   # < 2 min
        return "s", 1.0 / 86_400
    if median < 2.0 / 24:                             # < 2 h
        return "min", 1.0 / 1_440
    if median < 2.0:                                  # < 2 d
        return "h", 1.0 / 24
    if median < 90.0:                                 # < 90 d
        return "d", 1.0
    return "month", 30.0


def _stable_pair_id(audio_id, clip_index, image_id) -> str:
    return f"{audio_id}:{int(clip_index)}__{image_id}"


def _audio_node_id(audio_id, clip_index) -> str:
    return f"audio:{audio_id}:{int(clip_index)}"


def _image_node_id(image_id) -> str:
    return f"image:{image_id}"


def _first_existing(columns, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _finite_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    lon1_f = _finite_float(lon1)
    lat1_f = _finite_float(lat1)
    lon2_f = _finite_float(lon2)
    lat2_f = _finite_float(lat2)
    if None in (lon1_f, lat1_f, lon2_f, lat2_f):
        return float("nan")
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1_f)
    phi2 = math.radians(lat2_f)
    d_phi = math.radians(lat2_f - lat1_f)
    d_lam = math.radians(lon2_f - lon1_f)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius_m * math.asin(min(1.0, math.sqrt(a)))


def _coord_lookup_from_frame(
    frame: pd.DataFrame | None,
    *,
    id_col: str = "id",
) -> dict:
    if frame is None or getattr(frame, "empty", True) or id_col not in frame.columns:
        return {}
    lat_col = _first_existing(frame.columns, _LAT_COLS)
    lon_col = _first_existing(frame.columns, _LON_COLS)
    if lat_col is None or lon_col is None:
        return {}
    lookup = {}
    for row in frame[[id_col, lat_col, lon_col]].dropna(subset=[id_col]).itertuples(index=False):
        lookup[getattr(row, id_col)] = {
            "lat": getattr(row, lat_col),
            "lon": getattr(row, lon_col),
        }
    return lookup


def _coord_lookup_from_payload(payload: dict | None) -> dict:
    if not payload or "id" not in payload:
        return {}
    lat_key = _first_existing(payload.keys(), _LAT_COLS)
    lon_key = _first_existing(payload.keys(), _LON_COLS)
    if lat_key is None or lon_key is None:
        return {}
    ids = payload.get("id", [])
    lats = payload.get(lat_key, [])
    lons = payload.get(lon_key, [])
    return {
        item_id: {"lat": lat, "lon": lon}
        for item_id, lat, lon in zip(ids, lats, lons)
    }


def _coord_lookup_from_units(units) -> dict:
    if units is None or getattr(units, "empty", True) or "loc_id" not in units.columns:
        return {}
    try:
        frame = units.to_crs(4326) if getattr(units, "crs", None) else units
    except Exception:
        frame = units
    lookup = {}
    for row in frame[["loc_id", "geometry"]].itertuples(index=False):
        geom = getattr(row, "geometry", None)
        if geom is None or getattr(geom, "is_empty", False):
            continue
        pt = geom if getattr(geom, "geom_type", None) == "Point" else geom.centroid
        lookup[getattr(row, "loc_id")] = {"lat": pt.y, "lon": pt.x}
    return lookup


def _path_lookup_from_payload(payload: dict | None) -> dict:
    if not payload or "id" not in payload or "path" not in payload:
        return {}
    return {
        item_id: path
        for item_id, path in zip(payload.get("id", []), payload.get("path", []))
        if path
    }


def _graph_attr_value(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_ckpt(path: Path, record: dict) -> None:
    """Append one JSON object as a line to the JSONL checkpoint file *path*."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _mapping_get(mapping: dict | None, item_id, clip_index=None):
    """Lookup helper for id-keyed or ``(id, clip_index)``-keyed mappings."""
    if mapping is None:
        return None
    keys = []
    if clip_index is not None and not pd.isna(clip_index):
        try:
            ci = int(clip_index)
        except (TypeError, ValueError):
            ci = clip_index
        keys.extend([(item_id, ci), (str(item_id), ci)])
    keys.extend([item_id, str(item_id)])
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_rank_map(
    value,
    *,
    labels: list[str] | None = None,
    score_higher_is_better: bool = True,
    values_are_ranks: bool = False,
) -> dict[str, float] | None:
    """Convert a ranking/list/score-vector to ``{label: rank}``."""
    if value is None:
        return None
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if isinstance(value, (list, tuple)):
        return {str(label): float(i + 1) for i, label in enumerate(value)}
    if isinstance(value, dict):
        vals = {str(k): v for k, v in value.items() if v is not None and not pd.isna(v)}
        if not vals:
            return None
        if values_are_ranks:
            return {label: float(rank) for label, rank in vals.items()}
        ordered = sorted(vals.items(), key=lambda kv: float(kv[1]), reverse=score_higher_is_better)
        return {label: float(i + 1) for i, (label, _) in enumerate(ordered)}
    if labels is not None:
        try:
            arr = list(value)
        except TypeError:
            return None
        vals = {str(label): score for label, score in zip(labels, arr)}
        return _coerce_rank_map(
            vals,
            score_higher_is_better=score_higher_is_better,
            values_are_ranks=values_are_ranks,
        )
    return None


def _extract_prefixed_values(row: pd.Series, prefix: str) -> dict[str, float] | None:
    vals = {
        str(col)[len(prefix):]: row[col]
        for col in row.index
        if str(col).startswith(prefix) and pd.notna(row[col])
    }
    return vals or None


def _extract_rank_columns(row: pd.Series, prefix: str) -> list[str] | None:
    vals: list[tuple[int, str]] = []
    for col in row.index:
        col_s = str(col)
        if not col_s.startswith(prefix) or pd.isna(row[col]):
            continue
        try:
            rank = int(col_s[len(prefix):])
        except ValueError:
            continue
        vals.append((rank, str(row[col])))
    if not vals:
        return None
    return [label for _, label in sorted(vals)]


def _ndcg_at_k(reference_rank: dict[str, float], candidate_rank: dict[str, float], k: int) -> float:
    """NDCG@k using the reference top-k rank as graded relevance."""
    ref_top = sorted(reference_rank.items(), key=lambda kv: kv[1])[:k]
    relevance = {label: float(k - i) for i, (label, _) in enumerate(ref_top)}
    if not relevance:
        return float("nan")

    cand_order = [label for label, _ in sorted(candidate_rank.items(), key=lambda kv: kv[1])[:k]]

    def _dcg(labels: list[str]) -> float:
        return float(
            sum(relevance.get(label, 0.0) / math.log2(i + 2) for i, label in enumerate(labels))
        )

    dcg = _dcg(cand_order)
    ideal = _dcg([label for label, _ in ref_top])
    return float(dcg / ideal) if ideal > 0 else float("nan")


def _mrr(reference_rank: dict[str, float], candidate_rank: dict[str, float]) -> float:
    """Reciprocal rank of the reference top event in the candidate ranking."""
    if not reference_rank or not candidate_rank:
        return float("nan")
    top_event = min(reference_rank.items(), key=lambda kv: kv[1])[0]
    rank = candidate_rank.get(top_event)
    return float(1.0 / rank) if rank and rank > 0 else 0.0


def _spearman_from_ranks(a: dict[str, float], b: dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float("nan")
    av = pd.Series([a[x] for x in common], dtype="float64")
    bv = pd.Series([b[x] for x in common], dtype="float64")
    corr = av.corr(bv, method="pearson")
    return float(corr) if pd.notna(corr) else float("nan")


def _coerce_vector(value, labels: list[str] | None = None) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if isinstance(value, dict):
        vals = {str(k): float(v) for k, v in value.items() if v is not None and not pd.isna(v)}
        return vals or None
    if labels is not None:
        try:
            arr = list(value)
        except TypeError:
            return None
        vals = {
            str(label): float(v)
            for label, v in zip(labels, arr)
            if v is not None and not pd.isna(v)
        }
        return vals or None
    return None


def _pearson_vectors(a: dict[str, float], b: dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float("nan")
    av = pd.Series([a[x] for x in common], dtype="float64")
    bv = pd.Series([b[x] for x in common], dtype="float64")
    corr = av.corr(bv, method="pearson")
    return float(corr) if pd.notna(corr) else float("nan")


def _rmse_vectors(a: dict[str, float], b: dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan")
    diffs = np.array([a[x] - b[x] for x in common], dtype="float64")
    return float(np.sqrt(np.mean(diffs ** 2)))


def _coerce_event_aq_scores(event_aq_scores) -> dict[str, dict[str, float]] | None:
    if event_aq_scores is None:
        return None
    if isinstance(event_aq_scores, pd.DataFrame):
        df = event_aq_scores.copy()
        event_col = next((c for c in ("event", "sound_event", "label") if c in df.columns), None)
        if event_col is not None:
            df = df.set_index(event_col)
        return {
            str(idx): {
                str(col): float(val)
                for col, val in row.items()
                if pd.notna(val)
            }
            for idx, row in df.iterrows()
        }
    if isinstance(event_aq_scores, dict):
        return {
            str(event): _coerce_vector(vals) or {}
            for event, vals in event_aq_scores.items()
        }
    raise TypeError("event_aq_scores must be a DataFrame or dict.")


def _soundaqnet_to_event_mapping(df: pd.DataFrame, stem_to_key: dict[str, tuple]) -> dict:
    """Convert a SoundAQnet output DataFrame to pair audio-event mappings."""
    out: dict = {}
    for row in df.to_dict(orient="records"):
        clip_id = row.get("clip_id")
        key = stem_to_key.get(str(clip_id))
        if key is None:
            continue
        event_probs = row.get("event_probs")
        if isinstance(event_probs, str):
            try:
                event_probs = json.loads(event_probs)
            except json.JSONDecodeError:
                event_probs = None
        if isinstance(event_probs, dict) and event_probs:
            out[key] = event_probs
            continue
        top_events = row.get("top_events")
        if isinstance(top_events, str):
            try:
                top_events = json.loads(top_events)
            except json.JSONDecodeError:
                top_events = [e.strip() for e in top_events.split(",") if e.strip()]
        if isinstance(top_events, (list, tuple)) and top_events:
            out[key] = list(top_events)
    return out


def _soundaqnet_to_aq_mapping(
    df: pd.DataFrame,
    stem_to_key: dict[str, tuple],
    aq_dimensions: list[str],
) -> dict:
    """Convert flexible SoundAQnet output columns to audio AQ mappings."""
    out: dict = {}
    dict_cols = (
        "aq_scores",
        "affective_quality",
        "soundscape_quality",
        "soundscape_aq",
        "paq",
        "paq_scores",
    )
    prefixes = ("aq_", "audio_aq_", "soundscape_", "paq_")
    dim_lookup = {dim.lower(): dim for dim in aq_dimensions}

    for row in df.to_dict(orient="records"):
        clip_id = row.get("clip_id")
        key = stem_to_key.get(str(clip_id))
        if key is None:
            continue

        vec = None
        for col in dict_cols:
            val = row.get(col)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    val = None
            if isinstance(val, dict):
                vec = _coerce_vector(val, aq_dimensions)
                if vec:
                    break

        if vec is None:
            vals: dict[str, float] = {}
            for col, val in row.items():
                if isinstance(val, (dict, list, tuple)):
                    continue
                if pd.isna(val):
                    continue
                col_norm = str(col).lower()
                candidates = [col_norm]
                for prefix in prefixes:
                    if col_norm.startswith(prefix):
                        candidates.append(col_norm[len(prefix):])
                for cand in candidates:
                    if cand in dim_lookup:
                        vals[dim_lookup[cand]] = float(val)
                        break
            vec = vals or None

        if vec:
            out[key] = vec
    return out


def _image_event_ckpt_load(path: str | Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    mapping: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = rec.get("image_id")
            if image_id is None:
                continue
            if rec.get("ranking"):
                mapping[image_id] = rec["ranking"]
    return mapping


def _placepulse_ckpt_load(path: str | Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    direct: dict = {}
    view_rows: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = rec.get("image_id")
            scores = rec.get("scores")
            if image_id is not None and isinstance(scores, dict):
                if rec.get("view_id"):
                    view_rows.setdefault(image_id, []).append(scores)
                else:
                    direct[image_id] = scores
    for image_id, rows in view_rows.items():
        if image_id in direct:
            continue
        lively_vals = [r["lively"] for r in rows if "lively" in r]
        boring_vals = [r["boring"] for r in rows if "boring" in r]
        if lively_vals and boring_vals:
            direct[image_id] = {
                "lively": float(np.mean(lively_vals)),
                "boring": float(np.mean(boring_vals)),
                "n_views": int(len(rows)),
            }
    return direct


def _normalize_placepulse_scores(record) -> dict[str, float] | None:
    if record is None:
        return None
    if isinstance(record, pd.Series):
        record = record.to_dict()
    if isinstance(record, (list, tuple)) and len(record) >= 2:
        return {"lively": float(record[0]), "boring": float(record[1])}
    if not isinstance(record, dict):
        return None

    lowered = {str(k).lower(): v for k, v in record.items()}
    lively_keys = ("lively", "vibrant", "active", "eventful")
    boring_keys = ("boring", "monotonous", "dull", "inactive")
    lively = next((lowered[k] for k in lively_keys if k in lowered), None)
    boring = next((lowered[k] for k in boring_keys if k in lowered), None)
    if lively is None or boring is None:
        return None
    try:
        return {"lively": float(lively), "boring": float(boring)}
    except (TypeError, ValueError):
        return None


def _call_placepulse_predictor(placepulse_predictor, image_paths_by_id: dict) -> dict:
    """Call a user-supplied PlacePulse predictor and normalize its output."""
    if placepulse_predictor is None:
        raise ValueError("placepulse_predictor is required for PlacePulse image AQ inference.")
    if not image_paths_by_id:
        return {}

    image_ids = list(image_paths_by_id)
    image_paths = [str(image_paths_by_id[image_id]) for image_id in image_ids]

    if hasattr(placepulse_predictor, "predict_batch"):
        raw = placepulse_predictor.predict_batch(image_paths)
    elif hasattr(placepulse_predictor, "predict"):
        raw = placepulse_predictor.predict(image_paths)
    elif callable(placepulse_predictor):
        try:
            raw = placepulse_predictor(image_paths_by_id)
        except TypeError:
            raw = placepulse_predictor(image_paths)
    else:
        raise TypeError(
            "placepulse_predictor must be callable or expose predict()/predict_batch()."
        )

    if isinstance(raw, pd.DataFrame):
        records = raw.to_dict(orient="records")
        raw = records

    reverse_by_path = {str(path): image_id for image_id, path in image_paths_by_id.items()}
    out: dict = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            image_id = key if key in image_paths_by_id else reverse_by_path.get(str(key))
            if image_id is None and isinstance(value, dict):
                candidate_id = value.get("image_id") or value.get("id")
                if candidate_id in image_paths_by_id:
                    image_id = candidate_id
            scores = _normalize_placepulse_scores(value)
            if image_id is not None and scores:
                out[image_id] = scores
        return out

    if isinstance(raw, (list, tuple)):
        for image_id, value in zip(image_ids, raw):
            if isinstance(value, dict):
                candidate_id = value.get("image_id") or value.get("id")
                if candidate_id in image_paths_by_id:
                    image_id = candidate_id
            scores = _normalize_placepulse_scores(value)
            if scores:
                out[image_id] = scores
        return out

    raise TypeError(
        "placepulse_predictor must return a dict, list/tuple, or pandas DataFrame."
    )


def _event_prompts_for_classes(
    event_classes: list[str],
    prompt_templates: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    prompt_templates = prompt_templates or ZERO_SHOT_EVENT_PROMPTS
    prompts: list[str] = []
    owners: list[str] = []
    for event in event_classes:
        event_prompts = prompt_templates.get(
            event,
            [
                f"a place where {event.replace('_', ' ').lower()} can be heard",
                f"an outdoor scene associated with {event.replace('_', ' ').lower()} sounds",
            ],
        )
        for prompt in event_prompts:
            prompts.append(prompt)
            owners.append(event)
    return prompts, owners


def _rank_events_from_scores(
    scores: np.ndarray,
    event_classes: list[str],
    prompt_owners: list[str],
) -> tuple[list[str], dict[str, float]]:
    event_scores: dict[str, list[float]] = {event: [] for event in event_classes}
    for score, event in zip(scores.tolist(), prompt_owners):
        event_scores[event].append(float(score))
    averaged = {
        event: float(np.mean(vals)) if vals else float("-inf")
        for event, vals in event_scores.items()
    }
    ranking = sorted(event_classes, key=lambda event: averaged[event], reverse=True)
    return ranking, averaged


def _prepare_event_rank_image(
    src: Path,
    dest_dir: Path,
    stem: str,
    max_image_edge: int | None,
) -> Path | None:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImportError("Image event inference requires Pillow.") from exc

    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(src) as img:
            img.load()
            if max_image_edge is not None and max(img.width, img.height) > max_image_edge:
                img.thumbnail((max_image_edge, max_image_edge), Image.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")
            dest = dest_dir / f"{stem}.jpg"
            img.save(dest, format="JPEG", quality=92)
            return dest
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _is_probable_panorama(path: Path, min_aspect_ratio: float = 1.8) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
        return h > 0 and (w / h) >= min_aspect_ratio
    except Exception:
        return False


def _prepare_placepulse_images(
    src: Path,
    dest_dir: Path,
    stem: str,
    *,
    max_image_edge: int | None,
    panorama_strategy: str,
    panorama_headings: list[float],
    panorama_fov: float,
    panorama_phi: float,
    panorama_size: tuple[int, int],
    panorama_min_aspect_ratio: float,
) -> list[Path]:
    if panorama_strategy not in {"auto", "none", "perspective"}:
        raise ValueError("panorama_strategy must be 'auto', 'none', or 'perspective'.")

    should_project = panorama_strategy == "perspective" or (
        panorama_strategy == "auto"
        and _is_probable_panorama(src, min_aspect_ratio=panorama_min_aspect_ratio)
    )
    if not should_project:
        prepared = _prepare_event_rank_image(src, dest_dir, stem, max_image_edge)
        return [prepared] if prepared is not None else []

    try:
        from opensip.utils.pano2pers import Equirectangular
    except ImportError as exc:
        raise ImportError("Panorama PlacePulse preprocessing requires OpenCV.") from exc

    try:
        equ = Equirectangular(img_path=str(src))
    except Exception:
        prepared = _prepare_event_rank_image(src, dest_dir, stem, max_image_edge)
        return [prepared] if prepared is not None else []

    paths: list[Path] = []
    width, height = panorama_size
    for idx, heading in enumerate(panorama_headings):
        try:
            b64 = equ.GetPerspective(
                FOV=float(panorama_fov),
                THETA=float(heading),
                PHI=float(panorama_phi),
                height=int(height),
                width=int(width),
            )
            out = dest_dir / f"{stem}_view{idx:02d}.png"
            out.write_bytes(base64.b64decode(b64))
            prepared = _prepare_event_rank_image(
                out,
                dest_dir,
                f"{stem}_view{idx:02d}_prepared",
                max_image_edge,
            )
            if prepared is not None:
                paths.append(prepared)
        except Exception:
            continue
    return paths


def _zeroshot_image_event_ranks(
    image_paths_by_id: dict,
    event_classes: list[str],
    *,
    prompt_templates: dict[str, list[str]] | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
    max_image_pixels: int | None = None,
) -> tuple[dict, dict]:
    """Rank SoundAQnet events for images with ImageBind vision-text scores."""
    if not image_paths_by_id:
        return {}, {}

    if max_image_pixels is not None:
        max_image_edge = max_image_pixels

    import torch
    import torch.nn.functional as F

    from opensip.imagebind import data as ib_data
    from opensip.imagebind.models.imagebind_model import ModalityType

    device_ = _resolve_device(device)
    model = _load_imagebind(device_)

    prompts, prompt_owners = _event_prompts_for_classes(event_classes, prompt_templates)
    text_inp = {
        ModalityType.TEXT: ib_data.load_and_transform_text_data(prompts, device_)
    }
    with torch.no_grad():
        text_emb = model(text_inp)[ModalityType.TEXT]
    text_emb = F.normalize(text_emb, p=2, dim=-1)

    ranks: dict = {}
    score_maps: dict = {}
    items = list(image_paths_by_id.items())
    for offset in range(0, len(items), batch_size):
        chunk = items[offset : offset + batch_size]
        image_ids = [image_id for image_id, _ in chunk]
        image_paths = [str(path) for _, path in chunk]
        img_inp = {
            ModalityType.VISION: ib_data.load_and_transform_vision_data(
                image_paths,
                device_,
                max_image_edge=max_image_edge,
            )
        }
        with torch.no_grad():
            image_emb = model(img_inp)[ModalityType.VISION]
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        score_matrix = (image_emb @ text_emb.T).detach().cpu().numpy()
        for image_id, scores in zip(image_ids, score_matrix):
            ranking, averaged = _rank_events_from_scores(scores, event_classes, prompt_owners)
            ranks[image_id] = ranking
            score_maps[image_id] = averaged
    return ranks, score_maps


def _aq_from_event_rank(
    rank_map: dict[str, float] | None,
    event_aq_scores: dict[str, dict[str, float]] | None,
    top_k: int,
) -> dict[str, float] | None:
    if not rank_map or not event_aq_scores:
        return None
    events = [label for label, _ in sorted(rank_map.items(), key=lambda kv: kv[1])[:top_k]]
    rows = [event_aq_scores[event] for event in events if event in event_aq_scores]
    if not rows:
        return None
    dims = sorted(set().union(*(row.keys() for row in rows)))
    out: dict[str, float] = {}
    for dim in dims:
        vals = [row[dim] for row in rows if dim in row and pd.notna(row[dim])]
        if vals:
            out[dim] = float(np.mean(vals))
    return out or None


# ---------------------------------------------------------------------------
# AudioImagePairDataset
# ---------------------------------------------------------------------------


class AudioImagePairDataset:
    """Build, score, filter, and export audio–image pairs.

    The class owns a :class:`~opensip.dataset.GeoTaggedData` object and delegates
    collection methods/attributes to it, so the same object can collect sounds,
    collect images, build candidate pairs, score them, and export results.
    Passing an existing ``GeoTaggedData`` instance is still supported for
    backward compatibility.

    The typical pipeline is::

        apd = AudioImagePairDataset(locations=[[-83.073, 42.348]])
        apd.get_sound_from_location(source="aporee", distance=100)
        apd.get_svi_from_locations(source="google", key="...")
        apd.build_pairs(image_source="svi", time_window=None)
        apd.compute_embedding_similarity(output_dir="pairs/audio")
        apd.filter_pairs_by_similarity(min_similarity=0.20, top_k_per_audio=1)
        apd.export("pairs/")
        apd.plot()

    Parameters
    ----------
    gtd:
        Optional populated :class:`~opensip.dataset.GeoTaggedData` instance.
    locations, units:
        Forwarded to :class:`~opensip.dataset.GeoTaggedData` when ``gtd`` is not
        provided.
    """

    _DELEGATED_GTD_ATTRS = {
        "images",
        "locations",
        "units",
        "svis",
        "photos",
        "audios",
        "svi_metadata",
        "photo_metadata",
        "audio_metadata",
        "plot",
    }

    def __init__(
        self,
        gtd: "GeoTaggedData | None" = None,
        *,
        locations: list | tuple | dict | pd.DataFrame | None = None,
        units=None,
    ) -> None:
        if gtd is not None and (locations is not None or units is not None):
            raise ValueError("Pass either gtd=... or locations=/units=..., not both.")
        if gtd is None:
            from .dataset import GeoTaggedData

            gtd = GeoTaggedData(locations=locations, units=units)
        object.__setattr__(self, "_gtd", gtd)
        self.pairs: pd.DataFrame = pd.DataFrame(columns=_PAIR_COLS)
        self._soundaqnet_cache: dict = {}

    @property
    def gtd(self) -> "GeoTaggedData":
        """Underlying :class:`~opensip.dataset.GeoTaggedData` state object."""
        return self._gtd

    def __getattr__(self, name: str):
        """Delegate GeoTaggedData methods and attributes to ``self._gtd``."""
        return getattr(self._gtd, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._DELEGATED_GTD_ATTRS and "_gtd" in self.__dict__:
            setattr(self._gtd, name, value)
            return
        object.__setattr__(self, name, value)

    def _predict_soundaqnet(
        self,
        *,
        model_name: str | Path,
        device: str | None,
        batch_size: int,
        num_workers: int,
        output_dir: str | Path | None,
        cache: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, tuple]]:
        """Run SoundAQnet once for unique pair audio clips and cache outputs."""
        audio_signature = tuple(
            (
                row["audio_id"],
                int(row["clip_index"]),
                str(row.get("audio_url")),
            )
            for _, row in self.pairs[["audio_id", "clip_index", "audio_url"]]
            .drop_duplicates(subset=["audio_id", "clip_index"])
            .iterrows()
        )
        cache_key = (
            str(model_name),
            str(device),
            int(batch_size),
            int(num_workers),
            audio_signature,
        )
        if cache and cache_key in self._soundaqnet_cache:
            cached_df, cached_mapping = self._soundaqnet_cache[cache_key]
            return cached_df.copy(), dict(cached_mapping)

        try:
            from soundaqnet import SoundAQnet
        except ImportError as exc:
            raise ImportError(
                "SoundAQnet inference requires the soundaqnet package, which is "
                "part of openSIP's core dependencies.\n"
                "Install or repair the core package with: pip install openSIP"
            ) from exc

        _tmp_dir: tempfile.TemporaryDirectory | None = None
        if output_dir is None:
            _tmp_dir = tempfile.TemporaryDirectory(prefix="opensip_soundaqnet_")
            work_dir = Path(_tmp_dir.name)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
        soundaq_audio_dir = work_dir / "soundaqnet_audio"
        soundaq_audio_dir.mkdir(exist_ok=True)

        try:
            audio_files: list[str] = []
            stem_to_key: dict[str, tuple] = {}
            seen_audio: set[tuple] = set()
            for _, row in self.pairs.iterrows():
                key = (row["audio_id"], int(row["clip_index"]))
                if key in seen_audio:
                    continue
                seen_audio.add(key)
                local = _ensure_local(
                    row.get("audio_url"),
                    soundaq_audio_dir,
                    f"soundaq_{row['audio_id']}_{int(row['clip_index'])}",
                )
                if local is None:
                    continue
                audio_files.append(str(local))
                stem_to_key[Path(local).stem] = key

            if audio_files:
                model = SoundAQnet(model=str(model_name), device=device)
                soundaq_df = model.predict_from_audio(
                    audio_files=audio_files,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    show_progress=False,
                )
            else:
                soundaq_df = pd.DataFrame()

            if cache:
                self._soundaqnet_cache[cache_key] = (soundaq_df.copy(), dict(stem_to_key))
            return soundaq_df, stem_to_key
        finally:
            if _tmp_dir is not None:
                _tmp_dir.cleanup()

    # ------------------------------------------------------------------
    # build_pairs
    # ------------------------------------------------------------------

    def build_pairs(
        self,
        image_source: str = "svi",
        time_window: float | None = None,
    ) -> pd.DataFrame:
        """Pair every audio clip with co-located images.

        No media files are downloaded.  Only URLs and IDs are collected.
        ``similarity`` and ``n_sub_clips`` are left as ``NaN`` until
        :meth:`compute_embedding_similarity` is called.

        Parameters
        ----------
        image_source:
            ``"svi"``   — use ``gtd.svi_metadata`` (Google Street View /
            Mapillary images).
            ``"photo"`` — use ``gtd.photo_metadata`` (Flickr / other photos).
        time_window:
            Maximum absolute time difference in **days** between the audio
            recording and the image capture.  Pairs that exceed this threshold
            are discarded before similarity is computed.  ``None`` keeps all
            pairs.

        Returns
        -------
        pandas.DataFrame
            The pairs metadata table (also stored in ``self.pairs``).
        """
        gtd = self._gtd
        audios = gtd.audios  # dict of parallel lists: id, loc_id, data, path, slice?

        if not audios or not audios.get("id"):
            raise ValueError("gtd.audios is empty — collect audio data first.")

        n_clips = len(audios["id"])

        # select image metadata
        if image_source == "svi":
            img_meta = getattr(gtd, "svi_metadata", None)
            if img_meta is None or (hasattr(img_meta, "empty") and img_meta.empty):
                raise ValueError(
                    "gtd.svi_metadata is empty — collect street-view data first."
                )
            img_id_col = "id"
            img_ts_col = "captured_at"
            img_loc_col = "loc_id"
            # prefer 'image_url' (download pipeline) then 'url' (raw API);
            # fall back to id so the column is always populated.
            _svi_url_candidates = [c for c in ("image_url", "url") if c in img_meta.columns]
            img_url_col = _svi_url_candidates[0] if _svi_url_candidates else img_id_col
        elif image_source == "photo":
            img_meta = getattr(gtd, "photo_metadata", None)
            if img_meta is None or (hasattr(img_meta, "empty") and img_meta.empty):
                raise ValueError(
                    "gtd.photo_metadata is empty — collect photo data first."
                )
            img_id_col = "id"
            img_ts_col = "datetaken"
            img_loc_col = "loc_id" if "loc_id" in img_meta.columns else None
            _url_candidates = [c for c in ("url", "image_url", "photo_url") if c in img_meta.columns]
            img_url_col = _url_candidates[0] if _url_candidates else None
        else:
            raise ValueError(
                f"Unknown image_source={image_source!r}. Use 'svi' or 'photo'."
            )

        # audio metadata (timestamps)
        # Try common timestamp column names produced by each source:
        #   Aporee / Freesound: "created"
        #   enriched catalogs:  "created_dt", "capture_time"
        _AUD_TS_COLS = ("created_dt", "created", "capture_time", "captured_at")
        aud_meta = getattr(gtd, "audio_metadata", None)
        aud_ts_by_id: dict[str, pd.Timestamp | None] = {}
        if aud_meta is not None and not aud_meta.empty:
            _aud_ts_col = next(
                (c for c in _AUD_TS_COLS if c in aud_meta.columns),
                None,
            )
            for row in aud_meta.itertuples(index=False):
                raw = getattr(row, _aud_ts_col) if _aud_ts_col else None
                ts: pd.Timestamp | None = None
                if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                    try:
                        ts = pd.Timestamp(raw)
                    except Exception:
                        pass
                aud_ts_by_id[row.id] = ts

        # build per-loc_id image index
        imgs_by_loc: dict[str, list[dict]] = {}
        for row in img_meta.itertuples(index=False):
            lid = (
                getattr(row, img_loc_col)
                if img_loc_col and hasattr(row, img_loc_col)
                else "global"
            )
            raw_ts = getattr(row, img_ts_col, None)
            img_ts, ts_prec = (
                _parse_captured_at(str(raw_ts)) if raw_ts else (None, "unknown")
            )
            img_url = getattr(row, img_url_col) if img_url_col else None
            imgs_by_loc.setdefault(lid, []).append(
                {
                    "image_id": getattr(row, img_id_col),
                    "image_url": img_url,
                    "image_ts": img_ts,
                    "ts_precision": ts_prec,
                }
            )

        # build pairs
        clip_counters: dict[tuple, int] = {}
        rows: list[dict] = []

        for i in range(n_clips):
            a_id = audios["id"][i]
            a_loc = audios.get("loc_id", [None] * n_clips)[i]
            # 'data' field holds the URL/base64 payload; 'path' holds local path
            a_url = audios.get("data", [None] * n_clips)[i]
            a_path = audios.get("path", [None] * n_clips)[i]
            # Prefer the local path when the file is already on disk — avoids
            # redundant HTTP downloads on every embedding-similarity call.
            if a_path and Path(a_path).exists():
                a_url = a_path
            else:
                a_url = a_url or a_path

            a_slice = audios.get("slice", [None] * n_clips)[i]
            start_ms: int | None = int(a_slice[0]) if a_slice else None
            end_ms: int | None = int(a_slice[1]) if a_slice else None

            key = (a_loc, a_id)
            clip_idx = clip_counters.get(key, 0)
            clip_counters[key] = clip_idx + 1

            aud_ts = aud_ts_by_id.get(a_id)
            aud_source = _infer_audio_source(a_url)

            for img in imgs_by_loc.get(a_loc, []):
                diff = _time_diff_days(aud_ts, img["image_ts"], img["ts_precision"])
                if (
                    time_window is not None
                    and not math.isnan(diff)
                    and diff > time_window
                ):
                    continue
                rows.append(
                    {
                        "loc_id": a_loc,
                        "audio_source": aud_source,
                        "audio_id": a_id,
                        "clip_index": clip_idx,
                        "clip_start_ms": start_ms,
                        "clip_end_ms": end_ms,
                        "audio_url": a_url,
                        "audio_ts": aud_ts,
                        "image_source": _infer_image_source(img["ts_precision"]),
                        "image_id": img["image_id"],
                        "image_url": img["image_url"],
                        "image_ts": img["image_ts"],
                        "ts_precision": img["ts_precision"],
                        "time_diff_days": diff,
                        "n_sub_clips": float("nan"),
                        "similarity": float("nan"),
                    }
                )

        self.pairs = (
            pd.DataFrame(rows, columns=_PAIR_COLS)
            if rows
            else pd.DataFrame(columns=_PAIR_COLS)
        )
        return self.pairs

    # ------------------------------------------------------------------
    # compute_embedding_similarity
    # ------------------------------------------------------------------

    def compute_embedding_similarity(
        self,
        audio_strategy: str = "sample",
        sub_clip_sec: float = 2.0,
        clips_per_audio: int = 3,
        min_duration_sec: float = 2.0,
        batch_size: int = 32,
        device: str | None = None,
        output_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        max_image_pixels: int | None = None,
    ) -> pd.DataFrame:
        """Compute ImageBind cosine similarity for every pair.

        For each unique ``(audio_id, clip_index)`` the audio segment is embedded
        with one of two strategies:

        - ``"sample"`` lets ImageBind sample ``clips_per_audio`` windows of
          length ``sub_clip_sec`` from the segment and average them internally.
          This is faster and is the default.
        - ``"exhaustive"`` splits the whole segment into consecutive
          ``sub_clip_sec`` windows and averages all returned embeddings. This
          gives fuller temporal coverage for longer recordings.

        Aporee clips that do not have a ``slice`` entry have their actual
        duration verified from the downloaded file; clips shorter than
        *min_duration_sec* are skipped entirely.

        Parameters
        ----------
        audio_strategy:
            ``"sample"`` for fast ImageBind internal sampling, or
            ``"exhaustive"`` for full consecutive-window coverage.
        sub_clip_sec:
            Duration (seconds) of each ImageBind audio window. ImageBind was
            trained on 2-second clips.
        clips_per_audio:
            Number of windows ImageBind samples in ``audio_strategy="sample"``
            mode. Ignored in ``"exhaustive"`` mode.
        min_duration_sec:
            Minimum effective clip duration required.  Clips shorter than
            this (after trimming to ``[clip_start_ms, clip_end_ms]``) are
            dropped and their pairs receive ``NaN`` similarity.
        batch_size:
            Number of images embedded in a single forward pass.
        device:
            ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-detected when ``None``.
        output_dir:
            Directory for downloaded audio/image files.  When ``None`` a
            temporary directory is created and cleaned up automatically.
        checkpoint_path:
            Optional JSONL file for resuming interrupted runs.  Triples
            already recorded as ``(audio_id, clip_index, image_id)`` are
            skipped and loaded from this file.
        max_image_edge:
            Pre-resize images whose longer edge exceeds this pixel count before
            passing them to ImageBind. ImageBind's vision transform ultimately
            resizes/crops to 224 px, so pre-scaling large street-view panoramas
            and Flickr images reduces memory and CPU time without changing the
            model input size. Set to ``None`` to disable pre-resizing. Default
            is ``1024``.
        max_image_pixels:
            Deprecated alias for ``max_image_edge``.

        Returns
        -------
        pandas.DataFrame
            ``self.pairs`` with ``similarity`` and ``n_sub_clips`` filled in.
        """
        if self.pairs.empty:
            raise RuntimeError("Call build_pairs() first.")
        if audio_strategy not in {"sample", "exhaustive"}:
            raise ValueError("audio_strategy must be 'sample' or 'exhaustive'.")
        if clips_per_audio < 1:
            raise ValueError("clips_per_audio must be >= 1.")
        if max_image_pixels is not None:
            max_image_edge = max_image_pixels

        import logging as _logging

        import torch
        import torch.nn.functional as F
        from tqdm.auto import tqdm as _tqdm

        _logger = _logging.getLogger(__name__)

        device_ = _resolve_device(device)
        model = _load_imagebind(device_)

        from opensip.imagebind import data as ib_data
        from opensip.imagebind.models.imagebind_model import ModalityType

        # ── pre-flight: verify torchaudio is actually loadable ────────────────
        # (it may have been compiled against a different torch ABI)
        try:
            import torchaudio as _ta_check  # noqa: F401
        except OSError as exc:
            raise OSError(
                "torchaudio failed to load — audio embedding is impossible.\n"
                "Fix: pip install torchaudio==$(python -c 'import torch; print(torch.__version__)')\n"
                f"Original error: {exc}"
            ) from exc

        # checkpoint (resume support)
        ckpt_path: Path | None = Path(checkpoint_path) if checkpoint_path else None
        done: dict[tuple, tuple[float, int]] = {}
        if ckpt_path and ckpt_path.exists():
            with open(ckpt_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        key = (rec["audio_id"], int(rec["clip_index"]), rec["image_id"])
                        rec_strategy = rec.get("audio_strategy", audio_strategy)
                        rec_sub_clip = float(rec.get("sub_clip_sec", sub_clip_sec))
                        rec_clips_per_audio = int(
                            rec.get("clips_per_audio", clips_per_audio)
                        )
                        if (
                            rec_strategy != audio_strategy
                            or abs(rec_sub_clip - float(sub_clip_sec)) > 1e-9
                            or (
                                audio_strategy == "sample"
                                and rec_clips_per_audio != int(clips_per_audio)
                            )
                        ):
                            continue
                        done[key] = (float(rec["similarity"]), int(rec["n_sub_clips"]))
                    except Exception:
                        pass
            if done:
                print(f"Resuming from checkpoint — {len(done)} pair(s) already scored.")

        # storage setup
        _tmp_dir: tempfile.TemporaryDirectory | None = None
        if output_dir is None:
            _tmp_dir = tempfile.TemporaryDirectory(prefix="opensip_pairs_")
            work_dir = Path(_tmp_dir.name)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = work_dir / "audio"
        image_dir = work_dir / "images"
        subclip_dir = work_dir / "_subclips"
        for d in (audio_dir, image_dir, subclip_dir):
            d.mkdir(exist_ok=True)

        try:
            pairs = self.pairs.copy()
            sim_updates: dict[int, float] = {}
            nsub_updates: dict[int, int] = {}

            # Counters for the progress summary
            n_audio_groups = pairs.groupby(["audio_id", "clip_index"]).ngroups
            n_scored = 0
            n_skipped_download = 0
            n_skipped_short = 0
            n_skipped_embed = 0

            # Group by unique audio clip to avoid redundant downloads
            groups = list(pairs.groupby(["audio_id", "clip_index"], sort=False))
            for (a_id, clip_idx), grp in _tqdm(
                groups, desc="Scoring pairs", unit="clip"
            ):
                first = grp.iloc[0]
                start_ms = (
                    int(first["clip_start_ms"])
                    if pd.notna(first["clip_start_ms"])
                    else None
                )
                end_ms = (
                    int(first["clip_end_ms"])
                    if pd.notna(first["clip_end_ms"])
                    else None
                )
                a_url = first["audio_url"]

                # download audio
                a_local = _ensure_local(a_url, audio_dir, f"aud_{a_id}_{clip_idx}")
                if a_local is None:
                    _logger.warning("Audio download failed for id=%s url=%s", a_id, a_url)
                    n_skipped_download += 1
                    continue

                # Back-fill gtd.audios["path"] so save() captures the local
                # path and future loads skip the download entirely.
                _aud_ids = self._gtd.audios.get("id", [])
                try:
                    _aud_idx = _aud_ids.index(a_id)
                    _paths = self._gtd.audios.setdefault("path", [None] * len(_aud_ids))
                    if _aud_idx < len(_paths) and not _paths[_aud_idx]:
                        _paths[_aud_idx] = str(a_local)
                except (ValueError, IndexError, AttributeError):
                    pass

                # post-download duration check (important for Aporee)
                total_dur = _audio_duration(a_local)
                if total_dur is None:
                    _logger.warning("Could not read audio duration for %s", a_local)
                    n_skipped_download += 1
                    continue
                clip_dur = (
                    (end_ms - start_ms) / 1000.0
                    if (start_ms is not None and end_ms is not None)
                    else total_dur
                )
                if clip_dur < min_duration_sec:
                    n_skipped_short += 1
                    continue

                # ── audio embedding ───────────────────────────────────────
                sc_dir = subclip_dir / f"{a_id}_{clip_idx}"
                sc_dir.mkdir(exist_ok=True)
                audio_embs: list = []
                n_sub = 0

                if audio_strategy == "sample":
                    try:
                        segment_path = _trim_audio_segment(
                            a_local,
                            start_ms,
                            end_ms,
                            sc_dir / "segment.wav",
                        )
                        inp = {
                            ModalityType.AUDIO: ib_data.load_and_transform_audio_data(
                                [str(segment_path)],
                                device_,
                                clip_duration=sub_clip_sec,
                                clips_per_video=clips_per_audio,
                            )
                        }
                        with torch.no_grad():
                            audio_emb = model(inp)[ModalityType.AUDIO]  # (1, D)
                        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
                        n_sub = 1 if clip_dur <= sub_clip_sec else clips_per_audio
                    except (ImportError, OSError):
                        raise
                    except Exception as exc:
                        _logger.warning(
                            "Audio embedding failed for audio id=%s: %s: %s",
                            a_id, type(exc).__name__, exc,
                        )
                        n_skipped_embed += 1
                        continue
                else:
                    sub_paths = _split_audio_to_subclips(
                        a_local, sub_clip_sec, start_ms, end_ms, sc_dir
                    )
                    if not sub_paths:
                        _logger.warning(
                            "No sub-clips generated for audio id=%s (clip_dur=%.1fs, "
                            "sub_clip_sec=%.1f)",
                            a_id, clip_dur, sub_clip_sec,
                        )
                        n_skipped_short += 1
                        continue
                    n_sub = len(sub_paths)

                    for sc_p in sub_paths:
                        try:
                            inp = {
                                ModalityType.AUDIO: ib_data.load_and_transform_audio_data(
                                    [str(sc_p)],
                                    device_,
                                    clip_duration=sub_clip_sec,
                                    clips_per_video=1,
                                )
                            }
                            with torch.no_grad():
                                emb = model(inp)[ModalityType.AUDIO]  # (1, D)
                            audio_embs.append(emb)
                        except (ImportError, OSError):
                            # Fatal configuration errors — re-raise immediately
                            raise
                        except Exception as exc:
                            _logger.warning(
                                "Audio embedding failed for sub-clip %s: %s: %s",
                                sc_p, type(exc).__name__, exc,
                            )
                            continue

                    if not audio_embs:
                        _logger.warning(
                            "All sub-clips failed to embed for audio id=%s", a_id
                        )
                        n_skipped_embed += 1
                        continue

                    audio_emb = torch.cat(audio_embs, dim=0).mean(dim=0, keepdim=True)
                    audio_emb = F.normalize(audio_emb, p=2, dim=-1)  # (1, D)

                # image embeddings in batches
                img_ids = grp["image_id"].tolist()
                img_urls = grp["image_url"].tolist()
                row_idx = grp.index.tolist()

                for b0 in range(0, len(img_ids), batch_size):
                    b_ids = img_ids[b0 : b0 + batch_size]
                    b_urls = img_urls[b0 : b0 + batch_size]
                    b_ridx = row_idx[b0 : b0 + batch_size]

                    img_paths: list[str] = []
                    live_ridx: list[int] = []

                    for iid, iurl, ridx in zip(b_ids, b_urls, b_ridx):
                        ck = (a_id, int(clip_idx), iid)
                        if ck in done:
                            sim_updates[ridx] = done[ck][0]
                            nsub_updates[ridx] = done[ck][1]
                            n_scored += 1
                            continue
                        p = _ensure_local(iurl, image_dir, f"img_{iid}")
                        if p is None:
                            _logger.debug("Image download failed for id=%s url=%s", iid, iurl)
                            continue
                        img_paths.append(str(p))
                        live_ridx.append(ridx)

                    if not img_paths:
                        continue

                    try:
                        inp = {
                            ModalityType.VISION: ib_data.load_and_transform_vision_data(
                                img_paths, device_, max_image_edge=max_image_edge
                            )
                        }
                        with torch.no_grad():
                            img_embs = model(inp)[ModalityType.VISION]  # (B, D)
                        img_embs = F.normalize(img_embs, p=2, dim=-1)
                        sims = (audio_emb @ img_embs.T).squeeze(0).tolist()  # (B,)
                    except (ImportError, OSError):
                        raise
                    except Exception as exc:
                        _logger.warning(
                            "Image embedding failed for batch (audio_id=%s): %s", a_id, exc
                        )
                        continue

                    for ridx, sim_val in zip(live_ridx, sims):
                        sim_updates[ridx] = float(sim_val)
                        nsub_updates[ridx] = n_sub
                        n_scored += 1
                        if ckpt_path:
                            _append_ckpt(
                                ckpt_path,
                                {
                                    "audio_id": a_id,
                                    "clip_index": int(clip_idx),
                                    "image_id": pairs.at[ridx, "image_id"],
                                    "similarity": float(sim_val),
                                    "n_sub_clips": n_sub,
                                    "audio_strategy": audio_strategy,
                                    "sub_clip_sec": float(sub_clip_sec),
                                    "clips_per_audio": int(clips_per_audio),
                                },
                            )

            # Apply updates back to the DataFrame
            for ridx, val in sim_updates.items():
                pairs.at[ridx, "similarity"] = val
            for ridx, val in nsub_updates.items():
                pairs.at[ridx, "n_sub_clips"] = val

            total_pairs = len(pairs)
            print(
                f"Scored: {n_scored} / {total_pairs} pairs  "
                f"(skipped — download: {n_skipped_download}, "
                f"too short: {n_skipped_short}, "
                f"embed error: {n_skipped_embed})"
            )
            if n_scored == 0 and total_pairs > 0:
                print(
                    "  Tip: run with Python logging at DEBUG level to see per-clip "
                    "failure reasons:\n"
                    "    import logging; logging.basicConfig(level=logging.DEBUG)"
                )

            self.pairs = pairs
            return self.pairs

        finally:
            # Sub-clips are always temp — remove regardless of output_dir
            shutil.rmtree(str(subclip_dir), ignore_errors=True)
            if _tmp_dir is not None:
                _tmp_dir.cleanup()

    def compute_similarity(
        self,
        audio_strategy: str = "sample",
        sub_clip_sec: float = 2.0,
        clips_per_audio: int = 3,
        min_duration_sec: float = 2.0,
        batch_size: int = 32,
        device: str | None = None,
        output_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        max_image_pixels: int | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible alias for :meth:`compute_embedding_similarity`."""
        return self.compute_embedding_similarity(
            audio_strategy=audio_strategy,
            sub_clip_sec=sub_clip_sec,
            clips_per_audio=clips_per_audio,
            min_duration_sec=min_duration_sec,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            max_image_edge=max_image_edge,
            max_image_pixels=max_image_pixels,
        )

    # ------------------------------------------------------------------
    # infer_image_event_ranks / semantic and AQ similarity
    # ------------------------------------------------------------------

    def infer_image_event_ranks(
        self,
        *,
        image_event_backend: str = "zeroshot",
        event_classes: list[str] | None = None,
        prompt_templates: dict[str, list[str]] | None = None,
        device: str | None = None,
        output_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        batch_size: int = 32,
        similarity_threshold: float | None = None,
        similarity_column: str = "similarity",
        max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        max_image_pixels: int | None = None,
        max_image_size: tuple[int, int] | None = None,
    ) -> dict:
        """Infer image-side SoundAQnet event rankings with zero-shot ImageBind.

        ``image_event_backend`` currently only accepts ``"zeroshot"``. The
        known 15 SoundAQnet event labels are scored against each image by
        ImageBind vision-text cosine similarity, then sorted from most likely
        to least likely. This is much faster and more deterministic than a
        generative VLM because the event set is fixed.

        Returns
        -------
        dict
            ``{image_id: [event_label, ...]}`` rank mapping.
        """
        if self.pairs.empty:
            raise RuntimeError("Call build_pairs() first.")
        if image_event_backend != "zeroshot":
            raise ValueError("image_event_backend can only be 'zeroshot'.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_image_size is not None:
            max_image_edge = max(max_image_size)
        if max_image_pixels is not None:
            max_image_edge = max_image_pixels

        event_classes = event_classes or SOUND_EVENT_CLASSES

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        existing_ranks = _image_event_ckpt_load(ckpt_path)

        _tmp_dir: tempfile.TemporaryDirectory | None = None
        if output_dir is None:
            _tmp_dir = tempfile.TemporaryDirectory(prefix="opensip_image_events_")
            work_dir = Path(_tmp_dir.name)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        image_dir = work_dir / "image_event_images"
        image_dir.mkdir(exist_ok=True)

        try:
            candidates = self.pairs.copy()
            if similarity_threshold is not None and similarity_column in candidates.columns:
                candidates = candidates[
                    candidates[similarity_column].notna()
                    & (candidates[similarity_column] >= similarity_threshold)
                ]

            image_rows = (
                candidates[["image_id", "image_url"]]
                .dropna(subset=["image_id", "image_url"])
                .drop_duplicates(subset=["image_id"])
            )

            pending: dict = {}
            for row in image_rows.itertuples(index=False):
                image_id = row.image_id
                if image_id in existing_ranks:
                    continue
                local = _ensure_local(row.image_url, image_dir, f"img_{image_id}")
                if local is None:
                    if ckpt_path:
                        _append_ckpt(
                            ckpt_path,
                            {"image_id": image_id, "reason": "image_unavailable"},
                        )
                    continue
                prepared = _prepare_event_rank_image(
                    local,
                    image_dir,
                    f"zeroshot_{image_id}",
                    max_image_edge,
                )
                if prepared is None:
                    if ckpt_path:
                        _append_ckpt(
                            ckpt_path,
                            {"image_id": image_id, "reason": "unreadable_image"},
                        )
                    continue
                pending[image_id] = str(prepared)

            ranks = dict(existing_ranks)
            score_maps: dict = {}
            if pending:
                new_ranks, score_maps = _zeroshot_image_event_ranks(
                    pending,
                    event_classes,
                    prompt_templates=prompt_templates,
                    device=device,
                    batch_size=batch_size,
                    max_image_edge=max_image_edge,
                )
                ranks.update(new_ranks)
                if ckpt_path:
                    for image_id, ranking in new_ranks.items():
                        _append_ckpt(
                            ckpt_path,
                            {
                                "image_id": image_id,
                                "image_path": pending.get(image_id),
                                "ranking": ranking,
                                "scores": score_maps.get(image_id, {}),
                            },
                        )
            for image_id in image_rows["image_id"].tolist():
                if image_id not in ranks and ckpt_path:
                    _append_ckpt(
                        ckpt_path,
                        {
                            "image_id": image_id,
                            "reason": "not_ranked",
                        },
                    )

            for i in range(len(event_classes)):
                col = f"image_event_rank_{i + 1}"
                self.pairs[col] = self.pairs["image_id"].map(
                    lambda image_id, ii=i: (
                        ranks.get(image_id, [None] * len(event_classes))[ii]
                        if image_id in ranks and ii < len(ranks.get(image_id, []))
                        else None
                    )
                )
            all_scores = dict(getattr(self, "image_event_scores", {}) or {})
            all_scores.update(score_maps)
            for event in event_classes:
                self.pairs[f"image_event_score_{event}"] = self.pairs["image_id"].map(
                    lambda image_id, e=event: (
                        all_scores.get(image_id, {}).get(e)
                        if image_id in all_scores else None
                    )
                )
            self.image_event_ranks = ranks
            self.image_event_scores = all_scores
            return ranks
        finally:
            if _tmp_dir is not None:
                _tmp_dir.cleanup()

    def infer_image_placepulse_aq(
        self,
        placepulse_predictor=None,
        *,
        placepulse_repo_id: str = "Jiani11/human-perception-place-pulse",
        placepulse_cache_dir: str | Path | None = None,
        placepulse_device: str | None = None,
        output_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        batch_size: int = 32,
        similarity_threshold: float | None = None,
        similarity_column: str = "similarity",
        max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        max_image_pixels: int | None = None,
        panorama_strategy: str = "auto",
        panorama_headings: list[float] | None = None,
        panorama_fov: float = 90.0,
        panorama_phi: float = 0.0,
        panorama_size: tuple[int, int] = (768, 768),
        panorama_min_aspect_ratio: float = 1.8,
        lively_dimension: str = "vibrant",
        boring_dimension: str = "monotonous",
    ) -> dict:
        """Infer image AQ from PlacePulse lively/boring predictions.

        When ``placepulse_predictor`` is omitted, openSIP downloads and caches
        the Hugging Face ``lively.pth`` and ``boring.pth`` checkpoints from
        ``placepulse_repo_id``. A custom predictor can also be supplied as a
        callable, or an object with ``predict``/``predict_batch``. It should
        return lively and boring scores for each input image. Scores are mapped
        to image AQ dimensions as: ``lively -> vibrant`` and
        ``boring -> monotonous`` by default.

        For 360-degree equirectangular street-view panoramas,
        ``panorama_strategy="auto"`` detects wide images and scores perspective
        crops instead of squeezing the full panorama into the 384x384 model
        input. Use ``"none"`` to disable this or ``"perspective"`` to force it.

        Returns
        -------
        dict
            ``{image_id: {"vibrant": lively, "monotonous": boring}}`` mapping.
        """
        if self.pairs.empty:
            raise RuntimeError("Call build_pairs() first.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_image_pixels is not None:
            max_image_edge = max_image_pixels
        panorama_headings = panorama_headings or [0, 90, 180, 270]
        if placepulse_predictor is None:
            from opensip.placepulse import PlacePulsePredictor

            placepulse_predictor = PlacePulsePredictor(
                repo_id=placepulse_repo_id,
                cache_dir=placepulse_cache_dir,
                device=placepulse_device,
            )

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        existing_scores = _placepulse_ckpt_load(ckpt_path)

        _tmp_dir: tempfile.TemporaryDirectory | None = None
        if output_dir is None:
            _tmp_dir = tempfile.TemporaryDirectory(prefix="opensip_placepulse_")
            work_dir = Path(_tmp_dir.name)
        else:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        image_dir = work_dir / "placepulse_images"
        image_dir.mkdir(exist_ok=True)

        try:
            candidates = self.pairs.copy()
            if similarity_threshold is not None and similarity_column in candidates.columns:
                candidates = candidates[
                    candidates[similarity_column].notna()
                    & (candidates[similarity_column] >= similarity_threshold)
                ]

            image_rows = (
                candidates[["image_id", "image_url"]]
                .dropna(subset=["image_id", "image_url"])
                .drop_duplicates(subset=["image_id"])
            )

            pending: dict = {}
            view_to_image: dict = {}
            image_view_counts: dict = {}
            for row in image_rows.itertuples(index=False):
                image_id = row.image_id
                if image_id in existing_scores:
                    continue
                local = _ensure_local(row.image_url, image_dir, f"img_{image_id}")
                if local is None:
                    if ckpt_path:
                        _append_ckpt(
                            ckpt_path,
                            {"image_id": image_id, "reason": "image_unavailable"},
                        )
                    continue
                prepared_paths = _prepare_placepulse_images(
                    local,
                    image_dir,
                    f"placepulse_{image_id}",
                    max_image_edge=max_image_edge,
                    panorama_strategy=panorama_strategy,
                    panorama_headings=panorama_headings,
                    panorama_fov=panorama_fov,
                    panorama_phi=panorama_phi,
                    panorama_size=panorama_size,
                    panorama_min_aspect_ratio=panorama_min_aspect_ratio,
                )
                if not prepared_paths:
                    if ckpt_path:
                        _append_ckpt(
                            ckpt_path,
                            {"image_id": image_id, "reason": "unreadable_image"},
                        )
                    continue
                image_view_counts[image_id] = len(prepared_paths)
                for view_idx, prepared in enumerate(prepared_paths):
                    view_id = f"{image_id}__view{view_idx:02d}"
                    pending[view_id] = str(prepared)
                    view_to_image[view_id] = image_id

            placepulse_scores = dict(existing_scores)
            items = list(pending.items())
            view_scores_by_image: dict = {}
            for offset in range(0, len(items), batch_size):
                chunk = dict(items[offset : offset + batch_size])
                chunk_scores = _call_placepulse_predictor(placepulse_predictor, chunk)
                for view_id, scores in chunk_scores.items():
                    image_id = view_to_image.get(view_id, view_id)
                    view_scores_by_image.setdefault(image_id, []).append(scores)
                    if ckpt_path:
                        _append_ckpt(
                            ckpt_path,
                            {
                                "image_id": image_id,
                                "view_id": view_id,
                                "image_path": chunk.get(view_id),
                                "scores": scores,
                            },
                        )

            for image_id, score_list in view_scores_by_image.items():
                lively_vals = [s["lively"] for s in score_list if "lively" in s]
                boring_vals = [s["boring"] for s in score_list if "boring" in s]
                if lively_vals and boring_vals:
                    placepulse_scores[image_id] = {
                        "lively": float(np.mean(lively_vals)),
                        "boring": float(np.mean(boring_vals)),
                        "n_views": int(len(score_list)),
                    }

            image_aq = {
                image_id: {
                    lively_dimension: float(scores["lively"]),
                    boring_dimension: float(scores["boring"]),
                }
                for image_id, scores in placepulse_scores.items()
                if "lively" in scores and "boring" in scores
            }
            self.pairs["image_placepulse_lively"] = self.pairs["image_id"].map(
                lambda image_id: (
                    placepulse_scores.get(image_id, {}).get("lively")
                    if image_id in placepulse_scores else None
                )
            )
            self.pairs["image_placepulse_boring"] = self.pairs["image_id"].map(
                lambda image_id: (
                    placepulse_scores.get(image_id, {}).get("boring")
                    if image_id in placepulse_scores else None
                )
            )
            self.pairs["image_placepulse_n_views"] = self.pairs["image_id"].map(
                lambda image_id: (
                    placepulse_scores.get(image_id, {}).get("n_views")
                    if image_id in placepulse_scores else None
                )
            )
            self.pairs[f"image_aq_{lively_dimension}"] = self.pairs["image_id"].map(
                lambda image_id: (
                    image_aq.get(image_id, {}).get(lively_dimension)
                    if image_id in image_aq else None
                )
            )
            self.pairs[f"image_aq_{boring_dimension}"] = self.pairs["image_id"].map(
                lambda image_id: (
                    image_aq.get(image_id, {}).get(boring_dimension)
                    if image_id in image_aq else None
                )
            )
            self.image_placepulse_scores = placepulse_scores
            self.image_placepulse_aq = image_aq
            return image_aq
        finally:
            if _tmp_dir is not None:
                _tmp_dir.cleanup()

    def compute_semantic_similarity(
        self,
        audio_events: dict | None = None,
        image_events: dict | None = None,
        *,
        infer_audio_events: bool = True,
        soundaqnet_model: str | Path = "SoundAQnet_ASC96_AEC94_PAQ1027",
        soundaqnet_device: str | None = None,
        soundaqnet_batch_size: int = 32,
        soundaqnet_num_workers: int = 4,
        infer_image_events: bool = True,
        image_event_backend: str = "zeroshot",
        image_event_checkpoint_path: str | Path | None = None,
        image_event_batch_size: int = 32,
        image_event_similarity_threshold: float | None = None,
        image_event_max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        image_event_max_image_pixels: int | None = None,
        image_event_max_image_size: tuple[int, int] | None = None,
        output_dir: str | Path | None = None,
        events: list[str] | None = None,
        audio_prefix: str = "audio_event_",
        image_prefix: str = "image_event_",
        top_k: int = 5,
        score_higher_is_better: bool = True,
        values_are_ranks: bool = False,
    ) -> pd.DataFrame:
        """Evaluate semantic event-rank agreement for each audio-image pair.

        This follows the poster's semantic layer: compare sound-event rankings
        inferred for the audio clip and image using NDCG@k, MRR, and Spearman
        rank correlation. When ``audio_events`` is omitted,
        ``infer_audio_events=True`` runs the optional ``soundaqnet`` package on
        the audio clips to obtain audio event probabilities. Image-side event
        ranks are inferred by default with zero-shot ImageBind vision-text
        scoring over the same fixed 15 SoundAQnet event labels, unless
        ``image_events`` or image event columns are already present.

        Parameters
        ----------
        audio_events, image_events:
            Optional mappings. Audio keys may be ``audio_id`` or
            ``(audio_id, clip_index)``; image keys are ``image_id``. Values may
            be ordered event lists or ``{event: score}`` dictionaries.
        infer_audio_events:
            If True and ``audio_events`` is None, infer audio event ranks with
            ``soundaqnet``. SoundAQnet is part of openSIP's core install.
        soundaqnet_model, soundaqnet_device, soundaqnet_batch_size, soundaqnet_num_workers:
            Forwarded to ``soundaqnet.SoundAQnet`` and ``predict_from_audio``.
        infer_image_events:
            Must be True. Image-side ranks are required for semantic
            similarity and are inferred when not supplied.
        image_event_backend:
            Must be ``"zeroshot"``.
        image_event_checkpoint_path, image_event_batch_size,
        image_event_similarity_threshold, image_event_max_image_edge,
        image_event_max_image_pixels,
        image_event_max_image_size:
            Forwarded to :meth:`infer_image_event_ranks`.
        output_dir:
            Working directory for local audio/image files when internal
            inference is needed. If None, a temporary directory is used and
            cleaned up.
        events:
            Optional event label order when values are numeric arrays.
        audio_prefix, image_prefix:
            If mappings are not provided for a row, columns with these prefixes
            are read from ``self.pairs`` and interpreted as event scores.
        top_k:
            Rank cutoff for NDCG@k. The poster used top 5.
        score_higher_is_better:
            For score dictionaries/columns, higher values rank earlier when
            True. Ignored for ordered lists and ``values_are_ranks=True``.
        values_are_ranks:
            Interpret dictionary/column values as ranks directly, where rank 1
            is best.

        Returns
        -------
        pandas.DataFrame
            ``self.pairs`` with ``semantic_ndcg_at_k``, ``semantic_mrr``,
            ``semantic_spearman``, and ``semantic_similarity`` columns.
        """
        if self.pairs.empty:
            raise RuntimeError("Call build_pairs() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if infer_image_events is not True:
            raise ValueError("infer_image_events can only be True.")
        if image_event_backend != "zeroshot":
            raise ValueError("image_event_backend can only be 'zeroshot'.")

        has_audio_event_columns = any(
            str(c).startswith(audio_prefix) or str(c).startswith("audio_event_rank_")
            for c in self.pairs.columns
        )
        if audio_events is None and infer_audio_events and not has_audio_event_columns:
            try:
                soundaq_df, stem_to_key = self._predict_soundaqnet(
                    model_name=soundaqnet_model,
                    device=soundaqnet_device,
                    batch_size=soundaqnet_batch_size,
                    num_workers=soundaqnet_num_workers,
                    output_dir=output_dir,
                )
            except ImportError as exc:
                raise ImportError(
                    "compute_semantic_similarity can infer audio event ranks with "
                    "the soundaqnet package, which is part of openSIP's core "
                    "dependencies.\n"
                    "Install or repair with: pip install openSIP\n"
                    "Or pass audio_events=... to skip internal SoundAQnet inference."
                ) from exc
            if not soundaq_df.empty:
                audio_events = _soundaqnet_to_event_mapping(soundaq_df, stem_to_key)
            else:
                audio_events = {}

        has_image_event_columns = any(
            str(c).startswith(image_prefix) or str(c).startswith("image_event_rank_")
            for c in self.pairs.columns
        )
        if image_events is None and infer_image_events and not has_image_event_columns:
            image_events = self.infer_image_event_ranks(
                image_event_backend=image_event_backend,
                event_classes=events or SOUND_EVENT_CLASSES,
                output_dir=output_dir,
                checkpoint_path=image_event_checkpoint_path,
                batch_size=image_event_batch_size,
                similarity_threshold=image_event_similarity_threshold,
                max_image_edge=image_event_max_image_edge,
                max_image_pixels=image_event_max_image_pixels,
                max_image_size=image_event_max_image_size,
            )

        pairs = self.pairs.copy()
        ndcgs: list[float] = []
        mrrs: list[float] = []
        spearmans: list[float] = []

        for _, row in pairs.iterrows():
            a_val = _mapping_get(audio_events, row["audio_id"], row["clip_index"])
            i_val = _mapping_get(image_events, row["image_id"])
            if a_val is None:
                a_val = _extract_rank_columns(row, "audio_event_rank_")
            if a_val is None:
                a_val = _extract_prefixed_values(row, audio_prefix)
            if i_val is None:
                i_val = _extract_rank_columns(row, "image_event_rank_")
            if i_val is None:
                i_val = _extract_prefixed_values(row, image_prefix)

            a_rank = _coerce_rank_map(
                a_val,
                labels=events,
                score_higher_is_better=score_higher_is_better,
                values_are_ranks=values_are_ranks,
            )
            i_rank = _coerce_rank_map(
                i_val,
                labels=events,
                score_higher_is_better=score_higher_is_better,
                values_are_ranks=values_are_ranks,
            )

            if not a_rank or not i_rank:
                ndcgs.append(float("nan"))
                mrrs.append(float("nan"))
                spearmans.append(float("nan"))
                continue
            ndcgs.append(_ndcg_at_k(a_rank, i_rank, top_k))
            mrrs.append(_mrr(a_rank, i_rank))
            spearmans.append(_spearman_from_ranks(a_rank, i_rank))

        pairs["semantic_ndcg_at_k"] = ndcgs
        pairs["semantic_mrr"] = mrrs
        pairs["semantic_spearman"] = spearmans
        pairs["semantic_top_k"] = int(top_k)
        pairs["semantic_similarity"] = pairs[["semantic_ndcg_at_k", "semantic_mrr"]].mean(
            axis=1, skipna=True
        )
        self.pairs = pairs
        return self.pairs

    def compute_aq_similarity(
        self,
        audio_aq: dict | None = None,
        image_aq: dict | None = None,
        *,
        aq_dimensions: list[str] | None = None,
        audio_prefix: str = "audio_aq_",
        image_prefix: str = "image_aq_",
        infer_audio_aq: bool = True,
        infer_audio_events: bool = True,
        infer_image_events: bool = True,
        soundaqnet_model: str | Path = "SoundAQnet_ASC96_AEC94_PAQ1027",
        soundaqnet_device: str | None = None,
        soundaqnet_batch_size: int = 32,
        soundaqnet_num_workers: int = 4,
        infer_image_placepulse: bool = False,
        placepulse_predictor=None,
        placepulse_repo_id: str = "Jiani11/human-perception-place-pulse",
        placepulse_cache_dir: str | Path | None = None,
        placepulse_device: str | None = None,
        placepulse_checkpoint_path: str | Path | None = None,
        placepulse_batch_size: int = 32,
        placepulse_similarity_threshold: float | None = None,
        placepulse_max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        placepulse_max_image_pixels: int | None = None,
        placepulse_panorama_strategy: str = "auto",
        placepulse_panorama_headings: list[float] | None = None,
        placepulse_panorama_fov: float = 90.0,
        placepulse_panorama_phi: float = 0.0,
        placepulse_panorama_size: tuple[int, int] = (768, 768),
        placepulse_panorama_min_aspect_ratio: float = 1.8,
        image_event_backend: str = "zeroshot",
        image_event_checkpoint_path: str | Path | None = None,
        image_event_batch_size: int = 32,
        image_event_similarity_threshold: float | None = None,
        image_event_max_image_edge: int | None = DEFAULT_IMAGE_MAX_EDGE,
        image_event_max_image_pixels: int | None = None,
        image_event_max_image_size: tuple[int, int] | None = None,
        output_dir: str | Path | None = None,
        audio_events: dict | None = None,
        image_events: dict | None = None,
        event_aq_scores=None,
        event_labels: list[str] | None = None,
        top_k: int = 5,
        score_higher_is_better: bool = True,
        values_are_ranks: bool = False,
    ) -> pd.DataFrame:
        """Evaluate affective-quality agreement for each audio-image pair.

        This implements the poster's AQ layer. Direct AQ vectors are compared
        with RMSE and Pearson correlation. Missing audio AQ can be inferred with
        SoundAQnet when available. For either audio or image, missing AQ can
        also be approximated from semantic event rankings by averaging AQ scores
        for the top-k events using ``event_aq_scores``.

        Parameters
        ----------
        audio_aq, image_aq:
            Optional mappings of ``audio_id`` / ``(audio_id, clip_index)`` and
            ``image_id`` to ``{dimension: score}`` vectors.
        aq_dimensions:
            Optional dimension order for numeric arrays. Defaults to the eight
            ISO soundscape affective-quality dimensions.
        audio_prefix, image_prefix:
            Prefixes for direct AQ columns in ``self.pairs``.
        infer_audio_aq:
            If True and audio AQ is not supplied, try to infer direct audio AQ
            with ``soundaqnet``.
        infer_audio_events:
            If True and audio AQ still cannot be obtained directly, infer audio
            event ranks with ``soundaqnet`` so AQ can be approximated from
            ``event_aq_scores``.
        infer_image_events:
            If True and image AQ is not supplied, infer image event ranks with
            zero-shot ImageBind so AQ can be approximated from
            ``event_aq_scores``.
        soundaqnet_model, soundaqnet_device, soundaqnet_batch_size, soundaqnet_num_workers:
            Forwarded to ``soundaqnet.SoundAQnet`` and ``predict_from_audio``.
        infer_image_placepulse:
            If True and image AQ is not supplied, infer image AQ from
            PlacePulse lively/boring predictions. Lively maps to sound AQ
            ``vibrant`` and boring maps to ``monotonous``.
        placepulse_predictor, placepulse_repo_id, placepulse_cache_dir,
        placepulse_device, placepulse_checkpoint_path, placepulse_batch_size,
        placepulse_similarity_threshold, placepulse_max_image_edge,
        placepulse_max_image_pixels, placepulse_panorama_strategy,
        placepulse_panorama_headings, placepulse_panorama_fov,
        placepulse_panorama_phi, placepulse_panorama_size,
        placepulse_panorama_min_aspect_ratio:
            Forwarded to :meth:`infer_image_placepulse_aq`.
        image_event_backend, image_event_checkpoint_path, image_event_batch_size,
        image_event_similarity_threshold, image_event_max_image_edge,
        image_event_max_image_pixels, image_event_max_image_size:
            Forwarded to :meth:`infer_image_event_ranks` when image event ranks
            are needed for AQ approximation.
        output_dir:
            Working directory for local audio/image files when internal
            inference is needed. If None, a temporary directory is used and
            cleaned up.
        audio_events, image_events, event_aq_scores:
            Optional semantic rankings plus event-to-AQ table used to derive AQ
            vectors when direct vectors are unavailable. ``event_aq_scores`` can
            be a DataFrame indexed by event or with an ``event`` column, or a
            nested dict ``{event: {dimension: score}}``.
        top_k:
            Number of top semantic events to average when deriving AQ vectors.

        Returns
        -------
        pandas.DataFrame
            ``self.pairs`` with ``aq_rmse``, ``aq_pearson``, and
            ``aq_similarity`` columns. Lower ``aq_rmse`` is better; higher
            ``aq_pearson``/``aq_similarity`` is better.
        """
        if self.pairs.empty:
            raise RuntimeError("Call build_pairs() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        aq_dimensions = aq_dimensions or [
            "pleasant",
            "eventful",
            "chaotic",
            "vibrant",
            "uneventful",
            "calm",
            "annoying",
            "monotonous",
        ]
        event_aq = _coerce_event_aq_scores(event_aq_scores)

        has_audio_aq_columns = any(str(c).startswith(audio_prefix) for c in self.pairs.columns)
        has_audio_event_columns = any(
            str(c).startswith("audio_event_") or str(c).startswith("audio_event_rank_")
            for c in self.pairs.columns
        )
        can_derive_audio_aq = event_aq is not None and (
            audio_events is not None or has_audio_event_columns
        )
        needs_audio_direct = (
            audio_aq is None
            and infer_audio_aq
            and not has_audio_aq_columns
            and not can_derive_audio_aq
        )
        needs_audio_events = (
            audio_events is None
            and infer_audio_events
            and event_aq is not None
            and not has_audio_event_columns
        )

        if needs_audio_direct or needs_audio_events:
            try:
                soundaq_df, stem_to_key = self._predict_soundaqnet(
                    model_name=soundaqnet_model,
                    device=soundaqnet_device,
                    batch_size=soundaqnet_batch_size,
                    num_workers=soundaqnet_num_workers,
                    output_dir=output_dir,
                )
            except ImportError as exc:
                if needs_audio_direct:
                    raise ImportError(
                        "compute_aq_similarity can infer audio AQ with soundaqnet, "
                        "which is part of openSIP's core dependencies.\n"
                        "Install or repair with: pip install openSIP\n"
                        "Or pass audio_aq=... / audio_events=... to skip internal inference."
                    ) from exc
                soundaq_df = pd.DataFrame()
                stem_to_key = {}
            else:
                if not soundaq_df.empty:
                    if needs_audio_direct:
                        audio_aq = _soundaqnet_to_aq_mapping(
                            soundaq_df, stem_to_key, aq_dimensions
                        )
                    if needs_audio_events:
                        audio_events = _soundaqnet_to_event_mapping(soundaq_df, stem_to_key)
                else:
                    if needs_audio_direct:
                        audio_aq = {}
                    if needs_audio_events:
                        audio_events = {}

        has_image_aq_columns = any(str(c).startswith(image_prefix) for c in self.pairs.columns)
        has_image_event_columns = any(
            str(c).startswith("image_event_") or str(c).startswith("image_event_rank_")
            for c in self.pairs.columns
        )
        if (
            image_aq is None
            and infer_image_placepulse
            and not has_image_aq_columns
        ):
            image_aq = self.infer_image_placepulse_aq(
                placepulse_predictor=placepulse_predictor,
                placepulse_repo_id=placepulse_repo_id,
                placepulse_cache_dir=placepulse_cache_dir,
                placepulse_device=placepulse_device,
                output_dir=output_dir,
                checkpoint_path=placepulse_checkpoint_path,
                batch_size=placepulse_batch_size,
                similarity_threshold=placepulse_similarity_threshold,
                max_image_edge=placepulse_max_image_edge,
                max_image_pixels=placepulse_max_image_pixels,
                panorama_strategy=placepulse_panorama_strategy,
                panorama_headings=placepulse_panorama_headings,
                panorama_fov=placepulse_panorama_fov,
                panorama_phi=placepulse_panorama_phi,
                panorama_size=placepulse_panorama_size,
                panorama_min_aspect_ratio=placepulse_panorama_min_aspect_ratio,
            )
            has_image_aq_columns = any(
                str(c).startswith(image_prefix) for c in self.pairs.columns
            )
        if (
            image_aq is None
            and image_events is None
            and infer_image_events
            and event_aq is not None
            and not has_image_aq_columns
            and not has_image_event_columns
        ):
            if image_event_backend != "zeroshot":
                raise ValueError("image_event_backend can only be 'zeroshot'.")
            image_events = self.infer_image_event_ranks(
                image_event_backend=image_event_backend,
                event_classes=event_labels or SOUND_EVENT_CLASSES,
                output_dir=output_dir,
                checkpoint_path=image_event_checkpoint_path,
                batch_size=image_event_batch_size,
                similarity_threshold=image_event_similarity_threshold,
                max_image_edge=image_event_max_image_edge,
                max_image_pixels=image_event_max_image_pixels,
                max_image_size=image_event_max_image_size,
            )

        pairs = self.pairs.copy()
        rmses: list[float] = []
        pears: list[float] = []
        aq_types: list[str | None] = []

        for _, row in pairs.iterrows():
            a_val = _mapping_get(audio_aq, row["audio_id"], row["clip_index"])
            i_val = _mapping_get(image_aq, row["image_id"])
            row_has_placepulse = (
                "image_placepulse_lively" in row.index
                and pd.notna(row.get("image_placepulse_lively"))
            )
            aq_type = "placepulse" if row_has_placepulse else "direct"
            if a_val is None:
                a_val = _extract_prefixed_values(row, audio_prefix)
            if i_val is None:
                i_val = _extract_prefixed_values(row, image_prefix)
                if i_val is not None and row_has_placepulse:
                    aq_type = "placepulse"

            a_vec = _coerce_vector(a_val, aq_dimensions)
            i_vec = _coerce_vector(i_val, aq_dimensions)

            if (a_vec is None or i_vec is None) and event_aq is not None:
                derived_from_event_rank = False
                a_event_val = _mapping_get(audio_events, row["audio_id"], row["clip_index"])
                i_event_val = _mapping_get(image_events, row["image_id"])
                if a_event_val is None:
                    a_event_val = _extract_rank_columns(row, "audio_event_rank_")
                if a_event_val is None:
                    a_event_val = _extract_prefixed_values(row, "audio_event_")
                if i_event_val is None:
                    i_event_val = _extract_rank_columns(row, "image_event_rank_")
                if i_event_val is None:
                    i_event_val = _extract_prefixed_values(row, "image_event_")
                a_rank = _coerce_rank_map(
                    a_event_val,
                    labels=event_labels,
                    score_higher_is_better=score_higher_is_better,
                    values_are_ranks=values_are_ranks,
                )
                i_rank = _coerce_rank_map(
                    i_event_val,
                    labels=event_labels,
                    score_higher_is_better=score_higher_is_better,
                    values_are_ranks=values_are_ranks,
                )
                if a_vec is None:
                    a_vec = _aq_from_event_rank(a_rank, event_aq, top_k)
                    derived_from_event_rank = a_vec is not None
                if i_vec is None:
                    i_vec = _aq_from_event_rank(i_rank, event_aq, top_k)
                    derived_from_event_rank = i_vec is not None or derived_from_event_rank
                if derived_from_event_rank:
                    aq_type = "event_rank"

            if not a_vec or not i_vec:
                rmses.append(float("nan"))
                pears.append(float("nan"))
                aq_types.append(None)
                continue
            rmses.append(_rmse_vectors(a_vec, i_vec))
            pears.append(_pearson_vectors(a_vec, i_vec))
            aq_types.append(aq_type)

        pairs["aq_rmse"] = rmses
        pairs["aq_pearson"] = pears
        pairs["aq_similarity"] = pears
        pairs["aq_similarity_type"] = aq_types
        for aq_type in ("direct", "event_rank", "placepulse"):
            pairs[f"aq_similarity_{aq_type}"] = pairs["aq_similarity"].where(
                pairs["aq_similarity_type"] == aq_type
            )
        self.pairs = pairs
        return self.pairs

    # ------------------------------------------------------------------
    # filter_pairs_by_similarity
    # ------------------------------------------------------------------

    def filter_pairs_by_similarity(
        self,
        min_similarity: float = 0.0,
        min_sub_clips: int = 1,
        top_k_per_audio: int | None = None,
    ) -> "AudioImagePairDataset":
        """Filter ``self.pairs`` in place and return ``self``.

        Parameters
        ----------
        min_similarity:
            Drop pairs whose ``similarity`` is below this threshold.  Pairs
            with ``NaN`` similarity (not yet computed) are **kept**.
        min_sub_clips:
            Drop pairs from audio clips that produced fewer than this many
            sub-clips (e.g. very short recordings).
        top_k_per_audio:
            Keep at most *k* images per ``(loc_id, audio_id, clip_index)``.
            Ranked by ``similarity`` descending; ``NaN`` entries go last.
            ``top_k_per_audio=1`` selects the single best-matching image per
            audio clip.

        Returns
        -------
        AudioImagePairDataset
            ``self`` (for method chaining).
        """
        df = self.pairs.copy()

        if min_similarity > 0.0:
            mask = df["similarity"].isna() | (df["similarity"] >= min_similarity)
            df = df[mask]

        if min_sub_clips > 1:
            mask = df["n_sub_clips"].isna() | (df["n_sub_clips"] >= min_sub_clips)
            df = df[mask]

        if top_k_per_audio is not None:
            df = (
                df.sort_values("similarity", ascending=False, na_position="last")
                .groupby(["loc_id", "audio_id", "clip_index"], sort=False)
                .head(top_k_per_audio)
                .reset_index(drop=True)
            )

        self.pairs = df.reset_index(drop=True)
        return self

    def filter_pairs(
        self,
        min_similarity: float = 0.0,
        min_sub_clips: int = 1,
        top_k_per_audio: int | None = None,
    ) -> "AudioImagePairDataset":
        """Backward-compatible alias for :meth:`filter_pairs_by_similarity`."""
        return self.filter_pairs_by_similarity(
            min_similarity=min_similarity,
            min_sub_clips=min_sub_clips,
            top_k_per_audio=top_k_per_audio,
        )

    # ------------------------------------------------------------------
    # Final pair schema and graph view
    # ------------------------------------------------------------------

    def finalize_pairs(self, include_extra_columns: bool = True) -> pd.DataFrame:
        """Return a canonical final edge table for sound-image pairs.

        The returned table is one row per sound-image edge.  It adds stable
        pair/node IDs, coordinate-derived ``distance_m`` when coordinates are
        available, an explicit ``embedding_similarity`` alias for ImageBind
        cosine similarity, and keeps separate AQ similarity columns such as
        ``aq_similarity_direct``, ``aq_similarity_event_rank``, and
        ``aq_similarity_placepulse`` when they have been computed.
        """
        if self.pairs is None or self.pairs.empty:
            return pd.DataFrame(columns=_FINAL_PAIR_BASE_COLS)

        df = self.pairs.copy()
        gtd = self._gtd

        df["pair_id"] = [
            _stable_pair_id(aid, cidx, iid)
            for aid, cidx, iid in zip(df["audio_id"], df["clip_index"], df["image_id"])
        ]
        df["audio_node_id"] = [
            _audio_node_id(aid, cidx)
            for aid, cidx in zip(df["audio_id"], df["clip_index"])
        ]
        df["image_node_id"] = df["image_id"].map(_image_node_id)

        if "embedding_similarity" not in df.columns and "similarity" in df.columns:
            df["embedding_similarity"] = df["similarity"]
        if "filter_status" not in df.columns:
            df["filter_status"] = "kept"
        if "filter_reason" not in df.columns:
            df["filter_reason"] = None
        if "selected" not in df.columns:
            df["selected"] = True

        audio_coord_by_id = {}
        audio_coord_by_id.update(_coord_lookup_from_frame(getattr(gtd, "audio_metadata", None)))
        audio_coord_by_id.update(_coord_lookup_from_payload(getattr(gtd, "audios", None)))
        unit_coord_by_loc = _coord_lookup_from_units(getattr(gtd, "units", None))

        image_coord_by_id = {}
        image_coord_by_id.update(_coord_lookup_from_frame(getattr(gtd, "svi_metadata", None)))
        image_coord_by_id.update(_coord_lookup_from_frame(getattr(gtd, "photo_metadata", None)))
        image_coord_by_id.update(_coord_lookup_from_payload(getattr(gtd, "svis", None)))
        image_coord_by_id.update(_coord_lookup_from_payload(getattr(gtd, "photos", None)))

        audio_path_by_id = _path_lookup_from_payload(getattr(gtd, "audios", None))
        image_path_by_id = {}
        image_path_by_id.update(_path_lookup_from_payload(getattr(gtd, "svis", None)))
        image_path_by_id.update(_path_lookup_from_payload(getattr(gtd, "photos", None)))

        if "audio_path" not in df.columns:
            df["audio_path"] = df["audio_id"].map(audio_path_by_id)
        if "image_path" not in df.columns:
            df["image_path"] = df["image_id"].map(image_path_by_id)

        def _audio_coord(row, key: str):
            coord = audio_coord_by_id.get(row["audio_id"]) or unit_coord_by_loc.get(row["loc_id"], {})
            return coord.get(key)

        def _image_coord(row, key: str):
            coord = image_coord_by_id.get(row["image_id"]) or unit_coord_by_loc.get(row["loc_id"], {})
            return coord.get(key)

        if "audio_lat" not in df.columns:
            df["audio_lat"] = df.apply(lambda row: _audio_coord(row, "lat"), axis=1)
        if "audio_lon" not in df.columns:
            df["audio_lon"] = df.apply(lambda row: _audio_coord(row, "lon"), axis=1)
        if "image_lat" not in df.columns:
            df["image_lat"] = df.apply(lambda row: _image_coord(row, "lat"), axis=1)
        if "image_lon" not in df.columns:
            df["image_lon"] = df.apply(lambda row: _image_coord(row, "lon"), axis=1)

        if "distance_m" not in df.columns:
            df["distance_m"] = [
                _haversine_m(a_lon, a_lat, i_lon, i_lat)
                for a_lon, a_lat, i_lon, i_lat in zip(
                    df["audio_lon"], df["audio_lat"], df["image_lon"], df["image_lat"]
                )
            ]

        if "aq_similarity" in df.columns and "aq_similarity_type" in df.columns:
            for aq_type in ("direct", "event_rank", "placepulse"):
                col = f"aq_similarity_{aq_type}"
                if col not in df.columns:
                    df[col] = df["aq_similarity"].where(df["aq_similarity_type"] == aq_type)

        extra_similarity_cols = [
            col
            for col in df.columns
            if (
                col.endswith("_similarity")
                or col.startswith("aq_similarity_")
                or col in {"aq_rmse", "aq_pearson", "semantic_ndcg_at_k", "semantic_mrr", "semantic_spearman"}
            )
            and col not in _FINAL_PAIR_BASE_COLS
        ]
        ordered = [col for col in _FINAL_PAIR_BASE_COLS if col in df.columns]
        ordered += [col for col in extra_similarity_cols if col not in ordered]
        if include_extra_columns:
            ordered += [col for col in df.columns if col not in ordered]
        return df[ordered]

    def export_final_dataframe(self, path: str | Path, include_extra_columns: bool = True) -> str:
        """Write :meth:`finalize_pairs` to CSV and return the absolute path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        final_df = self.finalize_pairs(include_extra_columns=include_extra_columns)
        final_df.to_csv(out, index=False)
        return str(out.resolve())

    def to_graph(self, pairs: pd.DataFrame | None = None):
        """Build a bipartite NetworkX graph from the final pair table.

        Audio sub-clips and images are nodes.  Every sound-image pair is an
        edge carrying spatial, temporal, embedding, semantic, and AQ attributes.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "Graph export requires networkx, which is part of openSIP's core "
                "dependencies. Install or repair with: pip install openSIP"
            ) from exc

        final_df = pairs.copy() if pairs is not None else self.finalize_pairs()
        graph = nx.Graph()
        for row in final_df.to_dict(orient="records"):
            audio_node = row["audio_node_id"]
            image_node = row["image_node_id"]
            audio_attrs = {
                "node_type": "audio_clip",
                "bipartite": 0,
                "audio_id": _graph_attr_value(row.get("audio_id")),
                "clip_index": _graph_attr_value(row.get("clip_index")),
                "source": _graph_attr_value(row.get("audio_source")),
                "timestamp": _graph_attr_value(row.get("audio_ts")),
                "lat": _graph_attr_value(row.get("audio_lat")),
                "lon": _graph_attr_value(row.get("audio_lon")),
                "url": _graph_attr_value(row.get("audio_url")),
                "path": _graph_attr_value(row.get("audio_path")),
            }
            graph.add_node(
                audio_node,
                **{key: value for key, value in audio_attrs.items() if value is not None},
            )
            image_attrs = {
                "node_type": "image",
                "bipartite": 1,
                "image_id": _graph_attr_value(row.get("image_id")),
                "source": _graph_attr_value(row.get("image_source")),
                "timestamp": _graph_attr_value(row.get("image_ts")),
                "lat": _graph_attr_value(row.get("image_lat")),
                "lon": _graph_attr_value(row.get("image_lon")),
                "url": _graph_attr_value(row.get("image_url")),
                "path": _graph_attr_value(row.get("image_path")),
            }
            graph.add_node(
                image_node,
                **{key: value for key, value in image_attrs.items() if value is not None},
            )
            edge_attrs = {
                key: _graph_attr_value(value)
                for key, value in row.items()
                if key not in {"audio_node_id", "image_node_id"}
            }
            edge_attrs = {key: value for key, value in edge_attrs.items() if value is not None}
            graph.add_edge(audio_node, image_node, **edge_attrs)
        return graph

    def export_graph(self, path: str | Path, pairs: pd.DataFrame | None = None) -> str:
        """Export the pair graph as GraphML or GEXF based on file extension."""
        graph = self.to_graph(pairs=pairs)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".graphml":
            import networkx as nx

            nx.write_graphml(graph, out)
        elif suffix == ".gexf":
            import networkx as nx

            nx.write_gexf(graph, out)
        else:
            raise ValueError("export_graph path must end with .graphml or .gexf.")
        return str(out.resolve())

    def plot_pair_graph(
        self,
        pairs: pd.DataFrame | None = None,
        *,
        color_edges_by: str = "embedding_similarity",
        width_edges_by: str | None = "embedding_similarity",
        label_nodes: bool = True,
        label_edges: bool = True,
        figsize: tuple[float, float] = (10, 6),
        ax=None,
    ):
        """Plot the sound-image bipartite graph with distance/time edge labels."""
        import matplotlib.pyplot as plt
        import networkx as nx

        graph = self.to_graph(pairs=pairs)
        final_df = pairs.copy() if pairs is not None else self.finalize_pairs()
        audio_nodes = sorted(
            [n for n, d in graph.nodes(data=True) if d.get("node_type") == "audio_clip"]
        )
        image_nodes = sorted(
            [n for n, d in graph.nodes(data=True) if d.get("node_type") == "image"]
        )
        max_n = max(len(audio_nodes), len(image_nodes), 1)
        pos = {}
        for i, node in enumerate(audio_nodes):
            pos[node] = (0.0, 1.0 - (i + 0.5) / max_n)
        for i, node in enumerate(image_nodes):
            pos[node] = (1.0, 1.0 - (i + 0.5) / max_n)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        edge_values = []
        for u, v in graph.edges():
            value = graph.edges[u, v].get(color_edges_by)
            edge_values.append(float(value) if value is not None else float("nan"))
        edge_widths = []
        for u, v in graph.edges():
            value = graph.edges[u, v].get(width_edges_by) if width_edges_by else None
            if value is None or pd.isna(value):
                edge_widths.append(1.0)
            else:
                edge_widths.append(0.8 + max(0.0, float(value)) * 3.0)

        nx.draw_networkx_nodes(
            graph, pos, nodelist=audio_nodes, node_color="#4C72B0", node_size=700, ax=ax
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=image_nodes, node_color="#55A868", node_size=700, ax=ax
        )
        edges = nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=edge_values,
            edge_cmap=plt.cm.viridis,
            width=edge_widths,
            alpha=0.75,
            ax=ax,
        )
        if label_nodes:
            labels = {
                node: (
                    f"{data.get('audio_id')}:{data.get('clip_index')}"
                    if data.get("node_type") == "audio_clip"
                    else str(data.get("image_id"))
                )
                for node, data in graph.nodes(data=True)
            }
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
        if label_edges:
            edge_labels = {}
            for row in final_df.to_dict(orient="records"):
                dist = row.get("distance_m")
                days = row.get("time_diff_days")
                parts = []
                if pd.notna(dist):
                    parts.append(f"{float(dist):.0f} m")
                if pd.notna(days):
                    parts.append(f"{float(days):.1f} d")
                if parts:
                    edge_labels[(row["audio_node_id"], row["image_node_id"])] = "\n".join(parts)
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                rotate=False,
                ax=ax,
            )
        if edge_values and any(not pd.isna(v) for v in edge_values):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            sm.set_array([v for v in edge_values if not pd.isna(v)])
            fig.colorbar(sm, ax=ax, label=color_edges_by)
        ax.set_axis_off()
        ax.set_title("Sound-Image Pair Graph")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: str | Path,
        data: bool = False,
    ) -> str:
        """Write the pairs metadata CSV (and optionally media files).

        Parameters
        ----------
        output_dir:
            Destination directory (created if it does not exist).
        data:
            When ``True`` every audio file and image referenced in
            ``self.pairs`` is downloaded into ``<output_dir>/audio/`` and
            ``<output_dir>/images/`` respectively.  URLs already present as
            local paths are copied.

        Returns
        -------
        str
            Absolute path to the written ``pairs_metadata.csv``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "pairs_metadata.csv"
        self.pairs.to_csv(csv_path, index=False)

        if data and not self.pairs.empty:
            audio_dir = out / "audio"
            image_dir = out / "images"
            audio_dir.mkdir(exist_ok=True)
            image_dir.mkdir(exist_ok=True)

            seen_audio: set = set()
            seen_image: set = set()
            for _, row in self.pairs.iterrows():
                a_url = row.get("audio_url")
                a_key = (row.get("audio_id"), int(row["clip_index"]))
                if a_url and a_key not in seen_audio:
                    seen_audio.add(a_key)
                    _ensure_local(
                        a_url,
                        audio_dir,
                        f"aud_{row['audio_id']}_{int(row['clip_index'])}",
                    )
                i_url = row.get("image_url")
                i_id = row.get("image_id")
                if i_url and i_id not in seen_image:
                    seen_image.add(i_id)
                    _ensure_local(i_url, image_dir, f"img_{i_id}")

        return str(csv_path.resolve())

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(
        self,
        kind: str = "all",
        time_unit: str = "auto",
        bins: int = 30,
        figsize: tuple | None = None,
        ax=None,
        threshold: float | None = None,
    ):
        """Visualise the distribution of pair metrics.

        Parameters
        ----------
        kind:
            ``"all"``          — 2×3 figure with all six panels (default).

            ``"similarity"``   — histogram of cosine similarity scores.

            ``"time_diff"``    — histogram of audio–image time differences.

            ``"n_sub_clips"``  — histogram of sub-clip counts per audio clip.

            ``"pairs_per_loc"`` — histogram of pairs per location.

            ``"source"``       — stacked bar showing audio / image source counts.

            ``"scatter"``      — similarity vs. time-difference scatter plot.
        time_unit:
            Time axis unit: ``"auto"`` (inferred from data), ``"s"``,
            ``"min"``, ``"h"``, ``"d"``, or ``"month"``.
        bins:
            Number of histogram bins.
        figsize:
            Figure size tuple.  Defaults to ``(14, 10)`` for ``"all"`` and
            ``(6, 4)`` for single panels.
        ax:
            Pre-existing :class:`matplotlib.axes.Axes` (single-panel only).
        threshold:
            Draw a vertical / horizontal reference line at this similarity
            value on ``"similarity"`` and ``"scatter"`` panels.

        Returns
        -------
        (fig, axes)
            For ``kind="all"`` *axes* is a ``(2, 3)`` ndarray; otherwise it
            is a single :class:`~matplotlib.axes.Axes`.
        """
        import matplotlib.pyplot as plt

        df = self.pairs
        if df.empty:
            raise RuntimeError("No pairs to plot — call build_pairs() first.")

        PANELS = [
            "similarity",
            "time_diff",
            "n_sub_clips",
            "pairs_per_loc",
            "source",
            "scatter",
        ]

        if kind == "all":
            fig, axes = plt.subplots(2, 3, figsize=figsize or (14, 10))
            for panel, ax_ in zip(PANELS, axes.flatten()):
                self._draw_panel(ax_, panel, df, time_unit, bins, threshold)
            fig.tight_layout()
            return fig, axes

        if kind not in PANELS:
            raise ValueError(
                f"Unknown kind={kind!r}. Choose from {PANELS + ['all']}."
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (6, 4))
        else:
            fig = ax.get_figure()
        self._draw_panel(ax, kind, df, time_unit, bins, threshold)
        fig.tight_layout()
        return fig, ax

    # internal panel renderer
    @staticmethod
    def _draw_panel(
        ax,
        kind: str,
        df: pd.DataFrame,
        time_unit: str,
        bins: int,
        threshold: float | None,
    ) -> None:
        _PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

        if kind == "similarity":
            vals = df["similarity"].dropna()
            ax.hist(vals, bins=bins, color=_PALETTE[0], edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Count")
            ax.set_title("Audio–Image Similarity")
            if threshold is not None:
                ax.axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"threshold = {threshold}",
                )
                ax.legend(fontsize=8)

        elif kind == "time_diff":
            vals = df["time_diff_days"].dropna()
            unit_label, divisor = _pick_time_unit(vals, time_unit)
            ax.hist(
                vals / divisor,
                bins=bins,
                color=_PALETTE[1],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel(f"Time difference ({unit_label})")
            ax.set_ylabel("Count")
            ax.set_title("Audio–Image Time Difference")

        elif kind == "n_sub_clips":
            vals = (
                df.drop_duplicates(["audio_id", "clip_index"])["n_sub_clips"].dropna()
            )
            n_bins = max(1, int(vals.max())) if not vals.empty else bins
            ax.hist(
                vals,
                bins=n_bins,
                color=_PALETTE[2],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel("Sub-clips per audio clip")
            ax.set_ylabel("Count")
            ax.set_title("Sub-clip Count Distribution")

        elif kind == "pairs_per_loc":
            counts = df.groupby("loc_id").size()
            ax.hist(counts, bins=bins, color=_PALETTE[3], edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Pairs per location")
            ax.set_ylabel("Count")
            ax.set_title("Pairs per Location")

        elif kind == "source":
            aud_counts = df["audio_source"].value_counts()
            img_counts = df["image_source"].value_counts()
            all_sources = sorted(set(aud_counts.index) | set(img_counts.index))
            bot_a = bot_i = 0.0
            for ci, src in enumerate(all_sources):
                ac = int(aud_counts.get(src, 0))
                ic = int(img_counts.get(src, 0))
                color = _PALETTE[ci % len(_PALETTE)]
                ax.bar([0], [ac], bottom=[bot_a], color=color, label=src, width=0.5)
                ax.bar([1], [ic], bottom=[bot_i], color=color, width=0.5)
                bot_a += ac
                bot_i += ic
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Audio source", "Image source"])
            ax.set_ylabel("Count")
            ax.set_title("Source Breakdown")
            ax.legend(fontsize=8)

        elif kind == "scatter":
            td = df["time_diff_days"].dropna()
            unit_label, divisor = _pick_time_unit(td, time_unit)
            ax.scatter(
                df["time_diff_days"] / divisor,
                df["similarity"],
                alpha=0.35,
                s=12,
                color=_PALETTE[0],
                linewidths=0,
            )
            ax.set_xlabel(f"Time difference ({unit_label})")
            ax.set_ylabel("Cosine similarity")
            ax.set_title("Similarity vs. Time Difference")
            if threshold is not None:
                ax.axhline(
                    threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"threshold = {threshold}",
                )
                ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Persistence — save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Snapshot the full pipeline state to *path*.

        Saves both the underlying :class:`~opensip.dataset.GeoTaggedData`
        (units, metadata, payload dicts) and the current ``pairs`` table to a
        single directory so the entire workflow can be resumed without
        re-downloading anything::

            apd.save("run/state")
            # later, in a new session:
            apd2 = AudioImagePairDataset.load("run/state")

        Parameters
        ----------
        path:
            Directory to write into (created if it does not exist).
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Persist the underlying GeoTaggedData first
        self._gtd.save(save_dir)
        # Persist the pairs table
        if self.pairs is not None and not self.pairs.empty:
            self.pairs.to_csv(save_dir / "pairs.csv", index=False)
        print(f"AudioImagePairDataset saved → {save_dir}")

    @classmethod
    def load(cls, path: str | Path) -> "AudioImagePairDataset":
        """Restore a full pipeline state from a directory created by :meth:`save`.

        Reconstructs the :class:`~opensip.dataset.GeoTaggedData` and the
        ``pairs`` table in one call::

            apd = AudioImagePairDataset.load("run/state")
            # apd._gtd, apd._gtd.audio_metadata, apd.pairs are all ready

        Parameters
        ----------
        path:
            Directory previously written by :meth:`save`.
        """
        from .dataset import GeoTaggedData  # local import avoids circular dep

        load_dir = Path(path)
        gtd = GeoTaggedData.load(load_dir)
        apd = cls(gtd)
        pairs_path = load_dir / "pairs.csv"
        if pairs_path.exists():
            apd.pairs = pd.read_csv(pairs_path)
        return apd
