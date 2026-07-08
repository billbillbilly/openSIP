from __future__ import annotations

import logging
import os
import warnings
from importlib.resources import as_file

import pandas as pd
from geopandas import GeoDataFrame
from tqdm.auto import tqdm

from .models import get_yunet_model_path
from .utils.building import *  # noqa: F401,F403  (also re-exports gpd, Path, etc.)
from .utils.checkpoint import (
    append_collection_checkpoint,
    load_collection_checkpoint,
    restore_audios_from_checkpoint,
    restore_photos_from_checkpoint,
    restore_svis_from_checkpoint,
)
from .utils.pano2pers import Equirectangular
from .utils.utils import (
    calculate_bearing,
    clip,
    closest,
    download_freesound_preview,
    download_image_requests,
    is_selfie_photo,
    is_url,
    projection,
    retry_request,
    save_base64,
)

logger = logging.getLogger("opensip")

class GeoTaggedData:
    def __init__(self,
                 locations: list|tuple|dict|pd.DataFrame=None,
                 units: GeoDataFrame=None):
        '''
        Args:
            locations (list|tuple|dict|Dataframe): A list of coordinates (longitude/x and latitude/y) or a dictionary keyed by longitude and latitude or a dataframe with columns "longitude" and "latitude".
            units (GeoDataFrame): The path to the shapefile or geojson file, or GeoDataFrame.

        Examples:
            # retrieve street view with building footprints (OSM)
            gtd = GeoTaggedData()
            gtd.getBuildingFootprints(bbox=(-83.235572,42.348092,-83.235154,42.348806))
            gtd.get_svi_from_locations(key="your Mapillary token")

            # locations - a nested list of coordinates
            gtd = GeoTaggedData(location=[[-83.235572,42.348092],[-83.235154,42.348806]])
            # locations - a dataframe with columns "longitude" and "latitude"
            df = pd.Dataframe({"longitude":[-83.235572, -83.235154], "latitude":[42.348092, 42.348806]})
            gtd = GeoTaggedData(locations=df)
        '''

        self.images = None
        self.locations = locations
        self.units = units
        if locations is not None and units is None:
            self.construct_units()

        # NOTE: each must be its own dict literal — chained assignment would
        # alias all three names to the SAME underlying dict object.
        def _empty_payload():
            return {'loc_id': [], 'id': [], 'data': [], 'path': []}

        self.svis = _empty_payload()
        self.photos = _empty_payload()
        self.audios = _empty_payload()

        self.svi_metadata = None
        self.photo_metadata = None
        self.audio_metadata = None
        self.plot = None

    def construct_units(self):
        if isinstance(self.locations, list):
            if not self.locations or not isinstance(self.locations[0], (list, tuple)):
                raise ValueError(
                    "locations as a list must be a nested list of "
                    "[longitude, latitude] pairs."
                )
            xs = [loc[0] for loc in self.locations]
            ys = [loc[1] for loc in self.locations]
            geometry = gpd.points_from_xy(xs, ys)
            id_df = pd.DataFrame({'loc_id': list(range(len(self.locations)))})
        elif isinstance(self.locations, dict):
            if 'longitude' in self.locations and 'latitude' in self.locations:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': list(range(len(self.locations['longitude'])))})
            else:
                raise ValueError(
                    "locations as a dict must be keyed by 'longitude' and 'latitude'."
                )
        elif isinstance(self.locations, pd.DataFrame):
            if 'longitude' in self.locations.columns and 'latitude' in self.locations.columns:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': list(range(len(self.locations['longitude'])))})
            else:
                raise ValueError(
                    "locations as a DataFrame must include columns 'longitude' and 'latitude'."
                )
        else:
            raise TypeError(
                f"Unsupported locations type: {type(self.locations).__name__}. "
                "Use list[list], dict, or pandas.DataFrame."
            )
        self.units = gpd.GeoDataFrame(id_df, geometry=geometry, crs="EPSG:4326")
        return None

    def getBuildings(self,
                     bbox: list | tuple = None,
                     source: str = 'osm',
                     min_area: float | int = 0,
                     max_area: float | int = None,
                     random_sample: int = None,
                     gba_path: str = None,
                     gba_cache_dir: str = None) -> None:
        '''
            Extract buildings from a public source using the bbox.

            Args:
                bbox (list or tuple): The bounding box (min_lon, min_lat, max_lon, max_lat).
                source (str): One of:

                  * ``'osm'`` (default) — OpenStreetMap via the Overpass API.
                    Footprints only; no height.  Use only for small areas:
                    Overpass commonly times out for city-scale or larger
                    building queries.  For large-area outdoor filtering, prefer
                    ``filter_outdoor(building_source='overture')``.
                  * ``'microsoft'`` — Microsoft GlobalMLBuildingFootprints
                    (Bing). Footprints only; no height.
                  * ``'globfp3d'`` — `3D-GloBFP <https://zenodo.org/records/15487037>`_
                    (Che et al., ESSD 2024). **Footprints + per-building
                    height.** Auto-fetches from Zenodo + Figshare and
                    caches under ``gba_cache_dir`` (default
                    ``~/.cache/opensip/globfp3d``). Pass ``gba_path``
                    to load from a pre-downloaded local file. The
                    resulting ``self.units`` keeps a ``height_m`` column.
                  * ``'gba'`` — `GlobalBuildingAtlas <https://github.com/zhu-xlab/GlobalBuildingAtlas>`_
                    (Zhu et al., ESSD 2025). **A different dataset from
                    3D-GloBFP** — hosted on HuggingFace + mediaTUM. Auto-
                    fetches polygon tiles from
                    ``zhu-xlab/GBA.LoD1`` using ``representative/lod1.geojson``
                    as the manifest. Caches under ``~/.cache/opensip/gba``.
                    NOTE: GBA polygons ship without per-row height — heights
                    live in a separate mediaTUM dataset (`m1837832`) which
                    isn't joined yet (tracking issue in CHANGELOG). For
                    per-building height today, use ``source='globfp3d'``.

                min_area (float or int): The minimum area in m².
                max_area (float or int): The maximum area in m².
                random_sample (int): If set, randomly subsample to this many.
                gba_path (str): Optional. Path to a pre-downloaded file
                    (skips network calls). Used by both ``'globfp3d'`` and
                    ``'gba'`` sources.
                gba_cache_dir (str): Optional. Where to cache auto-fetched
                    files. Defaults differ per source
                    (``~/.cache/opensip/globfp3d`` or ``.../gba``).
        '''

        if source not in ('osm', 'microsoft', 'gba', 'globfp3d'):
            raise ValueError(
                f'Unsupported building source {source!r}; '
                f'choose from "osm", "microsoft", "globfp3d", or "gba".'
            )

        if source == 'osm':
            buildings = getOSMbuildings(bbox, min_area, max_area)
        elif source == 'microsoft':
            buildings = getGlobalMLBuilding(bbox, min_area, max_area)
        elif source == 'globfp3d':
            buildings = getGloBFP3DBuildings(
                bbox, gba_path=gba_path, min_area=min_area,
                max_area=max_area, cache_dir=gba_cache_dir,
            )
        else:  # 'gba'
            buildings = getGBABuildings(
                bbox, gba_path=gba_path, min_area=min_area,
                max_area=max_area, cache_dir=gba_cache_dir,
            )

        if buildings is None or buildings.empty:
            if source == 'osm':
                logger.warning(
                    "No buildings found in the bounding box. "
                    "Check https://overpass-turbo.eu/ for areas with buildings."
                )
            elif source == 'microsoft':
                logger.warning(
                    "No buildings found in the bounding box. "
                    "Check https://github.com/microsoft/GlobalMLBuildingFootprints "
                    "for areas with buildings."
                )
            elif source == 'globfp3d':
                logger.warning(
                    "No 3D-GloBFP buildings found in the bounding box %s "
                    "(local: %s).", bbox, gba_path,
                )
            else:
                logger.warning(
                    "No GBA buildings found in the bounding box %s "
                    "(local: %s).", bbox, gba_path,
                )
            return None
        if random_sample is not None:
            buildings = buildings.sample(random_sample)
        self.units = buildings.to_crs(4326)
        with_height = (
            int(buildings["height_m"].notna().sum())
            if "height_m" in buildings.columns else 0
        )
        if with_height:
            logger.info(
                "%d buildings found in the bounding box (%d with height_m).",
                len(buildings), with_height,
            )
        else:
            logger.info("%d buildings found in the bounding box.", len(buildings))
        return None

    def get_svi_from_locations(self,
                               id_column:str=None,
                               distance:int = 50,
                               key: str = None,
                               source: str = "mapillary",
                               pano: bool = True, reoriented: bool = True,
                               multi_num: int = 1, interval: int = 1,
                               fov: int | float | str = 80, heading: int = None, pitch: int = 5,
                               height: int = 500, width: int = 700,
                               year: list | tuple = None, season: str = None, time_of_day: str = 'day',
                               fov_margin: float = 0.10,
                               fov_min: float = 30.0,
                               fov_max: float = 120.0,
                               building_height: float = 9.0,
                               silent: bool = True,
                               checkpoint_path: str | None = None,
                               reset: bool = False):
        """
            get_svi_from_locations

            Retrieve the closest street view image(s) near each coordinate.
            The street view image will be reoriented to look at the coordinate when
            ``reoriented=True`` (Mapillary) or always (Google).

            Args:
                id_column (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): The max distance in meters between the centroid and the street view.
                key (str): API access token for the chosen source.
                    Mapillary — pass token or set env var ``MAPILLARY_API_KEY``.
                    Google    — pass token or set env var ``GOOGLE_STREETVIEW_API_KEY``.
                source (str): Street view data source. One of ``"mapillary"`` (default)
                    or ``"google"``.
                pano (bool): Whether to search for pano street view images only.
                    Mapillary only — ignored for Google. (Default is True)
                reoriented (bool): Whether to reorient and crop street view images.
                    Mapillary only — Google always faces the target. (Default is True)
                multi_num (int): The number of multiple SVIs.
                    Mapillary only — Google always returns 1. (Default is 1)
                interval (int): The interval in meters between each SVI.
                    Mapillary only. (Default is 1)
                fov (int | float | str): Field of view in degrees (default 80). Pass
                    ``'auto'`` (with ``reoriented=True``) to size the FOV per image
                    so the building footprint at each location is just framed.
                    The polygon used is each unit's ``row.geometry`` from
                    ``self.units`` — i.e. the building footprint loaded by
                    ``getBuildings()``. Falls back to a distance-based heuristic
                    if a unit's geometry is a point. Mapillary only for ``'auto'``.
                heading (int): Camera heading in degrees. If None, it will be computed based on the house orientation.
                pitch (int): Camera pitch angle. (Default is 5).
                height (int): Height in pixels of the returned image. (Default is 500).
                width (int): Width in pixels of the returned image. (Default is 700).
                year (list[str], optional): Year of data (start year, end year).
                    Mapillary only — ignored for Google with a warning.
                season (str, optional): Season of data. One of ["spring","summer","fall","autumn","winter"].
                    Mapillary only — ignored for Google with a warning.
                time_of_day (str, optional): Time of data. One of ["day","night"] (Default is 'day').
                    Mapillary only — ignored for Google with a warning.
                fov_margin (float): When ``fov='auto'``, fractional padding added to the
                    auto-computed FOV (0.10 = +10%). Default 0.10. Mapillary only.
                fov_min (float): Lower clamp for ``fov='auto'`` (degrees). Default 30°.
                    Mapillary only.
                fov_max (float): Upper clamp for ``fov='auto'`` (degrees). Default 120°.
                    Mapillary only.
                building_height (float): Assumed building height in meters used by
                    ``fov='auto'`` (default 9 m, ~3 stories). Mapillary only.
                silent (bool): If True, do not show error traceback (Default is True).
                reset (bool): If True, discard previously collected street-view
                    images before this fetch. Default False appends to existing
                    SVIs and removes duplicate image rows after collection.
                checkpoint_path (str, optional): Path to a JSONL file for
                    resume-safe checkpointing. When provided, each successfully
                    fetched location is written to the file immediately, and
                    base64 images are saved to a companion directory
                    (``<checkpoint_stem>_files/`` next to the JSONL) so the
                    session can be resumed after a crash.
            """

        def _empty_svis() -> dict:
            return {'loc_id': [], 'id': [], 'data': [], 'path': []}

        def _payload_len(payload: dict) -> int:
            return max((len(v) for v in payload.values() if isinstance(v, list)), default=0)

        def _normalize_svi_payload(payload: dict) -> dict:
            n = _payload_len(payload)
            out = {
                'loc_id': list(payload.get('loc_id', [])),
                'id': list(payload.get('id', [])),
                'data': list(payload.get('data', [])),
                'path': list(payload.get('path', [])),
            }
            for key in ('loc_id', 'id', 'data', 'path'):
                out[key].extend([None] * (n - len(out[key])))
            return out

        def _dedupe_svi_state() -> None:
            self.svis = _normalize_svi_payload(self.svis)
            n = _payload_len(self.svis)
            if n:
                svi_df = pd.DataFrame(self.svis)
                subset = [c for c in ('id', 'data') if c in svi_df.columns]
                if subset:
                    svi_df = svi_df.drop_duplicates(subset=subset, keep='first')
                else:
                    svi_df = svi_df.drop_duplicates(keep='first')
                self.svis = svi_df.to_dict(orient='list')

            if self.svi_metadata is not None and not self.svi_metadata.empty:
                meta = self.svi_metadata.copy()
                subset = [
                    c for c in ('id', 'url', 'image_url', 'captured_at')
                    if c in meta.columns
                ]
                if subset:
                    meta = meta.drop_duplicates(subset=subset, keep='first')
                else:
                    meta = meta.drop_duplicates(keep='first')
                self.svi_metadata = meta.reset_index(drop=True)

        existing_svis = _empty_svis()
        existing_svi_metadata = None
        if not reset:
            existing_svis = _normalize_svi_payload(getattr(self, 'svis', {}) or {})
            existing_svi_metadata = (
                self.svi_metadata.copy()
                if self.svi_metadata is not None and not self.svi_metadata.empty
                else None
            )

        self.svis = existing_svis

        if id_column is None:
            id_column = 'loc_id'
            if id_column not in self.units.columns:
                self.units[id_column] = [i for i in range(len(self.units))]
        # Resolve API key once with env var fallback.
        # Use the appropriate env var depending on the source.
        _env_var = "GOOGLE_STREETVIEW_API_KEY" if source.lower() == "google" else "MAPILLARY_API_KEY"
        resolved_key = key or os.getenv(_env_var)
        if not resolved_key:
            raise ValueError(
                f"Missing API key for source={source!r}. "
                f"Pass key=... or set env var {_env_var}."
            )
        # ``time_of_day='day'`` is the Mapillary default for historical reasons.
        # Google does not support time filters, so avoid warning when the caller
        # did not intentionally request one.
        time_of_day_for_source = (
            None
            if source.lower() == "google" and time_of_day == "day"
            else time_of_day
        )

        # ── resume from checkpoint ────────────────────────────────────────
        # The checkpoint records which loc_ids have been fetched and stores
        # the raw fetched data (base64 strings or URLs) so the full
        # self.svis payload can be restored without re-hitting the API.
        # File downloading is handled separately by download_to_dir().
        done_ids: set = set()
        if checkpoint_path is not None:
            done_ids, ckpt_records = load_collection_checkpoint(checkpoint_path)
            restored_svis, restored_frames = restore_svis_from_checkpoint(ckpt_records)
            restored_svis = _normalize_svi_payload(restored_svis)
            for _key, _vals in restored_svis.items():
                self.svis.setdefault(_key, [None] * _payload_len(self.svis))
                self.svis[_key] += _vals
            self.svis = _normalize_svi_payload(self.svis)
        else:
            restored_frames = []

        # Accumulate per-location frames and concat once for O(n) instead of O(n^2)
        frames: list[pd.DataFrame] = []
        if existing_svi_metadata is not None:
            frames.append(existing_svi_metadata)
        frames.extend(restored_frames)
        skip_count = 0
        for _index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]

            # Skip already-checkpointed locations
            if loc_id in done_ids:
                continue

            try:
                # Pass the unit's polygon to enable fov='auto' framing.
                # Points (no `.exterior`) become None, so getSV will fall
                # back to its distance-based heuristic.
                target_poly = getattr(row.geometry, "exterior", None)
                target_poly = row.geometry if target_poly is not None else None

                # Per-building height: if the units GeoDataFrame has a
                # height_m column (e.g. from source='gba'), use that row's
                # value. Fall back to the global ``building_height``
                # parameter when the row is missing/NaN.
                row_height = building_height
                if "height_m" in self.units.columns:
                    rh = row.get("height_m")
                    try:
                        if rh is not None and not pd.isna(rh) and float(rh) > 0:
                            row_height = float(rh)
                    except (TypeError, ValueError):
                        pass

                svis, output_df = getSV(
                    [row.geometry.centroid.x, row.geometry.centroid.y],
                    loc_id=loc_id,
                    distance=distance,
                    key=resolved_key,
                    source=source,
                    pano=pano,
                    reoriented=reoriented,
                    multi_num=multi_num,
                    interval=interval,
                    fov=fov, heading=heading, pitch=pitch,
                    height=height, width=width,
                    year=year, season=season, time_of_day=time_of_day_for_source,
                    target_polygon=target_poly,
                    fov_margin=fov_margin, fov_min=fov_min, fov_max=fov_max,
                    building_height=row_height,
                    silent=silent,
                )
                if svis is None:
                    skip_count += 1
                    continue

                self.svis['data'] += svis
                self.svis['loc_id'] += output_df['loc_id'].tolist()
                self.svis['id'] += output_df['id'].tolist()
                self.svis.setdefault('path', [])
                self.svis['path'] += [None] * len(output_df)

                # ── checkpoint: record the fetched data ──────────────────
                # Stores raw fetched data (base64 or URLs) so the session
                # can be fully restored without re-hitting the Mapillary API.
                # File downloading remains the responsibility of download_to_dir().
                if checkpoint_path is not None:
                    append_collection_checkpoint(checkpoint_path, {
                        'loc_id': loc_id,
                        'ids': output_df['id'].tolist(),
                        'paths': [],           # populated later by download_to_dir()
                        'data': svis,          # base64 strings or URLs as returned by getSV
                        'metadata': output_df.to_dict(orient='records'),
                    })

                frames.append(output_df)
            except Exception as e:
                if not silent:
                    logger.warning(
                        'skipping %s: %s',
                        [row.geometry.centroid.x, row.geometry.centroid.y], e,
                    )
                skip_count += 1
                continue
        self.svi_metadata = pd.concat(frames, ignore_index=True) if frames else None
        _dedupe_svi_state()
        if skip_count > 0:
            logger.info(
                'Collected data for %d locations; skipped %d (no data found).',
                len(self.units) - skip_count, skip_count,
            )
        return None

    def get_photo_from_location(self,
                                id_column:str=None,
                                distance: int | None = None,
                                key: str = None,
                                query: str | list[str] = None,
                                geo_context: int = None,
                                tag: str | list[str] = None,
                                max_return: int = 1,
                                year: list | tuple = None,
                                season: str = None,
                                time_of_day: str = None,
                                remove_selfies: bool = False,
                                exclude_personal_photo: bool | None = None,
                                selfie_model_path: str | None = None,
                                exclude_from_location:int = None,
                                silent = True,
                                checkpoint_path: str | None = None,
                                ):
        '''
            get_photo_from_location

            Retrieve geotagged photos from Flickr

            Args:
                id_column: (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): Search radius in meters (converted to km; Flickr radius max is 32 km).
                key (str): Flickr API key. If None, reads env var FLICKR_API_KEY.
                query (str, optional): Query string to search for.
                geo_context (int, optional): Specify whether a geotagged photo was taken indoors or outdoors. 0: Not defined; 1: Indoors; 2: Outdoors. (Default is None)
                tag (str | list[str]): Tag string or list of tags (comma-separated). Acts as a "limiting agent" for geo queries.
                max_return (int): Number of photos to return (after filters).
                year: [Y] or (Y,) or (Y1, Y2) inclusive. Filters by taken date range.
                season (str): One of {"spring","summer","fall","autumn","winter"} (post-filter by taken month).
                time_of_day (str): One of {"morning","afternoon","evening","night"} (post-filter by taken hour).
                remove_selfies (bool): If True, remove Flickr photos that appear
                    to contain a visible face. This is useful for avoiding
                    selfie/person-focused photos in sound-image pair datasets.
                    Default False.
                exclude_personal_photo (bool, optional): Deprecated alias for
                    ``remove_selfies``. If provided, it overrides
                    ``remove_selfies``.
                selfie_model_path (str, optional): Path to an OpenCV YuNet ONNX
                    face detector. If omitted, openSIP uses OpenCV's bundled
                    Haar cascade detector.
                exclude_from_location (int, optional): Drop retrieved data with a distance from the given location.
                silent (bool): If True, do not show error traceback (Default is True).
        '''
        if exclude_personal_photo is not None:
            warnings.warn(
                "exclude_personal_photo is deprecated; use remove_selfies instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            remove_selfies = exclude_personal_photo

        self.photos = {
            'loc_id': [],
            'id': [],
            'data': [],
            'path': [],
        }
        self.photo_metadata = None

        if id_column is None:
            id_column = 'loc_id'
            if id_column not in self.units.columns:
                self.units[id_column] = list(range(len(self.units)))

        # ── resume from checkpoint ────────────────────────────────────────
        done_ids_ph: set = set()
        if checkpoint_path is not None:
            done_ids_ph, ckpt_records_ph = load_collection_checkpoint(checkpoint_path)
            restored_photos, restored_frames_ph = restore_photos_from_checkpoint(ckpt_records_ph)
            self.photos = restored_photos
        else:
            restored_frames_ph = []

        frames: list[pd.DataFrame] = list(restored_frames_ph)
        skip_count = 0
        for _index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]
            if loc_id in done_ids_ph:
                continue
            try:
                output_df = getPhoto([row.geometry.centroid.x, row.geometry.centroid.y],
                                     loc_id,
                                     distance,
                                     key,
                                     query,
                                     geo_context,
                                     tag,
                                     max_return,
                                     year,
                                     season,
                                     time_of_day,
                                     exclude_from_location,
                                     output_df=True)
                if output_df is None or output_df.empty:
                    skip_count += 1
                    continue

                if remove_selfies:
                    drop_list = []
                    for ind, r in output_df.iterrows():
                        try:
                            if selfie_model_path is None:
                                with as_file(get_yunet_model_path()) as model_path:
                                    is_selfie = is_selfie_photo(
                                        model_path=model_path,
                                        img_url=r['url'],
                                    )
                            else:
                                is_selfie = is_selfie_photo(
                                    model_path=selfie_model_path,
                                    img_url=r['url'],
                                )
                        except Exception as face_err:
                            if not silent:
                                logger.warning(
                                    "selfie filter skipped photo %s: %s",
                                    r.get('id', ind),
                                    face_err,
                                )
                            is_selfie = False
                        if is_selfie:
                            drop_list += [ind]
                    if len(drop_list) > 0:
                        output_df.drop(drop_list, axis=0, inplace=True)
                        output_df.reset_index(drop=True, inplace=True)
                        if len(output_df) == 0:
                            continue

                self.photos['loc_id'] += output_df['loc_id'].tolist()
                self.photos['data'] += output_df['url'].tolist()
                self.photos['id'] += output_df['id'].tolist()
                self.photos['path'] += [None] * len(output_df)

                if checkpoint_path is not None:
                    append_collection_checkpoint(checkpoint_path, {
                        'loc_id': loc_id,
                        'ids': output_df['id'].tolist(),
                        'paths': [],
                        'data': output_df['url'].tolist(),
                        'metadata': output_df.to_dict(orient='records'),
                    })

                frames.append(output_df)
            except Exception as e:
                if not silent:
                    logger.warning("photo fetch error: %s", e)
                skip_count += 1
                continue
        self.photo_metadata = pd.concat(frames, ignore_index=True) if frames else None
        if skip_count > 0:
            logger.info(
                'Collected data for %d locations; skipped %d (no data found).',
                len(self.units) - skip_count, skip_count,
            )
        return None

    def get_sound_from_location(self,
                                id_column: str = None,
                                distance: int | None = None,
                                source: str = 'freesound',
                                key: str = None,
                                catalog: str | pd.DataFrame = None,
                                query: str | list[str] = None,
                                tag: str | list[str] = None,
                                max_return: int = 1,
                                year: list | tuple = None,
                                season: str = None,
                                time_of_day: str = None,
                                duration: int = None,
                                exclude_from_location: int = None,
                                slice_duration: int = None,
                                slice_max_num: int = None,
                                probe_durations: bool = True,
                                probe_workers: int = 8,
                                silent: bool = True,
                                checkpoint_path: str | None = None,
                                boundary=None,
                                bbox: tuple | None = None,
                                reset: bool = False,
                                ):

        '''
            get_sound_from_location

            Retrieve geotagged sound recordings from Freesound (default) or
            from a Radio Aporee catalog you provide as a CSV / DataFrame.

            Args:
                id_column (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int, optional): Search radius in meters for point/unit
                    mode. Ignored when ``bbox`` is provided; bbox mode computes
                    the internal radius needed to cover the box.
                source (str): one of {"freesound", "aporee"} (Default is "freesound").
                key (str): Freesound API key. Required only when source="freesound".
                    If None, reads env var FREESOUND_API_KEY.
                catalog (str | pandas.DataFrame): Required only when source="aporee".
                    Path to a CSV or an in-memory DataFrame containing at minimum the columns
                    ``url``, ``latitude``, ``longitude``. Optional columns recognised:
                    ``id``/``identifier``, ``name``/``title``, ``description``, ``tags``,
                    ``created`` (ISO timestamp), ``duration_s``.
                query (str, optional): Query string to search for.
                tag (str | list[str]): tag string or list of tags (used as filters).
                max_return (int): number of sounds to return (after post-filters).
                year (int | list): [Y] or (Y,) or (Y1, Y2) inclusive (filters by upload date "created").
                season (str): one of {"spring","summer","fall","autumn","winter"} (post-filter by created month).
                time_of_day (str): one of {"morning","afternoon","evening","night"} (post-filter by created hour).
                duration (int | list[int] | tuple[int]): maximum duration in seconds (<= duration). If you want a range, pass a tuple/list (min,max).
                exclude_from_location (int, optional): Drop retrieved data with a distance from the given location.
                slice_duration (int, optional): Split the original sound signal into clips with the given duration.
                slice_max_num (int, optional): Maximum number of clips sliced from the original sound signal.
                probe_durations (bool): Aporee-only. When ``slice_duration`` is set
                    but the catalog has no ``duration_s`` column, probe each
                    selected URL once to learn its length. Set False to skip
                    slicing instead. Default True.
                silent (bool): If True, do not show error traceback (Default is True).
                reset (bool): If True, discard previously collected sounds before
                    this fetch. Default False appends to existing sounds and removes
                    duplicate audio rows after collection.
                boundary (geopandas.GeoDataFrame | geopandas.GeoSeries | shapely geometry, optional):
                    Polygon boundary to collect sounds within.  If a GeoDataFrame
                    or GeoSeries has a CRS, it is reprojected to EPSG:4326 before
                    filtering.  ``boundary`` and ``bbox`` are mutually exclusive.
                    Boundary mode first queries the boundary bounds, then keeps
                    only recordings whose GPS points fall inside the actual
                    polygon.  No ``distance`` parameter is needed.
                bbox (tuple, optional): ``(min_lon, min_lat, max_lon, max_lat)``.
                    When provided, sounds are collected by filtering directly
                    on their GPS coordinates. ``bbox`` takes precedence over
                    existing ``self.units``.

                    * **Aporee** — the catalog is pre-filtered to rows whose
                      ``latitude``/``longitude`` fall within the bbox bounds,
                      then processed in a single pass.  No ``distance`` parameter
                      is needed.
                    * **Freesound / other API sources** — a single API call is
                      made from the bbox centre with a circumscribed radius
                      (half the diagonal), and the results are post-filtered to
                      the exact bbox coordinate bounds.

                    After collection ``self.units`` is rebuilt from the actual
                    sound coordinates (one point per unique recording) so that
                    a subsequent ``get_svi_from_locations()`` call fetches
                    images at the correct positions.
        '''

        def _empty_audios(with_slice: bool = False) -> dict:
            payload = {'loc_id': [], 'id': [], 'data': [], 'path': []}
            if with_slice:
                payload['slice'] = []
            return payload

        def _payload_len(payload: dict) -> int:
            return max((len(v) for v in payload.values() if isinstance(v, list)), default=0)

        def _normalize_audio_payload(payload: dict, with_slice: bool | None = None) -> dict:
            n = _payload_len(payload)
            out = {
                'loc_id': list(payload.get('loc_id', [])),
                'id': list(payload.get('id', [])),
                'data': list(payload.get('data', [])),
                'path': list(payload.get('path', [])),
            }
            for key in ('loc_id', 'id', 'data', 'path'):
                out[key].extend([None] * (n - len(out[key])))
            include_slice = (
                with_slice
                if with_slice is not None
                else ('slice' in payload and any(s is not None for s in payload.get('slice', [])))
            )
            if include_slice:
                out['slice'] = list(payload.get('slice', []))
                out['slice'].extend([None] * (n - len(out['slice'])))
            return out

        def _dedupe_audio_state() -> None:
            """Drop duplicate audio payload rows and metadata rows in-place."""
            self.audios = _normalize_audio_payload(self.audios)
            n = _payload_len(self.audios)
            if n:
                aud_df = pd.DataFrame(self.audios)
                if 'slice' in aud_df.columns:
                    aud_df['_slice_key'] = aud_df['slice'].apply(
                        lambda s: tuple(s) if isinstance(s, list) else s
                    )
                    subset = ['id', 'data', '_slice_key']
                else:
                    subset = ['id', 'data']
                aud_df = aud_df.drop_duplicates(subset=subset, keep='first').drop(
                    columns=['_slice_key'], errors='ignore'
                )
                deduped = aud_df.to_dict(orient='list')
                if 'slice' in deduped and all(s is None for s in deduped['slice']):
                    deduped.pop('slice')
                self.audios = deduped

            if self.audio_metadata is not None and not self.audio_metadata.empty:
                meta = self.audio_metadata.copy()
                subset = [
                    c for c in ('id', 'preview-hq-mp3', 'url')
                    if c in meta.columns
                ]
                if subset:
                    meta = meta.drop_duplicates(subset=subset, keep='first')
                else:
                    meta = meta.drop_duplicates(keep='first')
                self.audio_metadata = meta.reset_index(drop=True)

        def _coerce_boundary_geometry(boundary_obj):
            if boundary_obj is None:
                return None
            if hasattr(boundary_obj, "geometry"):
                boundary_gdf = boundary_obj
                if getattr(boundary_gdf, "crs", None) is not None:
                    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
                geom_series = boundary_gdf.geometry
                if geom_series.empty:
                    raise ValueError("boundary must contain at least one geometry.")
                geoms = [g for g in geom_series if g is not None and not g.is_empty]
                if not geoms:
                    raise ValueError("boundary geometry is empty.")
                geom = geoms[0]
                for next_geom in geoms[1:]:
                    geom = geom.union(next_geom)
            elif hasattr(boundary_obj, "geom_type") and hasattr(boundary_obj, "bounds"):
                geom = boundary_obj
            else:
                raise TypeError(
                    "boundary must be a GeoDataFrame, GeoSeries, or shapely geometry."
                )
            if geom is None or geom.is_empty:
                raise ValueError("boundary geometry is empty.")
            return geom

        boundary_geom = None
        if boundary is not None:
            if bbox is not None:
                raise ValueError("Pass either boundary or bbox, not both.")
            boundary_geom = _coerce_boundary_geometry(boundary)
            bbox = tuple(float(v) for v in boundary_geom.bounds)

        existing_audios = _empty_audios(with_slice=slice_duration is not None)
        existing_audio_metadata = None
        if not reset:
            existing_audios = _normalize_audio_payload(
                getattr(self, 'audios', {}) or {},
                with_slice=(
                    True
                    if slice_duration is not None or 'slice' in (getattr(self, 'audios', {}) or {})
                    else None
                ),
            )
            existing_audio_metadata = (
                self.audio_metadata.copy()
                if self.audio_metadata is not None and not self.audio_metadata.empty
                else None
            )

        self.audios = existing_audios

        # ── Aporee: auto-fetch catalog when none is provided ─────────────────
        # Internet Archive's Scrape API does not reliably index the lat/lon
        # metadata fields for the Aporee collection, so passing a tight bbox
        # as a Solr query almost always returns 0 results.  Instead we fetch
        # without a geographic filter and let getSoundAporee's per-point
        # Haversine distance test handle the spatial narrowing.
        _src_lower = (source or 'freesound').lower()
        if _src_lower == 'aporee' and catalog is None:
            try:
                print(
                    "source='aporee' — fetching catalog from Internet Archive "
                    "(no API key required; may take a moment)…"
                )
                catalog = fetch_aporee_catalog()   # no bbox filter — IA Solr lat/lon unreliable
                print(f"  Aporee catalog: {len(catalog):,} recordings available worldwide.")
            except Exception as _cat_err:
                import warnings as _warnings
                _warnings.warn(
                    f"Could not auto-fetch Aporee catalog: {_cat_err}. "
                    "Pass catalog= explicitly or check your internet connection.",
                    stacklevel=2,
                )
                catalog = None

        # ── decide execution path: bbox-direct vs per-building loop ──────────
        _from_bbox = False
        _distance = 50 if distance is None else distance

        # ── resume from checkpoint ────────────────────────────────────────────
        done_ids_au: set = set()
        if checkpoint_path is not None:
            done_ids_au, ckpt_records_au = load_collection_checkpoint(checkpoint_path)
            restored_audios, restored_frames_au = restore_audios_from_checkpoint(ckpt_records_au)
            # Guard: if every restored id is NaN the checkpoint was written by
            # a broken previous run (e.g. before URL or id fixes).  Discard it
            # so we re-fetch rather than perpetuating broken data.
            _restored_ids = restored_audios.get('id', [])
            _all_nan = _restored_ids and all(
                (isinstance(_i, float) and _i != _i)   # NaN check: NaN != NaN
                for _i in _restored_ids
            )
            if _all_nan:
                logger.warning(
                    "Checkpoint at %s contains only NaN ids — discarding stale "
                    "data and running a fresh fetch. Delete the file to suppress "
                    "this message.",
                    checkpoint_path,
                )
                done_ids_au = set()
                restored_audios = {
                    'loc_id': [], 'id': [], 'data': [], 'path': [],
                    **({'slice': []} if slice_duration is not None else {}),
                }
                restored_frames_au = []
            restored_audios = _normalize_audio_payload(
                restored_audios,
                with_slice=(
                    True
                    if slice_duration is not None or 'slice' in self.audios or 'slice' in restored_audios
                    else None
                ),
            )
            for _key, _vals in restored_audios.items():
                self.audios.setdefault(_key, [None] * _payload_len(self.audios))
                self.audios[_key] += _vals
            self.audios = _normalize_audio_payload(
                self.audios,
                with_slice=(
                    True
                    if slice_duration is not None or 'slice' in self.audios
                    else None
                ),
            )
            # restore_audios_from_checkpoint only adds 'slice' when the
            # existing records already contain slices.  If the checkpoint is
            # brand-new (empty) or was written without slices, the key is
            # absent — add it now so _absorb() never gets a KeyError.
            if slice_duration is not None and 'slice' not in self.audios:
                self.audios['slice'] = []
        else:
            restored_frames_au = []

        frames: list[pd.DataFrame] = []
        if existing_audio_metadata is not None:
            frames.append(existing_audio_metadata)
        frames.extend(restored_frames_au)

        # ── helper: expand one output_df into self.audios and frames ─────────
        def _absorb(output_df: pd.DataFrame, loc_id) -> None:
            """Expand slices and append to self.audios / frames in-place."""
            if output_df is None or output_df.empty:
                return
            if slice_duration is not None and 'slice' in output_df.columns:
                slice_list  = output_df['slice'].tolist()
                loc_id_list = output_df['loc_id'].tolist()
                data_list   = output_df['preview-hq-mp3'].tolist()
                id_list     = output_df['id'].tolist()
                flattened   = [s for subs in slice_list for s in subs]
                rep_loc, rep_data, rep_id = [], [], []
                for subs, lid, d, sid in zip(slice_list, loc_id_list, data_list, id_list):
                    n = len(subs)
                    rep_loc.extend([lid] * n)
                    rep_data.extend([d] * n)
                    rep_id.extend([sid] * n)
                self.audios['loc_id'] += rep_loc
                self.audios['data']   += rep_data
                self.audios['id']     += rep_id
                self.audios.setdefault('path', [])
                self.audios['path']   += [None] * len(rep_id)
                self.audios['slice']  += flattened
                if checkpoint_path is not None:
                    append_collection_checkpoint(checkpoint_path, {
                        'loc_id': loc_id, 'ids': rep_id, 'paths': [],
                        'data': rep_data, 'slices': flattened,
                        'metadata': output_df.to_dict(orient='records'),
                    })
            else:
                self.audios['loc_id'] += output_df['loc_id'].tolist()
                self.audios['data']   += output_df['preview-hq-mp3'].tolist()
                self.audios['id']     += output_df['id'].tolist()
                self.audios.setdefault('path', [])
                self.audios['path'] += [None] * len(output_df)
                if 'slice' in self.audios:
                    self.audios['slice'] += [None] * len(output_df)
                if checkpoint_path is not None:
                    append_collection_checkpoint(checkpoint_path, {
                        'loc_id': loc_id, 'ids': output_df['id'].tolist(),
                        'paths': [], 'data': output_df['preview-hq-mp3'].tolist(),
                        'slices': None,
                        'metadata': output_df.to_dict(orient='records'),
                    })
            frames.append(output_df)

        # ═══════════════════════════════════════════════════════════════════════
        # PATH A — bbox mode: filter directly by coordinate bounds, no grid.
        # Boundary mode uses the same retrieval path with the boundary bounds,
        # then applies an exact polygon filter before absorption.
        #
        #   Aporee  → pre-filter the catalog to rows within the bbox, then pass
        #             the circumscribed radius internally so getSoundAporee's
        #             point-radius helper admits all pre-filtered rows.
        #
        #   Freesound / other API → single call from bbox centre with circumscribed
        #             radius, then post-filter results to within the bbox bounds.
        # ═══════════════════════════════════════════════════════════════════════
        if bbox is not None:
            _from_bbox = True
            import math as _math
            from shapely.geometry import Point as _Point

            _lon_min, _lat_min, _lon_max, _lat_max = bbox
            _center_lon = (_lon_min + _lon_max) / 2.0
            _center_lat = (_lat_min + _lat_max) / 2.0
            # Half-diagonal of the bbox in metres → circumscribed radius
            _dlat_m = (_lat_max - _lat_min) / 2.0 * 111_320
            _dlon_m = (_lon_max - _lon_min) / 2.0 * 111_320 * _math.cos(
                _math.radians(_center_lat)
            )
            _circum_radius = int(_math.ceil(_math.sqrt(_dlat_m ** 2 + _dlon_m ** 2))) + 1

            def _filter_to_area(df: pd.DataFrame | None) -> pd.DataFrame | None:
                if df is None or df.empty:
                    return df
                if 'latitude' not in df.columns or 'longitude' not in df.columns:
                    return df
                filtered = df[
                    (df['latitude'] >= _lat_min)
                    & (df['latitude'] <= _lat_max)
                    & (df['longitude'] >= _lon_min)
                    & (df['longitude'] <= _lon_max)
                ].copy().reset_index(drop=True)
                if boundary_geom is not None and not filtered.empty:
                    inside = [
                        boundary_geom.covers(_Point(float(lon), float(lat)))
                        for lat, lon in zip(filtered['latitude'], filtered['longitude'])
                    ]
                    filtered = filtered.loc[inside].copy().reset_index(drop=True)
                if not filtered.empty:
                    filtered['loc_id'] = 0
                return filtered

            # Skip if checkpoint already has data
            if not done_ids_au:
                try:
                    if _src_lower == 'aporee':
                        # ── coordinate-based catalog filter ──────────────────
                        _cat_df = (
                            pd.read_csv(catalog) if isinstance(catalog, str)
                            else catalog
                        )
                        if _cat_df is None or _cat_df.empty:
                            output_df = None
                        else:
                            # Normalize lat/lon column names — the reference
                            # aporee_metadata.py script writes 'lat'/'lon'
                            # while fetch_aporee_catalog() writes 'latitude'/
                            # 'longitude'. Both should work.
                            _col_aliases = {'lat': 'latitude', 'lon': 'longitude'}
                            for _src, _dst in _col_aliases.items():
                                if _src in _cat_df.columns and _dst not in _cat_df.columns:
                                    _cat_df = _cat_df.rename(columns={_src: _dst})
                            if 'latitude' not in _cat_df.columns or 'longitude' not in _cat_df.columns:
                                raise ValueError(
                                    "Aporee catalog must have 'latitude'/'longitude' "
                                    "(or 'lat'/'lon') columns for bbox filtering."
                                )
                            _bbox_cat = _cat_df[
                                (_cat_df['latitude'] >= _lat_min)
                                & (_cat_df['latitude'] <= _lat_max)
                                & (_cat_df['longitude'] >= _lon_min)
                                & (_cat_df['longitude'] <= _lon_max)
                            ].copy().reset_index(drop=True)
                            if boundary_geom is not None and not _bbox_cat.empty:
                                _inside_cat = [
                                    boundary_geom.covers(_Point(float(lon), float(lat)))
                                    for lat, lon in zip(_bbox_cat['latitude'], _bbox_cat['longitude'])
                                ]
                                _bbox_cat = _bbox_cat.loc[_inside_cat].copy().reset_index(drop=True)
                            print(f"  Aporee recordings within bbox: {len(_bbox_cat)}")
                            if _bbox_cat.empty:
                                output_df = None
                            else:
                                output_df = getSoundAporee(
                                    location=[_center_lon, _center_lat],
                                    loc_id=0,
                                    distance=_circum_radius,
                                    catalog=_bbox_cat,
                                    query=query,
                                    tag=tag,
                                    max_return=100_000,
                                    year=year,
                                    season=season,
                                    time_of_day=time_of_day,
                                    duration=duration,
                                    slice_duration=slice_duration,
                                    slice_max_num=slice_max_num,
                                    probe_durations=probe_durations,
                                    probe_workers=probe_workers,
                                    output_df=True,
                                )
                                output_df = _filter_to_area(output_df)
                    else:
                        # ── single API call + bbox post-filter ───────────────
                        output_df = getSound(
                            [_center_lon, _center_lat],
                            loc_id=0,
                            distance=_circum_radius,
                            source=source,
                            key=key,
                            catalog=catalog,
                            query=query,
                            tag=tag,
                            max_return=100_000,
                            year=year,
                            season=season,
                            time_of_day=time_of_day,
                            duration=duration,
                            exclude_from_location=exclude_from_location,
                            slice_duration=slice_duration,
                            slice_max_num=slice_max_num,
                            probe_durations=probe_durations,
                            output_df=True,
                        )
                        # Keep only sounds whose coordinates fall inside the
                        # requested bbox or exact boundary polygon.
                        output_df = _filter_to_area(output_df)

                    _absorb(output_df, loc_id=0)

                except Exception as e:
                    # Always log bbox errors — silent only suppresses per-item
                    # errors in PATH B, not the whole-bbox fetch.
                    logger.warning("bbox sound fetch error: %s", e)

        # ═══════════════════════════════════════════════════════════════════════
        # PATH B — per-building (or per-POI) loop.
        # ═══════════════════════════════════════════════════════════════════════
        else:
            _loop_units = self.units
            if id_column is None:
                id_column = 'loc_id'
                if id_column not in _loop_units.columns:
                    _loop_units = _loop_units.copy()
                    _loop_units[id_column] = list(range(len(_loop_units)))

            skip_count = 0
            for _index, row in tqdm(_loop_units.iterrows(), total=len(_loop_units)):
                loc_id = row[id_column]
                if loc_id in done_ids_au:
                    continue
                try:
                    output_df = getSound(
                        [row.geometry.centroid.x, row.geometry.centroid.y],
                        loc_id=loc_id,
                        distance=_distance,
                        source=source,
                        key=key,
                        catalog=catalog,
                        query=query,
                        tag=tag,
                        max_return=max_return,
                        year=year,
                        season=season,
                        time_of_day=time_of_day,
                        duration=duration,
                        exclude_from_location=exclude_from_location,
                        slice_duration=slice_duration,
                        slice_max_num=slice_max_num,
                        probe_durations=probe_durations,
                        output_df=True,
                    )
                    _absorb(output_df, loc_id=loc_id)
                except Exception as e:
                    if not silent:
                        logger.warning("sound fetch error: %s", e)
                    skip_count += 1
                    continue

            if skip_count > 0:
                logger.info(
                    'Collected data for %d locations; skipped %d (no data found).',
                    len(_loop_units) - skip_count, skip_count,
                )

        self.audio_metadata = pd.concat(frames, ignore_index=True) if frames else None

        # ── bbox post-process: assign per-recording loc_ids and rebuild units ─
        # The collection phase uses loc_id=0 as a placeholder.  Here we remap
        # each recording to its own sequential loc_id derived from its actual
        # lat/lon, so get_svi_from_locations() and build_pairs() land at the
        # correct coordinates.
        if _from_bbox and self.audio_metadata is not None and not self.audio_metadata.empty:
            _aud = self.audio_metadata.copy()
            if 'latitude' in _aud.columns and 'longitude' in _aud.columns:
                # Use 'id' for deduplication when it has at least some non-NaN
                # values; otherwise fall back to a coordinate-derived key so
                # that NaN id columns (e.g. from a catalog CSV round-trip) do
                # not collapse everything into a single phantom point.
                if 'id' in _aud.columns and not _aud['id'].isna().all():
                    _dedup_col = 'id'
                else:
                    _aud['_coord_key'] = _aud.apply(
                        lambda r: f"{float(r['latitude']):.6f}_{float(r['longitude']):.6f}",
                        axis=1,
                    )
                    _dedup_col = '_coord_key'

                _uniq = _aud.drop_duplicates(subset=[_dedup_col]).reset_index(drop=True)
                _uniq['_new_loc_id'] = _uniq.index
                # Build a lookup that maps the dedup key → new loc_id.
                # Using a list comprehension avoids dict-get(NaN) pitfalls.
                _key_series = _uniq[_dedup_col].tolist()
                _id_to_loc: dict = {k: v for k, v in zip(_key_series, _uniq['_new_loc_id'].tolist())}

                # Remap self.audios['loc_id'] via the same dedup key.
                if _dedup_col == 'id':
                    self.audios['loc_id'] = [
                        _id_to_loc.get(sid, lid)
                        for sid, lid in zip(self.audios['id'], self.audios['loc_id'])
                    ]
                    self.audio_metadata['loc_id'] = self.audio_metadata['id'].map(_id_to_loc)
                else:
                    _am_coord_keys = self.audio_metadata.apply(
                        lambda r: f"{float(r['latitude']):.6f}_{float(r['longitude']):.6f}",
                        axis=1,
                    )
                    # self.audios['id'] is NaN too, so match via position
                    _am_audio_coord_keys = _aud.apply(
                        lambda r: f"{float(r['latitude']):.6f}_{float(r['longitude']):.6f}",
                        axis=1,
                    ).tolist()
                    self.audios['loc_id'] = [
                        _id_to_loc.get(ck, lid)
                        for ck, lid in zip(_am_audio_coord_keys, self.audios['loc_id'])
                    ]
                    self.audio_metadata['loc_id'] = _am_coord_keys.map(_id_to_loc)

                from shapely.geometry import Point as _SPoint
                import geopandas as _gpd2
                self.units = _gpd2.GeoDataFrame(
                    {'loc_id': _uniq['_new_loc_id'].tolist()},
                    geometry=[
                        _SPoint(float(r.longitude), float(r.latitude))
                        for r in _uniq.itertuples()
                    ],
                    crs='EPSG:4326',
                )
                logger.info(
                    'bbox collection: %d unique recordings → %d units created.',
                    len(_uniq), len(self.units),
                )

        _dedupe_audio_state()
        return None

    def filter_outdoor(
        self,
        data: str = "audio",
        building_source: str = "loaded",
        buffer_m: float = -2.0,
        bbox: tuple | None = None,
        overture_release: str = "latest",
        overture_local: str | None = None,
        silent: bool = True,
    ) -> dict:
        """Remove audio recordings and/or Flickr photos whose coordinates fall
        inside a building footprint — i.e., likely indoor recordings/shots.

        Street-view images (Mapillary, Google) are captured from the road and
        are always outdoor; they are not affected by this method.

        Args:
            data (str): Modality to filter.  One of ``"audio"`` (default),
                ``"photo"``, or ``"both"``.
            building_source (str): Source of building footprint polygons for
                the point-in-polygon test.

                * ``"loaded"`` *(default)* — use the polygons already in
                  ``self.units`` (populated by :meth:`getBuildings`).
                  Raises ``RuntimeError`` if ``self.units`` has no polygon
                  geometries.
                * ``"overture"`` *(recommended, especially for large areas)* — stream
                  building footprints from Overture Maps S3 parquet files
                  via DuckDB.  No API key required.  DuckDB pushes the
                  bounding-box predicate into the Parquet row-group
                  statistics so only the relevant slices are downloaded —
                  orders of magnitude faster than tile-based fetchers and much
                  more reliable than large OpenStreetMap/Overpass queries.
                  DuckDB is part of openSIP's core install.
                  Pass ``overture_local`` to use a pre-downloaded parquet
                  file for repeated runs on the same area.
                * ``"microsoft"`` — fetch from Microsoft Global ML Building
                  Footprints (Azure blob tiles, no API key, no rate limits).
                * ``"globfp3d"`` — fetch from GloBFP3D.
                * ``"gba"``      — fetch from the Google Building Accuracy
                  dataset.
                * ``"osm"``      — fetch from OpenStreetMap via the Overpass
                  API. Intended only for small areas; large-scale queries
                  commonly time out.

            buffer_m (float): Inward buffer applied to building polygons
                before the point-in-polygon test, in metres.  A negative
                value shrinks each polygon so that only coordinates clearly
                inside a footprint are treated as indoor — useful when GPS
                places a recording just inside a building edge.
                Default ``-2.0``.  For ``building_source="overture"`` the
                buffer is applied as an approximate degree value
                (1° ≈ 111 320 m).
            bbox (tuple | None): ``(min_lon, min_lat, max_lon, max_lat)``
                used when fetching buildings for non-``"loaded"`` sources.
                If ``None``, computed automatically from the audio / photo
                metadata coordinates.
            overture_release (str): Overture Maps release tag, e.g.
                ``"2026-03-18.0"``.  Default ``"latest"`` — resolves the
                current release via the Overture STAC catalogue.  Ignored
                unless ``building_source="overture"``.
            overture_local (str | None): Path to a local Overture buildings
                parquet file.

                * **File exists** — DuckDB reads it directly, skipping S3
                  entirely.  Fast for repeated runs on the same area.
                * **File does not exist** — the bbox slice is downloaded
                  from S3 and saved to this path with ZSTD compression,
                  then used immediately.  Subsequent calls with the same
                  path reuse the cached file.
                * ``None`` *(default)* — stream from S3 on every call
                  without saving.

                Ignored unless ``building_source="overture"``.
            silent (bool): Suppress per-item warnings.  Default ``True``.

        Returns:
            dict: ``{"audio_dropped": int, "photo_dropped": int}`` — counts
            of recordings / photos removed.

        Raises:
            RuntimeError: If ``building_source="loaded"`` but ``self.units``
                is empty or contains no polygon geometries.
            ValueError: If ``building_source`` is not one of the supported
                values.
        """
        result = {"audio_dropped": 0, "photo_dropped": 0}

        _VALID_SOURCES = ("loaded", "overture", "microsoft", "globfp3d", "gba", "osm")
        if building_source not in _VALID_SOURCES:
            raise ValueError(
                f"filter_outdoor: unknown building_source {building_source!r}. "
                f"Choose one of {_VALID_SOURCES}."
            )
        if data not in ("audio", "photo", "both"):
            raise ValueError(
                "filter_outdoor: data must be 'audio', 'photo', or 'both'."
            )

        # ── Overture Maps / DuckDB path ───────────────────────────────────
        def _overture_indoor_ids(pts_df: pd.DataFrame) -> set:
            """Stream Overture Maps buildings from S3 (or a local parquet)
            via DuckDB and return the set of recording/photo IDs whose
            coordinates fall inside a building polygon.

            Strategy mirrors the standalone drop_indoor_points.py script:
            DuckDB pushes the bbox predicate into the Parquet row-group
            statistics so it only downloads the relevant slices — no tiling
            loop, no per-tile S3 connections.
            """
            try:
                import duckdb
            except ImportError as exc:
                raise ImportError(
                    "building_source='overture' requires duckdb. "
                    "Install or repair the core package with: pip install openSIP"
                ) from exc

            _S3_PREFIX = "s3://overturemaps-us-west-2/release"
            _THEME     = "theme=buildings/type=building"

            con = duckdb.connect()
            for _ext in ("spatial", "httpfs"):
                con.execute(f"INSTALL {_ext}; LOAD {_ext};")
            con.execute("""
                CREATE SECRET IF NOT EXISTS overture_anon (
                    TYPE s3,
                    REGION 'us-west-2',
                    URL_STYLE 'vhost'
                );
            """)

            # resolve release
            _release = overture_release
            if _release == "latest":
                _row = con.execute(
                    "SELECT latest FROM "
                    "'https://stac.overturemaps.org/catalog.json'"
                ).fetchone()
                if _row:
                    _release = _row[0]
                else:
                    raise RuntimeError(
                        "filter_outdoor: could not resolve latest Overture "
                        "release from STAC."
                    )

            valid = pts_df.dropna(subset=["latitude", "longitude"])
            if valid.empty:
                con.close()
                return set()

            con.register("_pts_raw", valid[["id", "latitude", "longitude"]])

            # bbox for S3 predicate pushdown
            pad = 0.001
            lon_min = float(valid["longitude"].min()) - pad
            lat_min = float(valid["latitude"].min()) - pad
            lon_max = float(valid["longitude"].max()) + pad
            lat_max = float(valid["latitude"].max()) + pad

            # buildings source: local parquet or S3 streaming
            # If overture_local is given but the file does not yet exist,
            # download the bbox slice from S3 and save it there first so
            # subsequent calls can skip the S3 round-trip entirely.
            _local = overture_local
            if _local and not Path(_local).exists():
                logger.info(
                    "filter_outdoor: %r not found — downloading Overture "
                    "buildings for bbox (%.4f, %.4f, %.4f, %.4f) to %r …",
                    _local, lon_min, lat_min, lon_max, lat_max, _local,
                )
                _s3_glob = f"{_S3_PREFIX}/{_release}/{_THEME}/*"
                try:
                    Path(_local).parent.mkdir(parents=True, exist_ok=True)
                    con.execute(f"""
                        COPY (
                            SELECT ST_AsWKB(geometry) AS geometry
                            FROM read_parquet('{_s3_glob}',
                                             hive_partitioning=true)
                            WHERE bbox.xmin <= {lon_max}
                              AND bbox.xmax >= {lon_min}
                              AND bbox.ymin <= {lat_max}
                              AND bbox.ymax >= {lat_min}
                        ) TO '{_local}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)
                    """)
                    logger.info(
                        "filter_outdoor: Overture buildings saved to %r — "
                        "pass overture_local=%r to reuse on future calls.",
                        _local, _local,
                    )
                except Exception as exc:
                    logger.warning(
                        "filter_outdoor: could not save Overture buildings "
                        "to %r (%s). Falling back to live S3 streaming.",
                        _local, exc,
                    )
                    _local = None   # fall through to S3 streaming below

            if _local and Path(_local).exists():
                bldg_sql = f"""
                    SELECT ST_GeomFromWKB(geometry) AS geom
                    FROM read_parquet('{_local}')
                """
            else:
                _s3_glob = f"{_S3_PREFIX}/{_release}/{_THEME}/*"
                bldg_sql = f"""
                    SELECT geometry AS geom
                    FROM read_parquet('{_s3_glob}', hive_partitioning=true)
                    WHERE bbox.xmin <= {lon_max}
                      AND bbox.xmax >= {lon_min}
                      AND bbox.ymin <= {lat_max}
                      AND bbox.ymax >= {lat_min}
                """

            # approximate degree-based buffer (negative = inward shrink)
            # 1° ≈ 111 320 m; adequate precision for this filter
            _buf_deg = buffer_m / 111_320.0

            try:
                _result = con.execute(f"""
                    SELECT DISTINCT p.id
                    FROM (
                        SELECT id,
                               ST_Point(longitude, latitude) AS geom
                        FROM _pts_raw
                    ) p
                    JOIN (
                        SELECT ST_Buffer(geom, {_buf_deg}) AS geom
                        FROM ({bldg_sql})
                        WHERE geom IS NOT NULL
                    ) b ON ST_Within(p.geom, b.geom)
                """).fetchdf()
            except Exception as exc:
                logger.warning(
                    "filter_outdoor (overture): spatial join failed (%s). "
                    "No points will be filtered.", exc
                )
                con.close()
                return set()

            con.close()
            return set(_result["id"].tolist()) if not _result.empty else set()

        # ── geopandas path (loaded / microsoft / globfp3d / gba / osm) ───
        def _resolve_buildings_gdf(pts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            if building_source == "loaded":
                if self.units is None or len(self.units) == 0:
                    raise RuntimeError(
                        "filter_outdoor: self.units is empty. "
                        "Call getBuildings() first, or set building_source to "
                        "'overture', 'microsoft', 'globfp3d', 'gba', or 'osm'."
                    )
                poly_mask = self.units.geometry.geom_type.isin(
                    ["Polygon", "MultiPolygon"]
                )
                if not poly_mask.any():
                    raise RuntimeError(
                        "filter_outdoor: self.units contains no polygon "
                        "geometries (only points). Call getBuildings() or "
                        "use a different building_source."
                    )
                return self.units[poly_mask].copy()

            _bbox = bbox
            if _bbox is None:
                if len(pts_gdf) == 0:
                    return gpd.GeoDataFrame(
                        geometry=gpd.GeoSeries([], crs="EPSG:4326")
                    )
                pad = 0.001
                _bbox = (
                    float(pts_gdf.geometry.x.min()) - pad,
                    float(pts_gdf.geometry.y.min()) - pad,
                    float(pts_gdf.geometry.x.max()) + pad,
                    float(pts_gdf.geometry.y.max()) + pad,
                )

            from .utils.building import (
                getGBABuildings,
                getGloBFP3DBuildings,
                getGlobalMLBuilding,
                getOSMbuildings,
            )
            _fetcher = {
                "microsoft": lambda b: getGlobalMLBuilding(b),
                "globfp3d":  lambda b: getGloBFP3DBuildings(b),
                "gba":       lambda b: getGBABuildings(b),
                "osm":       lambda b: getOSMbuildings(b),
            }[building_source]
            try:
                gdf = _fetcher(_bbox)
            except Exception as exc:
                logger.warning(
                    "filter_outdoor: failed to fetch buildings from %r (%s). "
                    "No points will be filtered.", building_source, exc
                )
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs="EPSG:4326")
                )
            if gdf is None or len(gdf) == 0:
                logger.warning(
                    "filter_outdoor: no buildings returned from %r. "
                    "No points will be filtered.", building_source
                )
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs="EPSG:4326")
                )
            return gdf

        def _gpd_indoor_ids(pts_df: pd.DataFrame) -> set:
            valid = pts_df.dropna(subset=["latitude", "longitude"])
            if valid.empty:
                return set()
            pts_gdf = gpd.GeoDataFrame(
                valid[["id", "latitude", "longitude"]],
                geometry=gpd.points_from_xy(
                    valid["longitude"], valid["latitude"]
                ),
                crs="EPSG:4326",
            )
            buildings = _resolve_buildings_gdf(pts_gdf)
            if len(buildings) == 0:
                return set()

            utm = pts_gdf.estimate_utm_crs()
            bldg_proj = buildings[["geometry"]].to_crs(utm).copy()
            if buffer_m != 0.0:
                bldg_proj["geometry"] = bldg_proj.geometry.buffer(buffer_m)
            bldg_proj = bldg_proj[~bldg_proj.geometry.is_empty]

            joined = gpd.sjoin(
                pts_gdf.to_crs(utm)[["id", "geometry"]],
                bldg_proj,
                how="left",
                predicate="within",
            )
            return set(
                joined.loc[joined["index_right"].notna(), "id"].unique()
            )

        # ── shared: apply indoor-id set to payload + metadata ─────────────
        def _apply_filter(
            payload: dict,
            metadata,
            label: str,
        ) -> tuple[dict, object, int]:
            if metadata is None or len(metadata) == 0:
                if not silent:
                    logger.warning(
                        "filter_outdoor: no %s metadata available; "
                        "run get_%s_from_location() first.", label, label
                    )
                return payload, metadata, 0

            meta = metadata.copy()
            if "longitude" not in meta.columns or "latitude" not in meta.columns:
                if not silent:
                    logger.warning(
                        "filter_outdoor: %s metadata has no lat/lon columns; "
                        "skipping filter.", label
                    )
                return payload, metadata, 0

            pts_df = meta[["id", "latitude", "longitude"]].copy()

            # choose the spatial backend
            if building_source == "overture":
                indoor_ids = _overture_indoor_ids(pts_df)
            else:
                indoor_ids = _gpd_indoor_ids(pts_df)

            if not indoor_ids:
                return payload, metadata, 0

            keep_meta = ~meta["id"].isin(indoor_ids)
            new_meta  = meta[keep_meta].reset_index(drop=True)
            n_dropped = int((~keep_meta).sum())

            keep_payload = [pid not in indoor_ids for pid in payload["id"]]
            new_payload  = {
                k: [v for v, keep in zip(vs, keep_payload) if keep]
                for k, vs in payload.items()
            }
            return new_payload, new_meta, n_dropped

        # ── dispatch ──────────────────────────────────────────────────────
        if data in ("audio", "both"):
            self.audios, self.audio_metadata, n = _apply_filter(
                self.audios, self.audio_metadata, "audio"
            )
            result["audio_dropped"] = n
            if n:
                logger.info(
                    "filter_outdoor: removed %d indoor audio recording(s).", n
                )

        if data in ("photo", "both"):
            self.photos, self.photo_metadata, n = _apply_filter(
                self.photos, self.photo_metadata, "photo"
            )
            result["photo_dropped"] = n
            if n:
                logger.info(
                    "filter_outdoor: removed %d indoor photo(s).", n
                )

        return result

    def download_to_dir(self, data:str = None, to_dir:str = None, prefix: str = None)-> None:
        '''
            download_to_dir

            Download retrieved data (fetched by get_svi_from_locations,
            get_photo_from_location, or get_sound_from_location) to a local
            directory and populate the corresponding ``path`` list on the
            dataset object.

            This method is **resume-safe by default**: if a file already
            exists at its target path it is never re-downloaded.  You can
            safely re-run this call after a crash and it will only fetch
            the files that are still missing, then rebuild the complete
            path list from what is on disk.

            Args:
                data (str): Type of data to download: ['svi', 'audio', 'photo'].
                to_dir (str): the directory to save the downloaded data.
                prefix (str, optional):  The prefix to add to the output filename.
        '''
        if data not in ['svi', 'audio', 'photo']:
            raise ValueError('Invalid data type provided. It has to be one of ["svi", "audio", "photo"].')
        if to_dir is None:
            raise ValueError("to_dir must be provided.")
        if not os.path.exists(to_dir):
            logger.info("Directory %s does not exist; creating.", to_dir)
            Path(to_dir).mkdir(parents=True, exist_ok=True)
        if data == 'svi':
            if len(self.svis['id']) == 0:
                return None
            self.svis['path'] = []
            for i in tqdm(range(len(self.svis['data'])), total=len(self.svis['data'])):
                loc_id = self.svis['loc_id'][i]
                img_id = self.svis['id'][i]
                path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                p = path + f'_{img_id}.png'
                if not os.path.exists(p):
                    try:
                        if is_base64(self.svis['data'][i]):
                            save_base64(self.svis['data'][i], p)
                        else:
                            download_image_requests(self.svis['data'][i], p)
                    except Exception:
                        self.svis['path'] += [" "]
                        continue
                self.svis['path'] += [p]
        elif data == 'audio':
            if len(self.audios['id']) == 0:
                return None
            self.audios['path'] = []
            if 'slice' in self.audios:
                for i in tqdm(range(len(self.audios['data'])), total=len(self.audios['data'])):
                    loc_id = self.audios['loc_id'][i]
                    audio_id = self.audios['id'][i]
                    slices = self.audios['slice'][i]
                    path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                    start = slices[0]
                    end = slices[1]
                    p = path + f'_{audio_id}_clip_{start}_{end}.mp3'
                    if not os.path.exists(p):
                        try:
                            clip(self.audios['data'][i], start, end, p)
                        except Exception:
                            self.audios['path'] += [" "]
                            continue
                    self.audios['path'] += [p]
            else:
                for i in tqdm(range(len(self.audios['data'])), total=len(self.audios['data'])):
                    loc_id = self.audios['loc_id'][i]
                    audio_id = self.audios['id'][i]
                    path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                    p = path + f'_{audio_id}.mp3'
                    if not os.path.exists(p):
                        try:
                            download_freesound_preview(self.audios['data'][i], p)
                        except Exception:
                            self.audios['path'] += [" "]
                            continue
                    self.audios['path'] += [p]
        elif data == 'photo':
            if len(self.photos['id']) == 0:
                return None
            self.photos['path'] = []
            for i in tqdm(range(len(self.photos['data'])), total=len(self.photos['data'])):
                loc_id = self.photos['loc_id'][i]
                photo_id = self.photos['id'][i]
                path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                p = path + f'_{photo_id}.png'
                if not os.path.exists(p):
                    try:
                        download_image_requests(self.photos['data'][i], p)
                    except Exception:
                        # download failed: align list lengths with sentinel
                        self.photos['path'] += [" "]
                        continue
                self.photos['path'] += [p]
        return None

    def export(
        self,
        output_dir: str,
        data: str = 'svi',
        labels: pd.DataFrame = None,
    ) -> str:
        """Export collected data as an organized dataset folder.

        Creates::

            output_dir/
                metadata.csv          # loc_id, file_id, file_type, file_path
                                      # + optional label columns from `labels`
                images/               # when data in {'svi', 'photo'}
                    {loc_id}_{file_id}.png
                audio/                # when data == 'audio'
                    {loc_id}_{file_id}.mp3

        If a file already exists on disk at the target path it is not
        downloaded again, so the method is safe to call repeatedly.

        Args:
            output_dir: Root directory for the exported dataset.
            data: Which modality to export. One of ``'svi'``, ``'photo'``,
                or ``'audio'``.
            labels: Optional DataFrame produced by ``batch_inference()``.
                Must contain a ``loc_id`` column; it is left-joined onto the
                metadata table so each file row gets the matching label
                columns.

        Returns:
            Absolute path to the created ``metadata.csv``.
        """
        if data not in ('svi', 'photo', 'audio'):
            raise ValueError(
                "data must be one of 'svi', 'photo', 'audio'; "
                f"got {data!r}"
            )

        out_root = Path(output_dir)

        if data in ('svi', 'photo'):
            files_dir = out_root / 'images'
        else:
            files_dir = out_root / 'audio'
        files_dir.mkdir(parents=True, exist_ok=True)

        payload = (
            self.svis if data == 'svi'
            else self.photos if data == 'photo'
            else self.audios
        )

        if not payload['id']:
            logger.warning("export: no %s data to export.", data)
            return str(out_root / 'metadata.csv')

        ext = '.png' if data != 'audio' else '.mp3'
        rows: list[dict] = []

        for i in range(len(payload['id'])):
            loc_id = payload['loc_id'][i]
            file_id = payload['id'][i]
            source = payload['data'][i]
            existing_path = payload['path'][i] if i < len(payload['path']) else ''

            fname = f"{loc_id}_{file_id}{ext}"
            local_path = str(files_dir / fname)

            # Download / copy only if the file is not already in place
            if not Path(local_path).exists():
                try:
                    if existing_path and Path(existing_path).exists():
                        import shutil as _shutil
                        _shutil.copy2(existing_path, local_path)
                    elif data != 'audio':
                        if is_url(source):
                            download_image_requests(source, local_path)
                        elif is_image_path(source):
                            import shutil as _shutil
                            _shutil.copy2(source, local_path)
                        else:
                            # assume base64
                            save_base64(source, local_path)
                    else:
                        # audio
                        slices = (
                            payload['slice'][i]
                            if 'slice' in payload and i < len(payload['slice'])
                            else None
                        )
                        if slices is not None:
                            clip(source, slices[0], slices[1], local_path)
                        else:
                            download_freesound_preview(source, local_path)
                except Exception as _dl_err:
                    logger.warning(
                        "export: could not save %s (loc_id=%s, file_id=%s): %s",
                        local_path, loc_id, file_id, _dl_err,
                    )

            rows.append({
                'loc_id': loc_id,
                'file_id': file_id,
                'file_type': data,
                'file_path': local_path,
                'source_data': source if is_url(source) else '<local>',
            })

        meta_df = pd.DataFrame(rows)

        if labels is not None:
            if 'loc_id' in labels.columns:
                meta_df = meta_df.merge(labels, on='loc_id', how='left')
            else:
                logger.warning(
                    "export: labels DataFrame has no 'loc_id' column; skipping merge."
                )

        out_csv = out_root / 'metadata.csv'
        meta_df.to_csv(out_csv, index=False)
        logger.info("export: wrote %d rows to %s", len(meta_df), out_csv)
        return str(out_csv)

    # ------------------------------------------------------------------
    # Persistence — save / load
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Snapshot the current GeoTaggedData state to *path*.

        Creates a directory containing:

        * ``manifest.json`` — version and UTC timestamp
        * ``units.geojson`` — recording/building GeoDataFrame (if set)
        * ``audio_metadata.csv``, ``svi_metadata.csv``, ``photo_metadata.csv``
          (whichever are not None)
        * ``audios.json``, ``svis.json``, ``photos.json`` — payload dicts
          (ids, URLs, slice boundaries)

        Use :meth:`load` to restore::

            gtd.save("run/state")
            gtd2 = GeoTaggedData.load("run/state")

        :class:`~opensip.pairing.AudioImagePairDataset` calls this
        automatically from its own :meth:`~opensip.pairing.AudioImagePairDataset.save`.

        Parameters
        ----------
        path:
            Directory to write into (created if it does not exist).
        """
        import json as _json
        import datetime as _dt

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # manifest
        (save_dir / "manifest.json").write_text(
            _json.dumps(
                {"version": "1", "saved_at": _dt.datetime.utcnow().isoformat()},
                indent=2,
            )
        )

        # units (GeoDataFrame → GeoJSON)
        if self.units is not None and len(self.units):
            self.units.to_file(save_dir / "units.geojson", driver="GeoJSON")

        # metadata DataFrames → CSV
        for _attr in ("audio_metadata", "svi_metadata", "photo_metadata"):
            _df = getattr(self, _attr)
            if _df is not None and not _df.empty:
                _df.to_csv(save_dir / f"{_attr}.csv", index=False)

        # payload dicts → JSON
        for _attr in ("audios", "svis", "photos"):
            (save_dir / f"{_attr}.json").write_text(
                _json.dumps(getattr(self, _attr))
            )

        logger.info("GeoTaggedData saved to %s", save_dir)

    @classmethod
    def load(cls, path) -> "GeoTaggedData":
        """Restore a GeoTaggedData object from a directory created by :meth:`save`.

            gtd = GeoTaggedData.load("run/state")
            # gtd.units, gtd.audio_metadata, gtd.audios, … are all restored

        Parameters
        ----------
        path:
            Directory previously written by :meth:`save`.
        """
        import json as _json

        load_dir = Path(path)
        if not load_dir.is_dir():
            raise FileNotFoundError(
                f"GeoTaggedData state directory not found: {load_dir}"
            )

        gtd = cls()  # empty init — locations=None, units=None

        # units
        _units_path = load_dir / "units.geojson"
        if _units_path.exists():
            gtd.units = gpd.read_file(str(_units_path))

        # metadata DataFrames
        for _attr in ("audio_metadata", "svi_metadata", "photo_metadata"):
            _csv = load_dir / f"{_attr}.csv"
            if _csv.exists():
                setattr(gtd, _attr, pd.read_csv(str(_csv)))

        # payload dicts
        for _attr in ("audios", "svis", "photos"):
            _jp = load_dir / f"{_attr}.json"
            if _jp.exists():
                setattr(gtd, _attr, _json.loads(_jp.read_text()))

        logger.info("GeoTaggedData loaded from %s", load_dir)
        return gtd

    def set_images(self, img_type: str):
        '''
            set_images

            Set retrieved street view images or Flickr photos as images dataset

            Args:
                img_type (str): 'photo' or 'svi'
        '''
        if img_type == 'svi':
            self.images = self.svis
        elif img_type == 'photo':
            self.images = self.photos
        return None

    def plot_data(self, data:str = None, export_gdf: bool = False) -> None:
        '''

        Args:
            data (str): Type of data to download: ['svi', 'audio', 'photo'].
            export_gdf (bool): Export gpd.GeoDataFrame.
        '''
        if data is None:
            return None

        if data == 'svi':
            temp = self.svi_metadata
            geometry = gpd.points_from_xy(temp['image_lon'], temp['image_lat'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">View image details</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(temp, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "captured_at", "detail"]
        elif data == 'photo':
            temp = self.photo_metadata
            geometry = gpd.points_from_xy(temp['longitude'], temp['latitude'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">View photo details</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(temp, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "datetaken", "detail"]
        elif data == 'audio':
            temp = self.audio_metadata
            geometry = gpd.points_from_xy(temp['longitude'], temp['latitude'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">Listen to the sound</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(self.audio_metadata, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "created_dt", "detail"]
        else:
            raise ValueError('Invalid data type provided. It has to be one of ["svi", "audio", "photo"].')

        self.plot = gdf.explore(
            popup=popup,
            color="red",
            marker_kwds=dict(radius=5, fill=True),
            tiles="CartoDB positron",
            name="map",
        )
        return gdf if export_gdf else self.plot


# Get street view images from Mapillary
def _getSV_mapillary(location, loc_id, distance, key,
                     pano, reoriented, multi_num, interval,
                     fov, heading, pitch, height, width,
                     year, season, time_of_day,
                     target_polygon, fov_margin, fov_min, fov_max,
                     building_height, output_df, silent):
    """Retrieve street view image(s) from Mapillary. Called by :func:`getSV`."""
    # Resolve auto-fov mode upfront so per-image computation can branch.
    auto_fov = isinstance(fov, str) and fov.strip().lower() == "auto"
    if auto_fov and not reoriented:
        raise ValueError("fov='auto' requires reoriented=True (we need a directed view).")

    api_key = key or os.getenv("MAPILLARY_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Mapillary access token. Pass key=... or set env var MAPILLARY_API_KEY."
        )
    bbox = projection(location, r=distance)
    url = f"https://graph.mapillary.com/images?access_token={api_key}&fields=id,computed_compass_angle,thumb_original_url,captured_at,computed_geometry,sequence&bbox={bbox}"
    # 2048 -> original to get higher resolution
    if pano:
        url += "&is_pano=true"
    if not pano and reoriented:
        reoriented = False

    svis = []
    svi_df = {
        "id": [],
        "sequence": [],
        "captured_at": [],
        "compass_angle": [],
        "image_lon": [],
        "image_lat": [],
        'url': [],
        'loc_id': []
    }
    if loc_id is None:
        del svi_df['loc_id']

    try:
        response = retry_request(url)
        if response is None:
            if not silent:
                logger.warning('skip location %s: no data found', location)
            if output_df:
                return None, None
            return None
        response = response.json()
        # find the closest image
        response = closest(location, response, multi_num, interval, year, season, time_of_day, api_key)
        if response is None:
            if not silent:
                logger.warning('skip location %s: no data found', location)
            if output_df:
                return None, None
            return None

        for _index, row in response.iterrows():
            # Extract Image ID, Compass Angle, image url, and coordinates
            img_heading = float(row['computed_compass_angle'])
            img_url = row['thumb_original_url']

            if 'computed_geometry.coordinates' in row.index:
                image_lon, image_lat = row['computed_geometry.coordinates']
            elif 'coordinates' in row.index:
                image_lon, image_lat = row['coordinates']
            else:
                coor_columns = [col for col in row.index if 'coordinates' in col]
                image_lon, image_lat = row[coor_columns[0]]

            if heading is None:
                # calculate bearing to the house
                bearing_to_house = calculate_bearing(image_lat, image_lon, location[1], location[0])
                relative_heading = (bearing_to_house - img_heading) % 360
            else:
                relative_heading = heading

            # Resolve effective FOV per image — supports the literal 'auto'.
            if auto_fov:
                from .utils.utils import (
                    auto_fov_from_distance,
                    auto_fov_from_polygon,
                    haversine_m,
                )
                aspect = float(width) / float(height)
                if target_polygon is not None:
                    abs_bearing = (img_heading + relative_heading) % 360
                    effective_fov = auto_fov_from_polygon(
                        camera_lon=image_lon, camera_lat=image_lat,
                        polygon=target_polygon,
                        view_bearing_deg=abs_bearing,
                        margin=fov_margin, min_fov=fov_min, max_fov=fov_max,
                        building_height_m=building_height,
                        aspect_ratio=aspect,
                    )
                else:
                    dist_m = haversine_m(image_lat, image_lon, location[1], location[0])
                    effective_fov = auto_fov_from_distance(
                        distance_m=dist_m,
                        margin=fov_margin, min_fov=fov_min, max_fov=fov_max,
                        building_height_m=building_height,
                        aspect_ratio=aspect,
                    )
            else:
                effective_fov = fov

            # reframe image
            if reoriented:
                svi = Equirectangular(img_url=img_url)
                sv = svi.GetPerspective(effective_fov, relative_heading, pitch, height, width, 128)
                svis.append(sv)
            else:
                svis.append(img_url)

            if output_df:
                svi_df['id'].append(row['id'])
                svi_df['sequence'].append(row['sequence'])
                svi_df['captured_at'].append(f'{row["year"]}-{row["month"]}-{row["day"]}-{row["hour"]}')
                svi_df['image_lon'].append(image_lon)
                svi_df['image_lat'].append(image_lat)
                svi_df['compass_angle'].append(img_heading)
                svi_df['url'].append(img_url)
                if 'loc_id' in svi_df:
                    svi_df['loc_id'].append(loc_id)
        if output_df:
            return svis, pd.DataFrame(svi_df)
        else:
            return svis
    except Exception as e:
        if not silent:
            logger.warning('skip location %s: %s', location, e)
        if output_df:
            return None, None
        return None


def _getSV_google(location, loc_id, distance, key,
                  fov, heading, pitch, height, width,
                  output_df, silent):
    """Retrieve a street view image from Google Static Street View. Called by :func:`getSV`.

    Notes:
        - ``multi_num``, ``pano``, ``reoriented``, ``year``, ``season``, and
          ``time_of_day`` are not supported and are handled by the dispatcher.
        - The image is always returned as a base64-encoded JPEG string to match
          the Mapillary output format.
        - ``captured_at`` uses the format ``"YYYY-MM-1-1"`` (day and hour are
          nominal placeholders) because the Google Street View Metadata API only
          exposes year and month.
    """
    import base64

    api_key = key or os.getenv("GOOGLE_STREETVIEW_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Google Street View API key. "
            "Pass key=... or set env var GOOGLE_STREETVIEW_API_KEY."
        )

    svi_df = {
        "id": [],
        "sequence": [],
        "captured_at": [],
        "compass_angle": [],
        "image_lon": [],
        "image_lat": [],
        "url": [],
        "loc_id": [],
    }
    if loc_id is None:
        del svi_df["loc_id"]

    try:
        # 1. Metadata API — find the actual panorama position and capture date.
        meta_url = (
            "https://maps.googleapis.com/maps/api/streetview/metadata"
            f"?location={location[1]},{location[0]}"
            f"&radius={distance}"
            f"&key={api_key}"
        )
        meta_resp = retry_request(meta_url)
        if meta_resp is None or meta_resp.json().get("status") != "OK":
            if not silent:
                logger.warning("skip location %s: no Google Street View data found", location)
            if output_df:
                return None, None
            return None

        meta = meta_resp.json()
        pano_lat = meta["location"]["lat"]
        pano_lon = meta["location"]["lng"]
        pano_id  = meta.get("pano_id", "")
        date_str = meta.get("date", "")      # "YYYY-MM" or ""

        # Build a consistent captured_at string: "YYYY-MM-1-1".
        # Day and hour are nominal placeholders (Google only provides year-month).
        if date_str:
            captured_at = f"{date_str}-1-1"
        else:
            captured_at = None

        # 2. Compute absolute heading so the camera faces the target.
        effective_heading = (
            heading if heading is not None
            else calculate_bearing(pano_lat, pano_lon, location[1], location[0])
        )

        # Clamp fov to Google's supported range [10, 120].
        effective_fov = max(10, min(120, float(fov)))

        # 3. Static Street View API — returns a perspective JPEG directly.
        # Build two variants: a live URL for fetching, and a sanitized one (key
        # replaced with the literal "key") safe to store in the output DataFrame.
        _base = "https://maps.googleapis.com/maps/api/streetview"
        _params = (
            f"?size={width}x{height}"
            f"&pano={pano_id}"
            f"&heading={effective_heading}"
            f"&pitch={pitch}"
            f"&fov={effective_fov}"
        )
        img_url        = _base + _params + f"&key={api_key}"
        img_url_stored = _base + _params + "&key=key"
        img_resp = retry_request(img_url)
        if img_resp is None or img_resp.status_code != 200:
            if not silent:
                logger.warning("skip location %s: Google Street View image fetch failed", location)
            if output_df:
                return None, None
            return None

        img_b64 = base64.b64encode(img_resp.content).decode()

        if output_df:
            svi_df["id"].append(pano_id)
            svi_df["sequence"].append(None)
            svi_df["captured_at"].append(captured_at)
            svi_df["compass_angle"].append(effective_heading)
            svi_df["image_lon"].append(pano_lon)
            svi_df["image_lat"].append(pano_lat)
            svi_df["url"].append(img_url_stored)
            if "loc_id" in svi_df:
                svi_df["loc_id"].append(loc_id)
            return [img_b64], pd.DataFrame(svi_df)
        return [img_b64]

    except Exception as e:
        if not silent:
            logger.warning("skip location %s: %s", location, e)
        if output_df:
            return None, None
        return None


def getSV(location: list|tuple,
          loc_id: int | str = None,
          distance:int = 50,
          key: str = None,
          source: str = "mapillary",
          pano: bool = False,
          reoriented: bool = False,
          multi_num: int = 1,
          interval: int = 1,
          fov: int | float | str = 80, heading: int = None, pitch: int = 5,
          height: int = 500, width: int = 700,
          year: list | tuple = None,
          season: str = None,
          time_of_day: str = None,
          target_polygon=None,
          fov_margin: float = 0.10,
          fov_min: float = 30.0,
          fov_max: float = 120.0,
          building_height: float = 9.0,
          output_df: bool = True,
          silent: bool = False) -> pd.DataFrame | list | None:
    """
        getSV

        Retrieve the closest street view image(s) near a coordinate.
        Supports multiple sources; the image is reoriented to face the target
        coordinate when ``reoriented=True`` (Mapillary) or always (Google).

        Args:
            location: coordinates (longitude/x and latitude/y)
            loc_id (int|str, optional): The id of the location.
            distance (int): The max distance in meters between the centroid and the street view.
            key (str): API access token for the chosen source.
                Mapillary — pass token or set env var ``MAPILLARY_API_KEY``.
                Google    — pass token or set env var ``GOOGLE_STREETVIEW_API_KEY``.
            source (str): Street view data source. One of ``"mapillary"`` (default)
                or ``"google"``.
            pano (bool): Whether to search for panoramic images only.
                Mapillary only — ignored for Google. (Default is False)
            reoriented (bool): Whether to reorient and crop street view images to face
                the target. Mapillary only — Google always faces the target.
                (Default is False)
            multi_num (int): The number of multiple SVIs. Mapillary only — Google
                always returns 1. (Default is 1)
            interval (int): The interval in meters between each SVI.
                Mapillary only. (Default is 1)
            fov (int | float | str): Field of view in degrees for the perspective image
                (default 80). Pass ``'auto'`` together with ``reoriented=True`` to
                size the FOV per image so the target building is just framed —
                see ``target_polygon`` / ``fov_margin`` / ``fov_min`` / ``fov_max``.
                When ``target_polygon`` is None, ``'auto'`` falls back to a
                distance-based heuristic (assumes ~15 m wide building).
                Mapillary only — for Google, ``fov`` is passed directly to the API
                and clamped to [10, 120]; ``'auto'`` is not supported.
            heading (int): Camera heading in degrees. If None, computed from the
                bearing to the target location.
            pitch (int): Camera pitch angle. (Default is 5)
            height (int): Height in pixels of the returned image. (Default is 500)
            width (int): Width in pixels of the returned image. (Default is 700)
            year (list[str], optional): Year of data (start year, end year).
                Mapillary only — ignored for Google with a warning.
            season (str, optional): Season of data.
                Mapillary only — ignored for Google with a warning.
            time_of_day (str, optional): Time of data.
                Mapillary only — ignored for Google with a warning.
            target_polygon (shapely.geometry.Polygon, optional): Building footprint
                used by ``fov='auto'`` to compute the angular extent of the target.
                Coordinates are assumed to be ``(lon, lat)`` in WGS84.
                Mapillary only.
            fov_margin (float): Fractional padding added to the auto-computed
                FOV (0.10 = +10%). Default 0.10. Mapillary only.
            fov_min (float): Lower clamp for ``fov='auto'`` (degrees). Default 30°.
                Mapillary only.
            fov_max (float): Upper clamp for ``fov='auto'`` (degrees). Default 120°.
                Mapillary only.
            building_height (float): Assumed building height in meters used by
                ``fov='auto'`` (default 9 m, ~3 stories). Mapillary only.
            output_df (bool, optional): Whether to also return a DataFrame of metadata.
                (Default is True)
            silent (bool, optional): Whether to silence warnings. (Default is False)

        Returns:
            list[str]: A list of images in base64 format.
            DataFrame: A dataframe containing metadata about the street view images.
                ``captured_at`` format is ``"YYYY-M-D-H"`` for Mapillary and
                ``"YYYY-MM-1-1"`` for Google (day and hour are nominal placeholders).
    """
    source = source.lower().strip()

    if source == "google":
        # Warn about params that Google does not support.
        # warnings.warn() deduplicates by call-site, so each message appears
        # only once even when getSV() is called in a loop (e.g. from
        # get_svi_from_locations), unlike logger.warning() which fires every time.
        if multi_num > 1:
            warnings.warn(
                "getSV: multi_num > 1 is not supported for source='google'; using 1.",
                stacklevel=2,
            )
        if any([year, season, time_of_day]):
            warnings.warn(
                "getSV: year/season/time_of_day filtering is not supported for "
                "source='google' (API does not expose historical imagery). "
                "These parameters will be ignored.",
                stacklevel=2,
            )
        if isinstance(fov, str) and fov.strip().lower() == "auto":
            warnings.warn(
                "getSV: fov='auto' is not supported for source='google'. "
                "Falling back to fov=80.",
                stacklevel=2,
            )
            fov = 80
        return _getSV_google(
            location=location, loc_id=loc_id, distance=distance, key=key,
            fov=fov, heading=heading, pitch=pitch, height=height, width=width,
            output_df=output_df, silent=silent,
        )

    if source == "mapillary":
        return _getSV_mapillary(
            location=location, loc_id=loc_id, distance=distance, key=key,
            pano=pano, reoriented=reoriented, multi_num=multi_num, interval=interval,
            fov=fov, heading=heading, pitch=pitch, height=height, width=width,
            year=year, season=season, time_of_day=time_of_day,
            target_polygon=target_polygon, fov_margin=fov_margin,
            fov_min=fov_min, fov_max=fov_max, building_height=building_height,
            output_df=output_df, silent=silent,
        )

    raise ValueError(
        f"getSV: unknown source '{source}'. Choose 'mapillary' or 'google'."
    )


from .utils.utils import season_months, tod_hours, year_range


def getPhoto(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        key: str = None,
        query: str | list[str] = None,
        geo_context: int = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        exclude_from_location:int = None,
        output_df: bool = True
):
    """
        getPhoto

        Fetch public Flickr photos with geotags near a location (or within a Flickr place).

        Args:
            location (list|tuple): (lon, lat) required. Coordinates of location (longitude, latitude) for searching for geotagged photos
            loc_id (int | str): The id of the location.
            distance (int): Search radius in meters (converted to km; Flickr radius max is 32 km).
            key (str): Flickr API key. If None, reads env var FLICKR_API_KEY.
            query (str | list[str]): Query parameters to pass to Flickr API (free text search).
            geo_context (int): Specify whether a geotagged photo was taken indoors or outdoors. 0: Not defined; 1: Indoors; 2: Outdoors. (Default is None)
            tag: Tag string or list of tags (comma-separated). Acts as a "limiting agent" for geo queries.
            max_return: Number of photos to return (after filters).
            year (str | tuple): [Y] or (Y,) or (Y1, Y2) inclusive. Filters by taken date range.
            season (str): One of {"spring","summer","fall","autumn","winter"} (post-filter by taken month).
            time_of_day (str): One of {"morning","afternoon","evening","night"} (post-filter by taken hour).
            exclude_from_location (int, optional): drop retrieved photos within a distance (in meter) from the given location. (Default is None)
            output_df (bool): If True, return a pandas.DataFrame; otherwise return dict (if max_return==1)
                       or list[dict].

        Returns:
            dict | list[dict] | pandas.DataFrame
    """

    import os
    from datetime import datetime, timedelta, timezone

    import requests

    if exclude_from_location is not None:
        drop_area = projection(location, r=distance)

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    api_key = key or os.getenv("FLICKR_API_KEY")
    if not api_key:
        raise ValueError("Missing Flickr API key. Pass key=... or set env var FLICKR_API_KEY.")

    lon, lat = location
    months = season_months(season)
    hours = tod_hours(time_of_day)
    y_range = year_range(year)

    # Radius in km (Flickr max 32km) :contentReference[oaicite:3]{index=3}
    radius_km = max(float(distance) / 1000.0, 0.01)
    radius_km = min(radius_km, 32.0)

    # Geo queries need a "limiting agent"; tags or min/max dates qualify. :contentReference[oaicite:4]{index=4}
    # If user provided none, default to last 365 days so results aren’t silently limited to ~12 hours.
    now_utc = datetime.now(timezone.utc)
    default_min_upload_date = int((now_utc - timedelta(days=365)).timestamp())

    # -------------------------
    # Build Flickr request
    # -------------------------
    endpoint = "https://api.flickr.com/services/rest/"

    extras = ",".join(
        [
            "description",
            "license",
            "date_upload",
            "date_taken",
            "owner_name",
            "geo",
            "tags",
            "views",
            "media",
            "url_sq",
            "url_t",
            "url_s",
            "url_q",
            "url_m",
            "url_n",
            "url_z",
            "url_c",
            "url_l",
            "url_o",
        ]
    )

    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": 1,
        "extras": extras,
        "safe_search": 1, # safe only for un-authed calls
        "media": "photos",
        "has_geo": 1,
        "content_types": 0, # photos
        "sort": "relevance",
        "lat": lat,
        "lon": lon,
        "radius": radius_km,
        "radius_units": "km"
    }

    if query:
        q = query_string(query)
        if q:
            params["text"] = q

    if geo_context:
        params["geo_context"] = geo_context

    # tags
    if tag:
        if isinstance(tag, (list, tuple)):
            tags = ",".join([str(t).strip() for t in tag if str(t).strip()])
            params["tags"] = tags
            params["tag_mode"] = "all"
        else:
            params["tags"] = str(tag).strip()

    # date range (taken) if specified
    if y_range is not None:
        params["min_taken_date"], params["max_taken_date"] = y_range
    else:
        # If no explicit limiting agent, set min_upload_date (acts as limiting agent for geo queries). :contentReference[oaicite:7]{index=7}
        if not tag and season is None and time_of_day is None:
            params["min_upload_date"] = default_min_upload_date

    # -------------------------
    # Fetch + post-filter
    # -------------------------
    session = requests.Session()

    # Geo/bbox queries only return up to 250/page. :contentReference[oaicite:8]{index=8}
    per_page = min(250, max(50, max_return * 20))
    params["per_page"] = per_page

    results = []
    seen = set()

    max_pages = 150
    for page in range(1, max_pages + 1):
        params["page"] = page
        r = session.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("stat") != "ok":
            msg = data.get("message") or data.get("error") or str(data)
            raise RuntimeError(f"Flickr API error: {msg}")

        photos = (data.get("photos") or {}).get("photo") or []
        if not photos:
            break

        for p in photos:
            if exclude_from_location is not None:
                if is_coordinate_in_bbox(p["longitude"], p["latitude"], drop_area):
                    continue
            pid = p.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            taken_dt = parse_taken(p)
            if months and taken_dt and taken_dt.month not in months:
                continue
            if hours and taken_dt and taken_dt.hour not in hours:
                continue

            s_lat = float(p["latitude"]) if "latitude" in p and p["latitude"] not in (None, "") else None
            s_lon = float(p["longitude"]) if "longitude" in p and p["longitude"] not in (None, "") else None

            url = best_url(p)
            out = {
                "loc_id": '',
                "id": pid,
                "title": p.get("title"),
                "owner": p.get("owner"),
                # "ownername": p.get("ownername"),
                "datetaken": p.get("datetaken") or p.get("date_taken"),
                "latitude": s_lat,
                "longitude": s_lon,
                # "accuracy": int(p["accuracy"]) if "accuracy" in p and str(p["accuracy"]).isdigit() else None,
                "distance_m": haversine_m(lat, lon, s_lat, s_lon) if (s_lat is not None and s_lon is not None) else None,
                "tags": p.get("tags"),
                "description": p.get("description"),
                "views": int(p["views"]) if "views" in p and str(p["views"]).isdigit() else None,
                "license": p.get("license"),
                "url": url,
                # "page_url": f"https://www.flickr.com/photos/{p.get('owner')}/{pid}",
            }

            if loc_id is not None:
                out["loc_id"] = loc_id
            else:
                del out["loc_id"]

            results.append(out)

            # if len(results) >= max_return:
            #     break

        if len(results) >= max_return:
            break

    if output_df:
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values(by='distance_m', ascending=True)
        return df.head(max_return)

    if max_return == 1:
        return results[0] if results else None
    return results


def getSound(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        source: str = 'freesound',
        key: str = None,
        catalog: str | pd.DataFrame = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int = 300,
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        probe_durations: bool = True,
        probe_workers: int = 8,
        output_df: bool = True,
) -> pd.DataFrame | dict | list | None:
    """Dispatch to the per-source helpers.

    Args:
        source (str): one of {"freesound", "aporee"}. Default "freesound".
        catalog: required when source="aporee" — see :func:`getSoundAporee`.
        probe_durations: Aporee-only. See :func:`getSoundAporee`.

    All other arguments are forwarded; ``key`` is only used by Freesound,
    ``catalog`` and ``probe_durations`` only by Aporee.
    """
    src = (source or 'freesound').lower()
    if src == 'freesound':
        return _getSoundFreesound(
            location=location, loc_id=loc_id, distance=distance, key=key,
            query=query, tag=tag, max_return=max_return, year=year,
            season=season, time_of_day=time_of_day, duration=duration,
            exclude_from_location=exclude_from_location,
            slice_duration=slice_duration, slice_max_num=slice_max_num,
            output_df=output_df,
        )
    elif src == 'aporee':
        return getSoundAporee(
            location=location, loc_id=loc_id, distance=distance,
            catalog=catalog, query=query, tag=tag, max_return=max_return,
            year=year, season=season, time_of_day=time_of_day,
            duration=duration, exclude_from_location=exclude_from_location,
            slice_duration=slice_duration, slice_max_num=slice_max_num,
            probe_durations=probe_durations,
            probe_workers=probe_workers,
            output_df=output_df,
        )
    else:
        raise ValueError(
            f"Unsupported sound source {source!r}; choose 'freesound' or 'aporee'."
        )


def _getSoundFreesound(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        key: str = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int = 300,
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        output_df: bool = True,
) -> pd.DataFrame:

    """
        _getSoundFreesound (internal — call via :func:`getSound` with source='freesound')

        Fetch geotagged Freesound audio near a point, using Freesound API v2 search + geospatial filter.

        Notes:
        - Uses token authentication (API key) via Authorization header.
        - Returns preview URLs (mp3/ogg). Downloading original audio requires OAuth2.

        Args:
            location: (lon, lat) required.
            loc_id (int | str, optional): .
            distance (int): radius in meters (converted to km for Freesound geofilt).
            key (str): Freesound API key. If None, reads env var FREESOUND_API_KEY.
            query (str, optional): Freesound API query (e.g., 'traffic', '"bird song" -crow').
            tag: tag string or list of tags (used as filters).
            max_return: number of sounds to return (after post-filters).
            year: [Y] or (Y,) or (Y1, Y2) inclusive (filters by upload date "created").
            season (str): one of {"spring","summer","fall","autumn","winter"} (post-filter by created month).
            time_of_day (str): one of {"morning","afternoon","evening","night"} (post-filter by created hour).
            duration (int): maximum duration in seconds (<= duration). If you want a range, pass a tuple/list (min,max). (Default is 300)
            exclude_from_location (int, optional): Drop retrieved photos within a distance (in meter) from the given location.
            slice_duration (int, optional): Split the original sound signal into clips with the given duration.
            slice_max_num (int, optional): Maximum number of clips sliced from the original sound signal.
            output_df (bool): if True, return a pandas.DataFrame.

        Returns:
            dict | list[dict] | pandas.DataFrame
    """
    import os

    import requests

    from .utils.utils import parse_iso_created as _parse_created
    from .utils.utils import solr_year_range as _year_range

    if exclude_from_location is not None:
        drop_area = projection(location, r=distance)

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    api_key = key or os.getenv("FREESOUND_API_KEY")
    if not api_key:
        raise ValueError("Missing Freesound API key. Pass key=... or set env var FREESOUND_API_KEY.")

    lon, lat = location

    # meters -> km for geofilt d=<km>
    radius_km = max(float(distance) / 1000.0, 0.01)

    months = season_months(season)
    hours = tod_hours(time_of_day)

    # duration: allow int (max seconds) or tuple/list (min,max)
    dur_filter = None
    if duration is not None:
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            dmin = float(duration[0])
            dmax = float(duration[1])
            if dmax < dmin:
                dmin, dmax = dmax, dmin
            dur_filter = f"duration:[{dmin} TO {dmax}]"
        else:
            dmax = float(duration)
            dur_filter = f"duration:[0 TO {dmax}]"

    # -------------------------
    # Build request
    # -------------------------
    endpoint = "https://freesound.org/apiv2/search/"
    headers = {"Authorization": f"Token {api_key}"}  # token auth

    # Request useful fields, including previews (mp3/ogg URLs) and geotag.
    fields = ",".join(
        [
            "id",
            "name",
            "username",
            "license",
            "created",
            "duration",
            "geotag",
            "tags",
            "previews",
            "url",
            "num_downloads",
            "avg_rating",
            "description"
        ]
    )

    # Base filter parts
    filter_parts = []
    filter_parts.append("is_geotagged:1")
    filter_parts.append(f"{{!geofilt sfield=geotag pt={lat},{lon} d={radius_km}}}")

    # tag filters
    if tag:
        if isinstance(tag, (list, tuple)):
            for t in tag:
                t = str(t).strip()
                if t:
                    filter_parts.append(f"tag:{t}")
        else:
            t = str(tag).strip()
            if t:
                filter_parts.append(f"tag:{t}")

    if dur_filter:
        filter_parts.append(dur_filter)

    # year filter (created range): try with Z, retry without if API complains
    created_range_z = _year_range(year, with_z=True)
    created_range_noz = _year_range(year, with_z=False)

    qstr = query_string(query)

    def _do_request(created_range):
        fp = list(filter_parts)
        if created_range is not None:
            start, end = created_range
            fp.append(f"created:[{start} TO {end}]")
        params = {
            "query": qstr,  # empty query allowed
            "filter": " ".join(fp),
            "fields": fields,
            "page": 1,
            "page_size": min(150, max(50, max_return * 25)),
            "sort": "score",
        }
        return params

    session = requests.Session()

    # -------------------------
    # Fetch + post-filter
    # -------------------------
    results = []
    seen = set()
    max_pages = 150

    # First attempt (with Z)
    params = _do_request(created_range_z)

    for attempt in (1, 2):
        try:
            for page in range(1, max_pages + 1):
                params["page"] = page
                r = session.get(endpoint, params=params, headers=headers, timeout=60)

                if r.status_code == 400 and attempt == 1 and year is not None:
                    # likely date format issue; retry without Z
                    raise ValueError("Date filter rejected; retrying without 'Z'.")
                if r.status_code == 404:
                    break
                r.raise_for_status()
                data = r.json()

                page_results = data.get("results") or []
                if not page_results:
                    break

                for s in page_results:
                    sid = s.get("id")
                    if sid is None or sid in seen:
                        continue
                    seen.add(sid)

                    created_dt = _parse_created(s.get("created"))
                    if months and created_dt and created_dt.month not in months:
                        continue
                    if hours and created_dt and created_dt.hour not in hours:
                        continue

                    # Parse geotag "lat lon"
                    s_lat = s_lon = None
                    if s.get("geotag"):
                        parts = str(s["geotag"]).split()
                        if len(parts) == 2:
                            try:
                                s_lat = float(parts[0])
                                s_lon = float(parts[1])
                                if exclude_from_location is not None:
                                    if is_coordinate_in_bbox(s_lon, s_lat, drop_area):
                                        continue
                            except Exception:
                                pass

                    out = {
                        "loc_id": '',
                        "id": sid,
                        "name": s.get("name"),
                        "username": s.get("username"),
                        "license": s.get("license"),
                        "created": s.get("created"),
                        "duration": s.get("duration"),
                        "tags": s.get("tags"),
                        "geotag": s.get("geotag"),
                        "latitude": s_lat,
                        "longitude": s_lon,
                        "distance_m": haversine_m(lat, lon, s_lat, s_lon) if (s_lat is not None and s_lon is not None) else None,
                        "previews": s.get("previews"),
                        "url": s.get("url"),
                        "page_url": f"https://freesound.org/people/{s.get('username')}/sounds/{sid}/" if s.get("username") and sid else None,
                        "description": s.get("description"),
                        "num_downloads": s.get("num_downloads"),
                        "avg_rating": s.get("avg_rating"),
                        "slice": []
                    }
                    if loc_id is not None:
                        out["loc_id"] = loc_id
                    else:
                        del out["loc_id"]
                    if slice_duration is None:
                        del out["slice"]
                    else:
                        out["slice"] = sliced_duration(int(out["duration"]), slice_duration, slice_max_num)
                    results.append(out)
                    # if len(results) >= max_return:
                    #     break
                if not data.get("next"):
                    break
                if len(results) >= max_return:
                    break

            break  # success, don’t do second attempt
        except ValueError:
            # Retry without Z in created range
            if attempt == 1 and year is not None:
                params = _do_request(created_range_noz)
                continue
            raise

    # -------------------------
    # Return shape
    # -------------------------
    if output_df:
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values(by='distance_m', ascending=True)
        previews_df = df['previews'].apply(pd.Series)
        previews_df.columns = [f'{col}' for col in previews_df.columns]
        df = pd.concat([df.drop('previews', axis=1), previews_df], axis=1)
        return df.head(max_return)

    if max_return == 1:
        return results[0] if results else None
    return results


# --------------------------------------------------------------------
# Aporee source
# --------------------------------------------------------------------
def getSoundAporee(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        catalog: str | pd.DataFrame = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int | list | tuple = None,
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        probe_durations: bool = True,
        probe_workers: int = 8,
        output_df: bool = True,
) -> pd.DataFrame | dict | list | None:
    """Filter a Radio Aporee catalog (CSV or DataFrame) by spatial proximity.

    Aporee (radio aporee ::: maps) does not expose a public geo-query API the
    way Freesound does, so this helper takes a pre-built catalog of geotagged
    Aporee URLs and filters it with the same semantics as
    :func:`_getSoundFreesound`. The resulting DataFrame uses the same column
    names so the downstream ``GeoTaggedData`` / ``download_to_dir`` pipeline
    needs no changes.

    Args:
        location (list | tuple): (lon, lat) of the query point.
        loc_id (int | str, optional): Identifier for the query location.
        distance (int): Search radius in meters.
        catalog (str | pandas.DataFrame): Path to a CSV file or an in-memory
            DataFrame. Required columns: ``url``, ``latitude``, ``longitude``.
            Optional columns recognised by the filters: ``id``/``identifier``,
            ``name``/``title``, ``description``, ``tags``, ``created`` (ISO
            timestamp), ``duration_s``.
        query (str | list[str], optional): Substring(s) matched against
            ``name``/``title`` and ``description`` (case-insensitive). Skipped
            silently if neither column is present.
        tag (str | list[str], optional): Substring(s) matched against ``tags``
            (case-insensitive). Skipped if column is absent.
        max_return (int): Number of nearest sounds to return.
        year, season, time_of_day: Same semantics as :func:`getSound`. Applied
            against the ``created`` column if present.
        duration (int | list[int] | tuple[int]): Filter on ``duration_s`` if
            present. Pass an int for max-only or (min, max) for a range.
        exclude_from_location (int, optional): Drop rows inside this radius
            (m) around the query point — useful for "what's nearby but not
            *at* this exact spot".
        slice_duration (int, optional): Pre-compute clip windows on top of
            the chosen recording's ``duration_s`` (mirrors Freesound path).
        slice_max_num (int, optional): Cap on number of clips per recording.
        probe_durations (bool): If True (default) and ``slice_duration`` is
            requested but the catalog has no usable ``duration_s`` column,
            fetch each selected recording once with
            :func:`opensip.utils.utils.probe_audio_duration` to learn its
            length so slice windows can be computed. Set False to skip
            slicing instead (no per-row download).
        probe_workers (int): Number of parallel threads used when
            ``probe_durations=True``. Default 8.  Higher values reduce wall
            time proportionally up to your network bandwidth limit.
        output_df (bool): If True (default) return a ``pandas.DataFrame``.

    Returns:
        ``pandas.DataFrame``, ``dict``, ``list[dict]``, or ``None`` if the
        filtered catalog is empty.
    """
    import os

    from .utils.utils import (
        haversine_m,
        is_coordinate_in_bbox,
        parse_iso_created,
        probe_audio_duration,
        season_months,
        sliced_duration,
        tod_hours,
    )

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    if catalog is None:
        env_path = os.getenv("APOREE_CATALOG")
        if env_path:
            catalog = env_path
        else:
            raise ValueError(
                "source='aporee' requires a catalog (CSV path or DataFrame). "
                "Pass catalog=... or set APOREE_CATALOG env var."
            )

    if isinstance(catalog, str):
        df = pd.read_csv(catalog)
    elif isinstance(catalog, pd.DataFrame):
        df = catalog.copy()
    else:
        raise TypeError(
            "catalog must be a CSV path (str) or a pandas.DataFrame; "
            f"got {type(catalog).__name__}."
        )

    # Accept the alternate column names produced by fetch_aporee_catalog()
    # (which mirrors archive.org's `lat` / `lon` / `date` field names).
    _aliases = {"lat": "latitude", "lon": "longitude", "capture_time": "created"}
    for src, dst in _aliases.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = {"url", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Aporee catalog is missing required columns: {sorted(missing)}. "
            "At minimum it needs 'url', 'latitude', 'longitude' "
            "(or 'lat'/'lon' which will be renamed)."
        )

    if df.empty:
        return None if not output_df else pd.DataFrame()

    lon, lat = location

    # Coerce coords to float and drop rows that aren't usable.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "url"]).copy()
    if df.empty:
        return None if not output_df else pd.DataFrame()

    # -------------------------
    # Spatial filter
    # -------------------------
    df["distance_m"] = df.apply(
        lambda r: haversine_m(lat, lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    df = df[df["distance_m"] <= float(distance)]

    if exclude_from_location is not None and not df.empty:
        drop_area = projection(location, r=exclude_from_location)
        mask = df.apply(
            lambda r: not is_coordinate_in_bbox(
                float(r["longitude"]), float(r["latitude"]), drop_area
            ),
            axis=1,
        )
        df = df[mask]

    # -------------------------
    # Text / tag filters
    # -------------------------
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [str(t).strip().lower() for t in x if str(t).strip()]
        return [str(x).strip().lower()]

    qterms = _as_list(query)
    tterms = _as_list(tag)

    if qterms:
        text_cols = [c for c in ("name", "title", "description") if c in df.columns]
        if text_cols:
            haystack = df[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
            df = df[haystack.apply(lambda s: all(q in s for q in qterms))]

    if tterms and "tags" in df.columns:
        tag_haystack = df["tags"].astype(str).str.lower()
        df = df[tag_haystack.apply(lambda s: all(t in s for t in tterms))]

    # -------------------------
    # Time filters (only if `created` column is present)
    # -------------------------
    if "created" in df.columns and (year is not None or season or time_of_day):
        parsed = df["created"].apply(parse_iso_created)
        if year is not None:
            ys = year if isinstance(year, (list, tuple)) else [year]
            y1 = int(ys[0])
            y2 = int(ys[-1])
            if y2 < y1:
                y1, y2 = y2, y1
            df = df[parsed.apply(lambda dt: dt is not None and y1 <= dt.year <= y2)]
            parsed = parsed[df.index]
        if season:
            months = season_months(season)
            df = df[parsed.apply(lambda dt: dt is not None and dt.month in months)]
            parsed = parsed[df.index]
        if time_of_day:
            hours = tod_hours(time_of_day)
            df = df[parsed.apply(lambda dt: dt is not None and dt.hour in hours)]

    # -------------------------
    # Duration filter (only if `duration_s` column is present)
    # -------------------------
    if duration is not None and "duration_s" in df.columns:
        ds = pd.to_numeric(df["duration_s"], errors="coerce")
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            dmin, dmax = float(duration[0]), float(duration[1])
            if dmax < dmin:
                dmin, dmax = dmax, dmin
            df = df[(ds >= dmin) & (ds <= dmax)]
        else:
            df = df[ds <= float(duration)]

    if df.empty:
        return None if not output_df else pd.DataFrame()

    # -------------------------
    # Normalize output schema to match Freesound path
    # -------------------------
    df = df.sort_values(by="distance_m", ascending=True).head(max_return).reset_index(drop=True)

    # `id` column: prefer existing non-NaN values, then `identifier`, else
    # fall back to a coordinate-derived string so that downstream deduplication
    # (drop_duplicates, dict-lookup) always works even when the catalog CSV
    # round-tripped empty strings into NaN.
    _id_missing = "id" not in df.columns or df["id"].isna().all()
    if _id_missing:
        if "identifier" in df.columns and not df["identifier"].isna().all():
            df["id"] = df["identifier"].astype(str).where(
                df["identifier"].notna(), other=None
            )
            # Fill any remaining NaNs with coordinate key
            _nan_mask = df["id"].isna()
            if _nan_mask.any():
                df.loc[_nan_mask, "id"] = df[_nan_mask].apply(
                    lambda r: f"aporee_{r['latitude']:.6f}_{r['longitude']:.6f}", axis=1
                )
        else:
            df["id"] = df.apply(
                lambda r: f"aporee_{r['latitude']:.6f}_{r['longitude']:.6f}", axis=1
            )
    elif df["id"].isna().any():
        # Partial NaNs — fill only the missing ones with coordinate keys
        _nan_mask = df["id"].isna()
        df.loc[_nan_mask, "id"] = df[_nan_mask].apply(
            lambda r: f"aporee_{r['latitude']:.6f}_{r['longitude']:.6f}", axis=1
        )

    # Alias `url` as `preview-hq-mp3` so downstream ``download_to_dir`` works
    # without any branching.
    df["preview-hq-mp3"] = df["url"]

    if loc_id is not None:
        df["loc_id"] = loc_id
    elif "loc_id" not in df.columns:
        df["loc_id"] = ""

    # Optional slice column to mirror Freesound behavior. Aporee catalogs
    # often lack a `duration_s` column because that metadata isn't on the
    # site — probe each selected URL on-demand if requested, or skip
    # slicing with a clear warning.
    #
    # NOTE: treat a column that exists but is entirely NaN exactly the same
    # as a missing column (this happens when the catalog was saved/loaded
    # via CSV and fetch_aporee_catalog() wrote duration_s=None for all rows).
    if slice_duration is not None:
        _dur_numeric = (
            pd.to_numeric(df["duration_s"], errors="coerce")
            if "duration_s" in df.columns
            else pd.Series(dtype=float)
        )
        _dur_col_usable = _dur_numeric.notna().any()

        if not _dur_col_usable:
            if probe_durations:
                _n_to_probe = len(df)
                logger.info(
                    "Aporee catalog has no usable 'duration_s' values; probing %d "
                    "selected recordings with %d worker(s). "
                    "(Pass probe_durations=False to skip.)",
                    _n_to_probe, probe_workers,
                )
                # Parallel probe — much faster than serial .apply() for large
                # selections (e.g. 100+ recordings in a bbox).
                from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
                _durations: dict[int, float | None] = {}
                _probe_pbar = tqdm(
                    total=_n_to_probe, desc="probe durations", unit="rec",
                )
                with ThreadPoolExecutor(max_workers=max(1, probe_workers)) as _pool:
                    _futs = {
                        _pool.submit(probe_audio_duration, u): i
                        for i, u in enumerate(df["url"].tolist())
                    }
                    for _fut in _asc(_futs):
                        _durations[_futs[_fut]] = _fut.result()
                        _probe_pbar.update(1)
                _probe_pbar.close()
                df["duration_s"] = [_durations[i] for i in range(len(df))]
                _dur_numeric = pd.to_numeric(df["duration_s"], errors="coerce")
                _dur_col_usable = _dur_numeric.notna().any()
                # Summarise probe failures in one line rather than one WARNING
                # per URL (many Aporee recordings have been removed from IA).
                _n_probed = len(df)
                _n_ok = int(_dur_numeric.notna().sum())
                _n_fail = _n_probed - _n_ok
                if _n_fail > 0:
                    logger.warning(
                        "Aporee probe: %d/%d recordings could not be measured "
                        "(server error, 404, or network timeout) — those will "
                        "be skipped. Run enrich_aporee_catalog() to "
                        "pre-populate duration_s and avoid probing on every run.",
                        _n_fail, _n_probed,
                    )
                if _n_ok > 0:
                    logger.info("Aporee probe: %d/%d recordings successfully measured.", _n_ok, _n_probed)
            else:
                logger.warning(
                    "Aporee catalog has no usable 'duration_s' values and "
                    "probe_durations=False; skipping slice generation. "
                    "Run opensip.dataset.enrich_aporee_catalog() once to "
                    "permanently add duration_s to your CSV."
                )

        if _dur_col_usable:
            df["slice"] = _dur_numeric.apply(
                lambda d: sliced_duration(int(d), slice_duration, slice_max_num)
                if pd.notna(d) and float(d) > 0 else None
            )
            # Drop rows for which duration could not be determined so they
            # don't produce meaningless [[0, 0]] windows downstream.
            _no_slice = df["slice"].isna()
            if _no_slice.any():
                logger.debug(
                    "Dropping %d Aporee row(s) with undetermined duration "
                    "(probe failed or duration_s=0).",
                    _no_slice.sum(),
                )
                df = df[~_no_slice].copy()

    if output_df:
        return df

    records = df.to_dict(orient="records")
    if max_return == 1:
        return records[0] if records else None
    return records


def fix_aporee_urls(
        catalog: str | pd.DataFrame,
        out_path: str | None = None,
        workers: int = 8,
        timeout: float = 30.0,
        skip_existing: bool = True,
) -> pd.DataFrame:
    """Repair the ``url`` column of an Aporee catalog built with wrong URLs.

    ``fetch_aporee_catalog`` previously defaulted to ``verify_urls=False``,
    which constructed URLs as ``{identifier}/{identifier}.mp3``.  That
    pattern produces 404 errors for most Aporee items because the actual
    upload filename is unrelated to the identifier (e.g.
    ``201121engelbeckenwalk.mp3``).

    This function re-queries IA's metadata API for each row and writes the
    correct URL.  Lookups are parallelised with ``workers`` threads.

    Args:
        catalog (str | pandas.DataFrame): Path to an existing Aporee CSV or
            an in-memory DataFrame.  Must have an ``identifier`` column.
        out_path (str, optional): Write the corrected DataFrame here.  If
            ``None``, the DataFrame is returned without saving.
        workers (int): Thread-pool size for parallel metadata lookups.
            Default 8.
        timeout (float): Per-request HTTP timeout in seconds.
        skip_existing (bool): If ``True`` (default), rows whose ``url``
            already ends with the correct pattern (i.e. does *not* end with
            ``/{identifier}.mp3``) are left unchanged.  Set ``False`` to
            re-resolve every row.

    Returns:
        Corrected ``pandas.DataFrame``.
    """
    if isinstance(catalog, str):
        df = pd.read_csv(catalog)
    elif isinstance(catalog, pd.DataFrame):
        df = catalog.copy()
    else:
        raise TypeError(f"catalog must be a path or DataFrame; got {type(catalog).__name__}")

    if "identifier" not in df.columns:
        raise ValueError("catalog must have an 'identifier' column")

    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

    # Decide which rows need fixing
    if skip_existing and "url" in df.columns:
        def _is_bad(row):
            url = str(row.get("url", "") or "")
            ident = str(row.get("identifier", "") or "")
            return url.endswith(f"/{ident}.mp3") or not url

        needs_fix = df.apply(_is_bad, axis=1)
        indices = df.index[needs_fix].tolist()
    else:
        indices = df.index.tolist()

    if not indices:
        logger.info("fix_aporee_urls: all %d URLs appear correct, nothing to do.", len(df))
        if out_path:
            df.to_csv(out_path, index=False)
        return df

    logger.info(
        "fix_aporee_urls: resolving %d/%d URLs (workers=%d) …",
        len(indices), len(df), workers,
    )

    pbar = tqdm(total=len(indices), desc="fix URLs", unit="rec")

    def _resolve(idx):
        ident = str(df.at[idx, "identifier"])
        url = _ia_verify_mp3_url(ident, timeout=timeout)
        pbar.update(1)
        return idx, url

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_resolve, idx): idx for idx in indices}
        for fut in _asc(futs):
            idx, url = fut.result()
            df.at[idx, "url"] = url

    pbar.close()

    fixed = df.loc[indices, "url"].apply(
        lambda u: not str(u).endswith(f"/{df.at[df.index[df['url'] == u][0] if len(df[df['url'] == u]) else 0, 'identifier']}.mp3")
    ).sum()
    logger.info("fix_aporee_urls: done. %d/%d URLs updated.", len(indices), len(df))

    if out_path:
        df.to_csv(out_path, index=False)
        logger.info("fix_aporee_urls: saved to %s", out_path)

    return df


def enrich_aporee_catalog(
        catalog: str | pd.DataFrame,
        out_path: str | None = None,
        min_duration: float | None = None,
        skip_existing: bool = True,
        timeout: float = 60.0,
) -> pd.DataFrame:
    """Add a ``duration_s`` column to an Aporee catalog by probing each URL.

    Aporee URLs don't carry duration metadata, so this helper downloads each
    file once, reads its length with pydub (or mutagen as a fallback), and
    annotates the catalog. Optionally drops rows shorter than
    ``min_duration``.

    Use this once after building / updating your catalog so that subsequent
    :func:`getSoundAporee` calls with ``slice_duration`` can compute clip
    windows without paying the per-row probe cost every time.

    Args:
        catalog (str | pandas.DataFrame): CSV path or in-memory DataFrame.
            Must have a ``url`` column.
        out_path (str, optional): If provided, writes the enriched DataFrame
            back to this CSV path.
        min_duration (float, optional): Drop rows shorter than this many
            seconds (after probing). ``None`` keeps all rows.
        skip_existing (bool): If ``True`` (default) and ``duration_s`` is
            already populated for a row, leave it alone. Set ``False`` to
            re-probe every row.
        timeout (float): Per-URL request timeout (seconds).

    Returns:
        The enriched ``pandas.DataFrame``.
    """
    from .utils.utils import probe_audio_duration

    if isinstance(catalog, str):
        df = pd.read_csv(catalog)
    elif isinstance(catalog, pd.DataFrame):
        df = catalog.copy()
    else:
        raise TypeError(
            "catalog must be a CSV path (str) or a pandas.DataFrame; "
            f"got {type(catalog).__name__}."
        )

    if "url" not in df.columns:
        raise ValueError("Aporee catalog must have a 'url' column.")
    if "duration_s" not in df.columns:
        df["duration_s"] = pd.NA

    needs_probe = df.index if not skip_existing else df.index[df["duration_s"].isna()]
    logger.info(
        "enrich_aporee_catalog: probing %d / %d rows", len(needs_probe), len(df),
    )

    for i in tqdm(needs_probe, desc="probing", ncols=75):
        url = df.at[i, "url"]
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        d = probe_audio_duration(url, timeout=timeout)
        if d is not None:
            df.at[i, "duration_s"] = round(float(d), 2)

    if min_duration is not None:
        before = len(df)
        df = df[
            df["duration_s"].notna()
            & (pd.to_numeric(df["duration_s"], errors="coerce") >= float(min_duration))
        ].reset_index(drop=True)
        logger.info(
            "enrich_aporee_catalog: dropped %d rows shorter than %ss",
            before - len(df), min_duration,
        )

    if out_path is not None:
        df.to_csv(out_path, index=False)
        logger.info("enrich_aporee_catalog: wrote %d rows to %s", len(df), out_path)

    return df


# --------------------------------------------------------------------
# fetch_aporee_catalog — pulls geolocated metadata from Internet Archive
# (radio aporee ::: maps live in the `radio-aporee-maps` IA collection).
# --------------------------------------------------------------------
_IA_SCRAPE = "https://archive.org/services/search/v1/scrape"
_IA_METADATA = "https://archive.org/metadata"
_IA_DOWNLOAD = "https://archive.org/download"
_APOREE_COLLECTION = "radio-aporee-maps"

_IA_FIELDS = [
    "identifier",
    "title",
    "date",
    "subject",
    "description",
    "latitude",
    "longitude",
    "licenseurl",
]


def _season_for(month: int, lat: float | None, southern: bool = False) -> str:
    """Hemisphere-aware season label. Mirrors the helper in scratch_aporee.py."""
    if month is None:
        return ""
    if southern or (lat is not None and lat < 0):
        month = ((month - 1 + 6) % 12) + 1
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


_IA_RETRY_LIMIT = 5
_IA_RETRY_DELAY = 3   # base delay in seconds; doubles each attempt


def _ia_verify_mp3_url(identifier: str, timeout: float = 30.0) -> str:
    """Look up the actual MP3 filename via IA's metadata API.

    Aporee items store the recording under the original upload filename
    (e.g. ``201121engelbeckenwalk.mp3``), which is unrelated to the IA
    identifier.  This function queries ``/metadata/{identifier}/files``,
    prefers non-derivative MP3s, and retries up to :data:`_IA_RETRY_LIMIT`
    times with exponential back-off before falling back to the (usually
    wrong) ``<identifier>.mp3`` pattern.
    """
    import time as _time
    import requests

    base = f"{_IA_DOWNLOAD}/{identifier}"
    api_url = f"{_IA_METADATA}/{identifier}/files"
    headers = {"User-Agent": "openSIP/0.x (+aporee fetcher)"}

    for attempt in range(1, _IA_RETRY_LIMIT + 1):
        try:
            r = requests.get(api_url, timeout=timeout, headers=headers)
            r.raise_for_status()
            files = r.json().get("result", [])
            # Prefer originals over derivatives (same logic as reference script)
            mp3s = [
                f["name"] for f in files
                if f.get("name", "").endswith(".mp3")
                and f.get("source", "") != "derivative"
            ]
            if not mp3s:
                mp3s = [f["name"] for f in files if f.get("name", "").endswith(".mp3")]
            if mp3s:
                return f"{base}/{mp3s[0]}"
            break   # API responded but no MP3 found — fall through to fallback
        except Exception as e:
            delay = _IA_RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < _IA_RETRY_LIMIT:
                logger.debug(
                    "IA mp3 lookup attempt %d/%d failed for %s (%s) — retrying in %ds",
                    attempt, _IA_RETRY_LIMIT, identifier, e, delay,
                )
                _time.sleep(delay)
            else:
                logger.debug(
                    "IA mp3 lookup failed for %s after %d attempts: %s",
                    identifier, _IA_RETRY_LIMIT, e,
                )

    # Fallback: identifier-based guess (will be wrong for most Aporee items)
    return f"{base}/{identifier}.mp3"


def fetch_aporee_catalog(
        bbox: tuple[float, float, float, float] | list | None = None,
        year: int | tuple[int, int] | list | None = None,
        hour: int | tuple[int, int] | list | None = None,
        season: str | list[str] | None = None,
        southern: bool = False,
        rows: int = 0,
        verify_urls: bool = True,
        verify_workers: int = 8,
        out_path: str | None = None,
        enrich_durations: bool = False,
        min_duration: float | None = None,
        timeout: float = 60.0,
        page_size: int = 500,
) -> pd.DataFrame:
    """Fetch the Aporee sound-map catalog from Internet Archive.

    All Aporee field recordings are mirrored on archive.org under the
    ``radio-aporee-maps`` collection. This helper queries IA's Scrape API
    with optional server-side ``bbox`` / ``year`` filters and applies
    ``hour`` / ``season`` filters client-side, then returns a DataFrame in
    the schema :func:`getSoundAporee` expects.

    Args:
        bbox: ``(lat_min, lon_min, lat_max, lon_max)`` to filter server-side.
            Pass ``None`` for the whole world.
        year: Single year (``2021``) or inclusive range (``(2018, 2022)``).
            Filtered server-side via IA's ``date`` field.
        hour: UTC hour or inclusive range (``(9, 17)`` or ``(22, 4)`` for
            midnight-wrap). Applied client-side against ``capture_time``.
        season: One of ``"spring" | "summer" | "autumn"/"fall" | "winter"``,
            or a list. Hemisphere is auto-detected from each row's latitude;
            pass ``southern=True`` to force southern interpretation.
        southern (bool): Force southern-hemisphere season interpretation.
        rows (int): Maximum number of records to fetch. ``0`` means all.
        verify_urls (bool): If True (default), query IA's metadata API for
            each identifier to find the exact MP3 filename.  Aporee items
            use the original upload name (e.g.
            ``201121engelbeckenwalk.mp3``), which is *not* predictable from
            the identifier, so the old ``<identifier>.mp3`` fallback
            produces 404s.  Lookups are parallelised with
            ``verify_workers`` threads so the overhead is modest (~5–30 s
            for 10 k records on a typical connection).  Set ``False`` only
            if you already have a verified catalog and want to skip the
            network round-trip.
        verify_workers (int): Number of parallel threads used when
            ``verify_urls=True``. Default 8.
        out_path (str, optional): If provided, write the resulting DataFrame
            to this CSV path.
        enrich_durations (bool): If True, also probe each fetched URL for
            its duration via :func:`enrich_aporee_catalog` (slow — one
            request per row).
        min_duration (float, optional): When ``enrich_durations=True``,
            drop rows shorter than this many seconds.
        timeout (float): Per-request HTTP timeout (seconds).
        page_size (int): Records per Scrape-API page (min 100).

    Returns:
        ``pandas.DataFrame`` with columns:
        ``identifier, id, latitude, longitude, url, capture_time, created,
        year, month, hour, season, title, name, description, tags, licence,
        duration_s``. ``id`` aliases ``identifier`` and ``name`` aliases
        ``title`` for compatibility with :func:`getSoundAporee`'s filters.
    """
    import requests

    # Build query
    query = f"collection:{_APOREE_COLLECTION}"
    whole_world = (-90.0, -180.0, 90.0, 180.0)
    bbox_t = tuple(bbox) if bbox is not None else whole_world
    if bbox_t != whole_world:
        lat_min, lon_min, lat_max, lon_max = bbox_t
        query += (
            f" AND lat:[{lat_min:g} TO {lat_max:g}]"
            f" AND lon:[{lon_min:g} TO {lon_max:g}]"
        )
    if year is not None:
        if isinstance(year, (list, tuple)):
            y1, y2 = int(year[0]), int(year[-1])
            if y2 < y1:
                y1, y2 = y2, y1
        else:
            y1 = y2 = int(year)
        query += f" AND date:[{y1}-01-01T00:00:00Z TO {y2}-12-31T23:59:59Z]"

    # Normalize hour filter
    hour_range: tuple[int, int] | None = None
    if hour is not None:
        if isinstance(hour, (list, tuple)):
            hour_range = (int(hour[0]), int(hour[-1]))
        else:
            hour_range = (int(hour), int(hour))

    # Normalize season filter to a set of months
    season_set: set[int] | None = None
    if season is not None:
        from .utils.utils import season_months as _sm
        names = season if isinstance(season, (list, tuple)) else [season]
        season_set = set()
        for s in names:
            season_set |= _sm(s)

    logger.info("fetch_aporee_catalog: query=%s", query)
    headers = {"User-Agent": "openSIP/0.x (+aporee fetcher)"}
    page_size = max(100, int(page_size))

    items: list[dict] = []
    cursor: str | None = None
    fetched = 0
    skip_no_geo = skip_hour = skip_season = 0

    pbar = tqdm(desc="fetch aporee", unit="rec", disable=False)
    while True:
        if rows and fetched >= rows:
            break
        page_n = page_size if not rows else min(page_size, rows - fetched)
        params = {
            "q": query,
            "fields": ",".join(_IA_FIELDS),
            "count": max(100, page_n),
        }
        if cursor:
            params["cursor"] = cursor

        import time as _time
        data = None
        for _attempt in range(1, _IA_RETRY_LIMIT + 1):
            try:
                r = requests.get(_IA_SCRAPE, params=params, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as _exc:
                _delay = _IA_RETRY_DELAY * (2 ** (_attempt - 1))
                if _attempt < _IA_RETRY_LIMIT:
                    logger.warning(
                        "fetch_aporee_catalog: scrape API attempt %d/%d failed (%s) "
                        "— retrying in %ds", _attempt, _IA_RETRY_LIMIT, _exc, _delay,
                    )
                    _time.sleep(_delay)
                else:
                    raise RuntimeError(
                        f"IA scrape API unavailable after {_IA_RETRY_LIMIT} retries: {_exc}"
                    ) from _exc
        if "items" not in data:
            raise RuntimeError(f"IA scrape API error: {data}")

        docs = data["items"]
        next_cursor = data.get("cursor")
        if not docs:
            break

        for doc in docs:
            try:
                lat_v = float(doc.get("latitude") or "")
                lon_v = float(doc.get("longitude") or "")
            except (ValueError, TypeError):
                skip_no_geo += 1
                continue

            ident = doc.get("identifier", "")
            title = doc.get("title", "")
            ctime = (doc.get("date") or "").strip()
            description = doc.get("description", "")
            licence = doc.get("licenseurl", "")
            subject = doc.get("subject", "")
            # `subject` may come back as a list — collapse to comma-string
            if isinstance(subject, list):
                subject = ",".join(str(s) for s in subject)

            # Client-side hour filter
            if hour_range is not None:
                hh = _ia_extract_hour(ctime)
                if hh is None:
                    skip_hour += 1
                    continue
                h_start, h_end = hour_range
                if h_start <= h_end:
                    matched = h_start <= hh <= h_end
                else:
                    matched = hh >= h_start or hh <= h_end
                if not matched:
                    skip_hour += 1
                    continue

            # Client-side season filter
            if season_set is not None:
                mm = _ia_extract_month(ctime)
                if mm is None:
                    skip_season += 1
                    continue
                row_month = mm
                if southern or lat_v < 0:
                    row_month = ((mm - 1 + 6) % 12) + 1
                if row_month not in season_set:
                    skip_season += 1
                    continue

            items.append({
                "identifier": ident,
                "id": ident,                        # alias for getSoundAporee
                "latitude": lat_v,
                "longitude": lon_v,
                "url": None,                        # resolved below
                "capture_time": ctime,              # script's column name
                "created": ctime,                   # getSoundAporee filter name
                "title": title,
                "name": title,                      # alias for getSoundAporee.query
                "description": description,
                "tags": subject,                    # IA's `subject` -> our tags
                "licence": licence,
                "duration_s": None,
            })
            fetched += 1
            pbar.update(1)
            if rows and fetched >= rows:
                break

        # Cursor is the source of truth for "more pages available" — IA's
        # scrape API can return a partial page mid-stream, so don't bail
        # out just because len(docs) < page_n.
        if not next_cursor:
            break
        cursor = next_cursor

    pbar.close()
    logger.info(
        "fetch_aporee_catalog: kept %d, skipped no_geo=%d hour=%d season=%d",
        len(items), skip_no_geo, skip_hour, skip_season,
    )

    # ── URL resolution ──────────────────────────────────────────────────────
    # Aporee items store the recording under the uploader's original filename
    # (e.g. "201121engelbeckenwalk.mp3"), which is unrelated to the IA
    # identifier.  We must call IA's metadata API to discover it.
    # Parallelise so the ~1 request/record overhead stays manageable.
    if items:
        if verify_urls:
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

            n_workers = max(1, int(verify_workers))
            logger.info(
                "fetch_aporee_catalog: resolving %d MP3 URLs "
                "(verify_workers=%d) …",
                len(items), n_workers,
            )
            pbar_url = tqdm(total=len(items), desc="resolve URLs", unit="rec")

            def _resolve(idx_ident):
                idx, ident = idx_ident
                url = _ia_verify_mp3_url(ident, timeout=timeout)
                pbar_url.update(1)
                return idx, url

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {
                    pool.submit(_resolve, (i, item["identifier"])): i
                    for i, item in enumerate(items)
                }
                for fut in _as_completed(futs):
                    idx, url = fut.result()
                    items[idx]["url"] = url

            pbar_url.close()
        else:
            # Fast but unreliable — filename ≠ identifier for most items.
            import warnings as _warnings
            _warnings.warn(
                "fetch_aporee_catalog: verify_urls=False uses "
                "<identifier>.mp3 as the download filename, which is wrong "
                "for most Aporee items and will produce 404 errors on "
                "playback/download. Use verify_urls=True (the default) to "
                "resolve the real filename from IA's metadata API.",
                UserWarning,
                stacklevel=2,
            )
            for item in items:
                ident = item["identifier"]
                item["url"] = f"{_IA_DOWNLOAD}/{ident}/{ident}.mp3"

    df = pd.DataFrame(items)
    if df.empty:
        if out_path:
            df.to_csv(out_path, index=False)
        return df

    # Enrich with derived time columns (year/month/hour/season) for
    # downstream convenience. ``parse_iso_created`` handles missing
    # fractional-seconds gracefully.
    from .utils.utils import parse_iso_created
    parsed = df["capture_time"].apply(parse_iso_created)
    df["year"] = parsed.apply(lambda d: d.year if d is not None else None)
    df["month"] = parsed.apply(lambda d: d.month if d is not None else None)
    df["hour"] = parsed.apply(lambda d: d.hour if d is not None else None)
    df["season"] = df.apply(
        lambda r: _season_for(r["month"], r["latitude"], southern) if r["month"] else "",
        axis=1,
    )

    if enrich_durations:
        df = enrich_aporee_catalog(df, min_duration=min_duration, timeout=timeout)

    if out_path:
        df.to_csv(out_path, index=False)
        logger.info("fetch_aporee_catalog: wrote %d rows to %s", len(df), out_path)

    return df


def _ia_extract_hour(s: str) -> int | None:
    """Pull the UTC hour out of an IA ``date`` field, or None."""
    from datetime import datetime
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).hour
        except ValueError:
            continue
    return None


def _ia_extract_month(s: str) -> int | None:
    """Pull the month out of an IA ``date`` field, or None."""
    from datetime import datetime
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s.strip(), fmt).month
        except ValueError:
            continue
    return None
