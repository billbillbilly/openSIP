"""Smoke + behavior tests for GeoTaggedData (no network)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from opensip.dataset import GeoTaggedData
from opensip.models import get_yunet_model_path


def test_payload_dicts_are_independent():
    """Regression: __init__ used chained assignment and aliased the three dicts."""
    g = GeoTaggedData()
    g.svis["data"].append("svi-img")
    g.photos["data"].append("flickr-img")
    assert g.svis["data"] == ["svi-img"]
    assert g.photos["data"] == ["flickr-img"]
    assert g.audios["data"] == []  # untouched


def test_construct_units_from_nested_list():
    g = GeoTaggedData(locations=[[-83.235572, 42.348092], [-83.235154, 42.348806]])
    assert g.units is not None
    assert len(g.units) == 2
    assert "loc_id" in g.units.columns


def test_construct_units_from_dataframe():
    df = pd.DataFrame({"longitude": [-83.235572], "latitude": [42.348092]})
    g = GeoTaggedData(locations=df)
    assert g.units is not None
    assert len(g.units) == 1


def test_construct_units_invalid_list_raises():
    with pytest.raises(ValueError):
        GeoTaggedData(locations=[1, 2])  # not a nested list


def test_construct_units_invalid_dict_raises():
    with pytest.raises(ValueError):
        GeoTaggedData(locations={"x": [1], "y": [2]})


def test_construct_units_unsupported_type_raises():
    with pytest.raises(TypeError):
        GeoTaggedData(locations=42)


def test_filter_outdoor_filters_audio_with_loaded_buildings():
    g = GeoTaggedData()
    g.units = gpd.GeoDataFrame(
        {"loc_id": [0]},
        geometry=[Polygon([(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)])],
        crs="EPSG:4326",
    )
    g.audios = {
        "loc_id": [0, 1],
        "id": ["inside", "outside"],
        "data": ["inside.mp3", "outside.mp3"],
        "path": [None, None],
    }
    g.audio_metadata = pd.DataFrame(
        [
            {"loc_id": 0, "id": "inside", "url": "inside.mp3", "latitude": 0.0, "longitude": 0.0},
            {"loc_id": 1, "id": "outside", "url": "outside.mp3", "latitude": 0.01, "longitude": 0.01},
        ]
    )

    result = g.filter_outdoor(data="audio", building_source="loaded", buffer_m=0)

    assert result == {"audio_dropped": 1, "photo_dropped": 0}
    assert g.audios["id"] == ["outside"]
    assert g.audio_metadata["id"].tolist() == ["outside"]


def test_filter_outdoor_filters_flickr_photos_with_loaded_buildings():
    g = GeoTaggedData()
    g.units = gpd.GeoDataFrame(
        {"loc_id": [0]},
        geometry=[Polygon([(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)])],
        crs="EPSG:4326",
    )
    g.photos = {
        "loc_id": [0, 1],
        "id": ["inside-photo", "outside-photo"],
        "data": ["inside.jpg", "outside.jpg"],
        "path": [None, None],
    }
    g.photo_metadata = pd.DataFrame(
        [
            {
                "loc_id": 0,
                "id": "inside-photo",
                "url": "inside.jpg",
                "latitude": 0.0,
                "longitude": 0.0,
            },
            {
                "loc_id": 1,
                "id": "outside-photo",
                "url": "outside.jpg",
                "latitude": 0.01,
                "longitude": 0.01,
            },
        ]
    )

    result = g.filter_outdoor(data="photo", building_source="loaded", buffer_m=0)

    assert result == {"audio_dropped": 0, "photo_dropped": 1}
    assert g.photos["id"] == ["outside-photo"]
    assert g.photo_metadata["id"].tolist() == ["outside-photo"]


def test_filter_outdoor_filters_audio_and_photos_together():
    g = GeoTaggedData()
    g.units = gpd.GeoDataFrame(
        {"loc_id": [0]},
        geometry=[Polygon([(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)])],
        crs="EPSG:4326",
    )
    g.audios = {"loc_id": [0], "id": ["inside-audio"], "data": ["inside.mp3"], "path": [None]}
    g.audio_metadata = pd.DataFrame(
        [{"loc_id": 0, "id": "inside-audio", "url": "inside.mp3", "latitude": 0.0, "longitude": 0.0}]
    )
    g.photos = {"loc_id": [0], "id": ["inside-photo"], "data": ["inside.jpg"], "path": [None]}
    g.photo_metadata = pd.DataFrame(
        [{"loc_id": 0, "id": "inside-photo", "url": "inside.jpg", "latitude": 0.0, "longitude": 0.0}]
    )

    result = g.filter_outdoor(data="both", building_source="loaded", buffer_m=0)

    assert result == {"audio_dropped": 1, "photo_dropped": 1}
    assert g.audios["id"] == []
    assert g.photos["id"] == []


def test_get_photo_from_location_can_remove_selfies():
    g = GeoTaggedData(locations=[[0.0, 0.0]])
    flickr_rows = pd.DataFrame(
        [
            {
                "loc_id": 0,
                "id": "selfie",
                "url": "https://example.com/selfie.jpg",
                "latitude": 0.0,
                "longitude": 0.0,
            },
            {
                "loc_id": 0,
                "id": "street",
                "url": "https://example.com/street.jpg",
                "latitude": 0.0,
                "longitude": 0.0,
            },
        ]
    )

    seen_model_paths = []

    def fake_is_selfie_photo(model_path=None, img_url=None):
        seen_model_paths.append(model_path)
        return img_url.endswith("selfie.jpg")

    with patch("opensip.dataset.getPhoto", return_value=flickr_rows), patch(
        "opensip.dataset.is_selfie_photo",
        side_effect=fake_is_selfie_photo,
    ):
        g.get_photo_from_location(key="fake-key", remove_selfies=True)

    assert g.photos["id"] == ["street"]
    assert g.photos["data"] == ["https://example.com/street.jpg"]
    assert g.photos["path"] == [None]
    assert g.photo_metadata["id"].tolist() == ["street"]
    assert len(seen_model_paths) == 2
    assert all(str(path).endswith("face_detection_yunet_2023mar.onnx") for path in seen_model_paths)


def test_get_photo_from_location_exclude_personal_photo_alias_removes_selfies():
    g = GeoTaggedData(locations=[[0.0, 0.0]])
    flickr_rows = pd.DataFrame(
        [
            {
                "loc_id": 0,
                "id": "selfie",
                "url": "https://example.com/selfie.jpg",
                "latitude": 0.0,
                "longitude": 0.0,
            }
        ]
    )

    with patch("opensip.dataset.getPhoto", return_value=flickr_rows), patch(
        "opensip.dataset.is_selfie_photo",
        return_value=True,
    ):
        with pytest.warns(DeprecationWarning, match="remove_selfies"):
            g.get_photo_from_location(key="fake-key", exclude_personal_photo=True)

    assert g.photos["id"] == []
    assert g.photo_metadata is None


def test_builtin_yunet_model_is_packaged():
    model_path = get_yunet_model_path()
    data = model_path.read_bytes()

    assert model_path.name == "face_detection_yunet_2023mar.onnx"
    assert len(data) > 100_000
    assert data[:2] == b"\x08\x06"
    assert b"pytorch" in data[:64]
