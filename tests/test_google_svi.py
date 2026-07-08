from __future__ import annotations

import base64

import pandas as pd
import pytest

from opensip import GeoTaggedData


class _FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload or {}
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload


def test_get_svi_from_locations_google_requires_google_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_STREETVIEW_API_KEY", raising=False)
    gtd = GeoTaggedData(locations=[[-83.073, 42.348]])

    with pytest.raises(ValueError) as exc:
        gtd.get_svi_from_locations(source="google")

    assert "GOOGLE_STREETVIEW_API_KEY" in str(exc.value)


def test_get_svi_from_locations_google_with_api_key(monkeypatch):
    image_bytes = b"fake-jpeg"
    calls = []

    def fake_retry_request(url):
        calls.append(url)
        if "streetview/metadata" in url:
            return _FakeResponse(
                {
                    "status": "OK",
                    "pano_id": "pano-123",
                    "date": "2024-06",
                    "location": {"lat": 42.3481, "lng": -83.0731},
                }
            )
        if "maps.googleapis.com/maps/api/streetview" in url:
            return _FakeResponse(content=image_bytes)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("opensip.dataset.retry_request", fake_retry_request)
    gtd = GeoTaggedData(locations=[[-83.073, 42.348]])

    gtd.get_svi_from_locations(
        source="google",
        key="google-key",
        distance=75,
        height=400,
        width=600,
    )

    assert len(gtd.svis["data"]) == 1
    assert base64.b64decode(gtd.svis["data"][0]) == image_bytes
    assert gtd.svis["id"] == ["pano-123"]
    assert gtd.svi_metadata.loc[0, "captured_at"] == "2024-06-1-1"
    assert gtd.svi_metadata.loc[0, "url"].endswith("&key=key")
    assert "key=google-key" not in gtd.svi_metadata.loc[0, "url"]
    assert any("streetview/metadata" in url for url in calls)
    assert any("size=600x400" in url for url in calls)


def test_get_svi_from_locations_accumulates_and_dedupes(monkeypatch):
    first = (
        ["img-a-bytes", "img-b-bytes"],
        pd.DataFrame(
            [
                {"loc_id": 0, "id": "img-a", "url": "https://image.example/a.jpg", "captured_at": "2024-1-1-12"},
                {"loc_id": 0, "id": "img-b", "url": "https://image.example/b.jpg", "captured_at": "2024-1-2-12"},
            ]
        ),
    )
    second = (
        ["img-b-bytes", "img-c-bytes"],
        pd.DataFrame(
            [
                {"loc_id": 0, "id": "img-b", "url": "https://image.example/b.jpg", "captured_at": "2024-1-2-12"},
                {"loc_id": 0, "id": "img-c", "url": "https://image.example/c.jpg", "captured_at": "2024-1-3-12"},
            ]
        ),
    )

    responses = [first, second]

    def fake_getsv(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("opensip.dataset.getSV", fake_getsv)
    gtd = GeoTaggedData(locations=[[0.0, 0.0]])

    gtd.get_svi_from_locations(source="mapillary", key="mapillary-key")
    gtd.get_svi_from_locations(source="mapillary", key="mapillary-key", distance=100)

    assert gtd.svis["id"] == ["img-a", "img-b", "img-c"]
    assert gtd.svi_metadata["id"].tolist() == ["img-a", "img-b", "img-c"]


def test_get_svi_from_locations_can_reset(monkeypatch):
    first = (
        ["img-a-bytes"],
        pd.DataFrame(
            [{"loc_id": 0, "id": "img-a", "url": "https://image.example/a.jpg", "captured_at": "2024-1-1-12"}]
        ),
    )
    second = (
        ["img-b-bytes"],
        pd.DataFrame(
            [{"loc_id": 0, "id": "img-b", "url": "https://image.example/b.jpg", "captured_at": "2024-1-2-12"}]
        ),
    )

    responses = [first, second]

    def fake_getsv(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("opensip.dataset.getSV", fake_getsv)
    gtd = GeoTaggedData(locations=[[0.0, 0.0]])

    gtd.get_svi_from_locations(source="mapillary", key="mapillary-key")
    gtd.get_svi_from_locations(source="mapillary", key="mapillary-key", reset=True)

    assert gtd.svis["id"] == ["img-b"]
    assert gtd.svi_metadata["id"].tolist() == ["img-b"]
