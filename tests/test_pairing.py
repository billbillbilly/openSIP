from __future__ import annotations

import sys
import types

import pandas as pd
import pytest
from PIL import Image

from opensip import AudioImagePairDataset, GeoTaggedData
from opensip.pairing import SOUND_EVENT_CLASSES, _prepare_event_rank_image


def _gtd() -> GeoTaggedData:
    gtd = GeoTaggedData(locations=[[0.0, 0.0]])
    gtd.audios = {
        "loc_id": [0, 0],
        "id": ["aud-1", "aud-2"],
        "data": ["https://freesound.example/aud-1.mp3", "https://aporee.example/aud-2.mp3"],
        "path": ["", ""],
        "slice": [[0, 30000], [0, 30000]],
    }
    gtd.audio_metadata = pd.DataFrame(
        [
            {
                "id": "aud-1",
                "created": "2024-01-15T12:00:00",
                "latitude": 0.0,
                "longitude": 0.0,
            },
            {
                "id": "aud-2",
                "created": "2024-01-20T12:00:00",
                "latitude": 0.0,
                "longitude": 0.0,
            },
        ]
    )
    gtd.svi_metadata = pd.DataFrame(
        [
            {
                "loc_id": 0,
                "id": "img-mapillary",
                "image_url": "https://image.example/mapillary.jpg",
                "captured_at": "2024-1-16-13",
                "latitude": 0.0,
                "longitude": 0.001,
            },
            {
                "loc_id": 0,
                "id": "img-google",
                "image_url": "https://image.example/google.jpg",
                "captured_at": "2024-01-1-1",
                "latitude": 0.001,
                "longitude": 0.0,
            },
        ]
    )
    return gtd


def test_build_pairs_parses_source_timestamps():
    apd = AudioImagePairDataset(_gtd())

    pairs = apd.build_pairs(image_source="svi")

    assert len(pairs) == 4
    assert set(pairs["ts_precision"]) == {"datetime", "year-month"}
    assert set(pairs["image_source"]) == {"mapillary", "google"}
    assert pairs["clip_start_ms"].tolist() == [0, 0, 0, 0]
    assert pairs["clip_end_ms"].tolist() == [30000, 30000, 30000, 30000]


def test_dataset_can_own_geotagged_data():
    apd = AudioImagePairDataset(locations=[[0.0, 0.0]])
    apd.audios = _gtd().audios
    apd.audio_metadata = _gtd().audio_metadata
    apd.svi_metadata = _gtd().svi_metadata

    pairs = apd.build_pairs(image_source="svi")

    assert len(apd.units) == 1
    assert len(pairs) == 4
    assert apd.gtd is apd._gtd


def test_dataset_delegates_geotagged_methods():
    apd = AudioImagePairDataset(locations=[[0.0, 0.0]])

    assert apd.construct_units() is None
    assert "loc_id" in apd.units.columns


def test_dataset_rejects_mixed_gtd_and_constructor_state():
    try:
        AudioImagePairDataset(_gtd(), locations=[[0.0, 0.0]])
    except ValueError as exc:
        assert "either gtd" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_filter_pairs_threshold_and_top_k():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]
    apd.pairs["n_sub_clips"] = [2, 2, 1, 2]

    apd.filter_pairs_by_similarity(min_similarity=0.2, min_sub_clips=2, top_k_per_audio=1)

    assert len(apd.pairs) == 2
    assert apd.pairs["audio_id"].tolist() == ["aud-1", "aud-2"]
    assert apd.pairs["image_id"].tolist() == ["img-mapillary", "img-google"]


def test_filter_pairs_alias_still_works():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]

    apd.filter_pairs(min_similarity=0.8)

    assert apd.pairs["similarity"].tolist() == [0.9, 0.8]


def test_finalize_pairs_adds_schema_distance_and_similarity_columns():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]
    apd.pairs["semantic_similarity"] = [1.0, 0.5, 0.2, 0.1]
    apd.pairs["aq_similarity"] = [0.4, 0.3, 0.2, 0.1]
    apd.pairs["aq_similarity_type"] = ["direct", "event_rank", "placepulse", "direct"]

    final_df = apd.finalize_pairs()

    assert final_df["pair_id"].iloc[0] == "aud-1:0__img-mapillary"
    assert final_df["audio_node_id"].iloc[0] == "audio:aud-1:0"
    assert final_df["image_node_id"].iloc[0] == "image:img-mapillary"
    assert final_df["embedding_similarity"].tolist() == [0.9, 0.1, 0.8, 0.7]
    assert final_df["distance_m"].notna().all()
    assert final_df["distance_m"].iloc[0] == pytest.approx(111.2, rel=0.02)
    assert "aq_similarity_direct" in final_df.columns
    assert "aq_similarity_event_rank" in final_df.columns
    assert "aq_similarity_placepulse" in final_df.columns
    assert final_df["aq_similarity_event_rank"].iloc[1] == pytest.approx(0.3)


def test_to_graph_exports_audio_image_edges_with_distance_and_time():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]

    graph = apd.to_graph()

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert graph.nodes["audio:aud-1:0"]["node_type"] == "audio_clip"
    attrs = graph.edges["audio:aud-1:0", "image:img-mapillary"]
    assert attrs["pair_id"] == "aud-1:0__img-mapillary"
    assert attrs["distance_m"] == pytest.approx(111.2, rel=0.02)
    assert attrs["time_diff_days"] == pytest.approx(1.0416666, rel=0.02)
    assert attrs["embedding_similarity"] == pytest.approx(0.9)


def test_export_graph_writes_graphml(tmp_path):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]

    out = apd.export_graph(tmp_path / "pairs.graphml")

    assert out.endswith("pairs.graphml")
    assert (tmp_path / "pairs.graphml").exists()


def test_compute_embedding_similarity_alias_exists():
    apd = AudioImagePairDataset(_gtd())

    assert callable(apd.compute_embedding_similarity)
    assert callable(apd.compute_similarity)


def _mock_imagebind_for_embedding(monkeypatch, tmp_path, audio_calls):
    import torch

    from opensip.imagebind.models.imagebind_model import ModalityType

    class FakeImageBind:
        def __call__(self, inputs):
            if ModalityType.AUDIO in inputs:
                return {ModalityType.AUDIO: torch.tensor([[1.0, 0.0]])}
            batch = inputs[ModalityType.VISION].shape[0]
            return {ModalityType.VISION: torch.tensor([[1.0, 0.0]] * batch)}

    def fake_audio_loader(paths, device, **kwargs):
        audio_calls.append({"paths": list(paths), **kwargs})
        return torch.ones((len(paths), 1, 1, 2, 2))

    def fake_vision_loader(paths, device, **kwargs):
        return torch.ones((len(paths), 3, 224, 224))

    monkeypatch.setattr("opensip.pairing._load_imagebind", lambda device: FakeImageBind())
    monkeypatch.setattr("opensip.pairing._resolve_device", lambda device: "cpu")
    monkeypatch.setattr("opensip.pairing._audio_duration", lambda path: 6.0)
    monkeypatch.setattr("opensip.imagebind.data.load_and_transform_audio_data", fake_audio_loader)
    monkeypatch.setattr("opensip.imagebind.data.load_and_transform_vision_data", fake_vision_loader)

    audio = tmp_path / "audio.wav"
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    audio.write_bytes(b"fake")
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    return audio, img1, img2


def test_compute_embedding_similarity_sample_strategy_uses_imagebind_sampling(
    tmp_path,
    monkeypatch,
):
    audio_calls = []
    audio, img1, img2 = _mock_imagebind_for_embedding(monkeypatch, tmp_path, audio_calls)
    monkeypatch.setattr(
        "opensip.pairing._trim_audio_segment",
        lambda audio_path, start_ms, end_ms, out_path: audio_path,
    )
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["audio_url"] = str(audio)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    pairs = apd.compute_embedding_similarity(
        audio_strategy="sample",
        clips_per_audio=5,
        output_dir=tmp_path / "work",
    )

    assert len(audio_calls) == 2
    assert all(call["clips_per_video"] == 5 for call in audio_calls)
    assert all(call["clip_duration"] == 2.0 for call in audio_calls)
    assert pairs["n_sub_clips"].tolist() == [5, 5, 5, 5]
    assert pairs["similarity"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_compute_embedding_similarity_exhaustive_strategy_uses_all_subclips(
    tmp_path,
    monkeypatch,
):
    audio_calls = []
    audio, img1, img2 = _mock_imagebind_for_embedding(monkeypatch, tmp_path, audio_calls)
    subclips = [tmp_path / f"sub_{i}.wav" for i in range(3)]
    monkeypatch.setattr(
        "opensip.pairing._split_audio_to_subclips",
        lambda audio_path, sub_clip_sec, start_ms, end_ms, work_dir: subclips,
    )
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["audio_url"] = str(audio)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    pairs = apd.compute_embedding_similarity(
        audio_strategy="exhaustive",
        output_dir=tmp_path / "work",
    )

    assert len(audio_calls) == 6
    assert all(call["clips_per_video"] == 1 for call in audio_calls)
    assert pairs["n_sub_clips"].tolist() == [3, 3, 3, 3]
    assert pairs["similarity"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_compute_embedding_similarity_rejects_unknown_audio_strategy():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    with pytest.raises(ValueError, match="audio_strategy"):
        apd.compute_embedding_similarity(audio_strategy="all")


def test_compute_semantic_similarity_from_rankings():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    pairs = apd.compute_semantic_similarity(
        audio_events={
            ("aud-1", 0): ["Traffic", "Bird", "Water"],
            ("aud-2", 0): ["Bird", "Water", "Traffic"],
        },
        image_events={
            "img-mapillary": ["Traffic", "Water", "Bird"],
            "img-google": ["Water", "Bird", "Traffic"],
        },
        top_k=3,
    )

    first = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert first["semantic_ndcg_at_k"] == pytest.approx(0.972, abs=0.001)
    assert first["semantic_mrr"] == 1.0
    assert first["semantic_spearman"] == pytest.approx(0.5)
    assert "semantic_similarity" in pairs.columns


def test_compute_semantic_similarity_can_infer_audio_events_with_soundaqnet(
    tmp_path,
    monkeypatch,
):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    aud1 = tmp_path / "aud1.wav"
    aud2 = tmp_path / "aud2.wav"
    aud1.write_bytes(b"fake")
    aud2.write_bytes(b"fake")
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-1", "audio_url"] = str(aud1)
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-2", "audio_url"] = str(aud2)

    class FakeSoundAQnet:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

        def predict_from_audio(self, audio_files, batch_size, num_workers, show_progress=False):
            return pd.DataFrame(
                [
                    {
                        "clip_id": "aud1",
                        "event_probs": {"Traffic": 0.9, "Bird": 0.1, "Water": 0.2},
                        "top_events": ["Traffic", "Water", "Bird"],
                    },
                    {
                        "clip_id": "aud2",
                        "event_probs": {"Traffic": 0.1, "Bird": 0.9, "Water": 0.2},
                        "top_events": ["Bird", "Water", "Traffic"],
                    },
                ]
            )

    fake_module = types.ModuleType("soundaqnet")
    fake_module.SoundAQnet = FakeSoundAQnet
    monkeypatch.setitem(sys.modules, "soundaqnet", fake_module)

    pairs = apd.compute_semantic_similarity(
        image_events={
            "img-mapillary": ["Traffic", "Water", "Bird"],
            "img-google": ["Water", "Bird", "Traffic"],
        },
        top_k=3,
    )

    first = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert first["semantic_mrr"] == pytest.approx(1.0)
    assert first["semantic_ndcg_at_k"] == pytest.approx(1.0)


def test_compute_semantic_similarity_raises_cleanly_without_soundaqnet():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    with pytest.raises(ImportError, match="pip install openSIP"):
        apd.compute_semantic_similarity(
            image_events={"img-mapillary": ["Traffic"], "img-google": ["Traffic"]},
        )


def test_infer_image_event_ranks_uses_zeroshot_backend(tmp_path, monkeypatch):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    seen = {}

    def fake_zeroshot(image_paths_by_id, event_classes, **kwargs):
        seen.update(image_paths_by_id)
        return (
            {image_id: list(event_classes) for image_id in image_paths_by_id},
            {
                image_id: {event: float(i) for i, event in enumerate(event_classes)}
                for image_id in image_paths_by_id
            },
        )

    monkeypatch.setattr("opensip.pairing._zeroshot_image_event_ranks", fake_zeroshot)

    ranks = apd.infer_image_event_ranks(
        image_event_backend="zeroshot",
        output_dir=tmp_path / "work",
        checkpoint_path=tmp_path / "image_events.jsonl",
    )

    assert ranks["img-mapillary"] == SOUND_EVENT_CLASSES
    assert ranks["img-google"] == SOUND_EVENT_CLASSES
    assert set(seen) == {"img-mapillary", "img-google"}
    assert "image_event_rank_1" in apd.pairs.columns
    assert "image_event_score_Traffic" in apd.pairs.columns


def test_event_rank_image_pre_resize_limits_long_edge(tmp_path):
    src = tmp_path / "large.jpg"
    Image.new("RGB", (3000, 1200), color="green").save(src)

    prepared = _prepare_event_rank_image(
        src,
        tmp_path,
        "prepared",
        max_image_edge=512,
    )

    assert prepared is not None
    with Image.open(prepared) as img:
        assert max(img.size) <= 512


def test_infer_image_event_ranks_rejects_non_zeroshot_backend():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    with pytest.raises(ValueError, match="zeroshot"):
        apd.infer_image_event_ranks(image_event_backend="vlm")


def test_compute_semantic_similarity_infers_image_events_with_zeroshot(tmp_path, monkeypatch):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    def fake_zeroshot(image_paths_by_id, event_classes, **kwargs):
        return (
            {image_id: list(event_classes) for image_id in image_paths_by_id},
            {image_id: {} for image_id in image_paths_by_id},
        )

    monkeypatch.setattr("opensip.pairing._zeroshot_image_event_ranks", fake_zeroshot)

    pairs = apd.compute_semantic_similarity(
        audio_events={
            ("aud-1", 0): SOUND_EVENT_CLASSES,
            ("aud-2", 0): SOUND_EVENT_CLASSES,
        },
        infer_audio_events=False,
        infer_image_events=True,
        image_event_backend="zeroshot",
        output_dir=tmp_path / "work",
        top_k=5,
    )

    first = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert first["semantic_ndcg_at_k"] == pytest.approx(1.0)
    assert first["semantic_mrr"] == pytest.approx(1.0)


def test_compute_semantic_similarity_requires_image_inference_true():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    with pytest.raises(ValueError, match="infer_image_events"):
        apd.compute_semantic_similarity(
            audio_events={("aud-1", 0): SOUND_EVENT_CLASSES},
            infer_audio_events=False,
            infer_image_events=False,
        )


def test_compute_aq_similarity_from_direct_vectors():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    pairs = apd.compute_aq_similarity(
        audio_aq={
            ("aud-1", 0): {"pleasant": 1.0, "eventful": 3.0, "calm": 2.0},
            ("aud-2", 0): {"pleasant": 3.0, "eventful": 1.0, "calm": 2.0},
        },
        image_aq={
            "img-mapillary": {"pleasant": 1.0, "eventful": 3.0, "calm": 2.0},
            "img-google": {"pleasant": 3.0, "eventful": 1.0, "calm": 2.0},
        },
    )

    perfect = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert perfect["aq_rmse"] == pytest.approx(0.0)
    assert perfect["aq_pearson"] == pytest.approx(1.0)
    assert perfect["aq_similarity"] == pytest.approx(1.0)


def test_compute_aq_similarity_can_derive_from_event_rankings():
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    event_aq = pd.DataFrame(
        [
            {"event": "Traffic", "pleasant": 1.0, "eventful": 3.0},
            {"event": "Bird", "pleasant": 3.0, "eventful": 1.0},
            {"event": "Water", "pleasant": 2.0, "eventful": 2.0},
        ]
    )
    pairs = apd.compute_aq_similarity(
        audio_events={("aud-1", 0): ["Traffic", "Water"]},
        image_events={"img-mapillary": ["Traffic", "Water"]},
        event_aq_scores=event_aq,
        top_k=2,
    )

    perfect = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert perfect["aq_rmse"] == pytest.approx(0.0)


def test_compute_aq_similarity_can_infer_audio_aq_with_soundaqnet(
    tmp_path,
    monkeypatch,
):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    aud1 = tmp_path / "aud1.wav"
    aud2 = tmp_path / "aud2.wav"
    aud1.write_bytes(b"fake")
    aud2.write_bytes(b"fake")
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-1", "audio_url"] = str(aud1)
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-2", "audio_url"] = str(aud2)

    class FakeSoundAQnet:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

        def predict_from_audio(self, audio_files, batch_size, num_workers, show_progress=False):
            return pd.DataFrame(
                [
                    {"clip_id": "aud1", "pleasant": 1.0, "eventful": 3.0, "calm": 2.0},
                    {"clip_id": "aud2", "pleasant": 3.0, "eventful": 1.0, "calm": 2.0},
                ]
            )

    fake_module = types.ModuleType("soundaqnet")
    fake_module.SoundAQnet = FakeSoundAQnet
    monkeypatch.setitem(sys.modules, "soundaqnet", fake_module)

    pairs = apd.compute_aq_similarity(
        image_aq={
            "img-mapillary": {"pleasant": 1.0, "eventful": 3.0, "calm": 2.0},
            "img-google": {"pleasant": 3.0, "eventful": 1.0, "calm": 2.0},
        },
        aq_dimensions=["pleasant", "eventful", "calm"],
    )

    perfect = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert perfect["aq_rmse"] == pytest.approx(0.0)
    assert perfect["aq_pearson"] == pytest.approx(1.0)


def test_soundaqnet_prediction_is_reused_for_semantic_and_aq(tmp_path, monkeypatch):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    aud1 = tmp_path / "aud1.wav"
    aud2 = tmp_path / "aud2.wav"
    aud1.write_bytes(b"fake")
    aud2.write_bytes(b"fake")
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-1", "audio_url"] = str(aud1)
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-2", "audio_url"] = str(aud2)
    calls = {"predict": 0}

    class FakeSoundAQnet:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

        def predict_from_audio(self, audio_files, batch_size, num_workers, show_progress=False):
            calls["predict"] += 1
            return pd.DataFrame(
                [
                    {
                        "clip_id": "aud1",
                        "event_probs": {"Traffic": 0.9, "Water": 0.1},
                        "pleasant": 1.0,
                        "eventful": 3.0,
                        "calm": 2.0,
                    },
                    {
                        "clip_id": "aud2",
                        "event_probs": {"Traffic": 0.1, "Water": 0.9},
                        "pleasant": 2.0,
                        "eventful": 2.0,
                        "calm": 1.0,
                    },
                ]
            )

    fake_module = types.ModuleType("soundaqnet")
    fake_module.SoundAQnet = FakeSoundAQnet
    monkeypatch.setitem(sys.modules, "soundaqnet", fake_module)

    apd.compute_semantic_similarity(
        image_events={
            "img-mapillary": ["Traffic", "Water"],
            "img-google": ["Water", "Traffic"],
        },
        top_k=2,
    )
    apd.compute_aq_similarity(
        image_aq={
            "img-mapillary": {"pleasant": 1.0, "eventful": 3.0, "calm": 2.0},
            "img-google": {"pleasant": 2.0, "eventful": 2.0, "calm": 1.0},
        },
        aq_dimensions=["pleasant", "eventful", "calm"],
    )

    assert calls["predict"] == 1


def test_compute_aq_similarity_can_infer_image_aq_from_image_event_ranks(
    tmp_path,
    monkeypatch,
):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    def fake_zeroshot(image_paths_by_id, event_classes, **kwargs):
        return (
            {image_id: ["Traffic", "Water"] for image_id in image_paths_by_id},
            {image_id: {} for image_id in image_paths_by_id},
        )

    monkeypatch.setattr("opensip.pairing._zeroshot_image_event_ranks", fake_zeroshot)
    event_aq = pd.DataFrame(
        [
            {"event": "Traffic", "pleasant": 1.0, "eventful": 3.0},
            {"event": "Water", "pleasant": 2.0, "eventful": 2.0},
        ]
    )

    pairs = apd.compute_aq_similarity(
        audio_aq={
            ("aud-1", 0): {"pleasant": 1.5, "eventful": 2.5},
            ("aud-2", 0): {"pleasant": 1.5, "eventful": 2.5},
        },
        event_aq_scores=event_aq,
        top_k=2,
        output_dir=tmp_path / "work",
    )

    first = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert first["aq_rmse"] == pytest.approx(0.0)
    assert "image_event_rank_1" in apd.pairs.columns


def test_infer_image_placepulse_aq_maps_lively_boring_to_aq(tmp_path):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    def fake_placepulse(paths):
        return [{"lively": 0.8, "boring": 0.2} for _ in paths]

    image_aq = apd.infer_image_placepulse_aq(
        placepulse_predictor=fake_placepulse,
        output_dir=tmp_path / "work",
        checkpoint_path=tmp_path / "placepulse.jsonl",
    )

    assert image_aq["img-mapillary"] == {"vibrant": 0.8, "monotonous": 0.2}
    assert image_aq["img-google"] == {"vibrant": 0.8, "monotonous": 0.2}
    assert "image_placepulse_lively" in apd.pairs.columns
    assert "image_aq_vibrant" in apd.pairs.columns
    assert (tmp_path / "placepulse.jsonl").exists()


def test_compute_aq_similarity_can_compare_soundaqnet_with_placepulse(
    tmp_path,
    monkeypatch,
):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    aud1 = tmp_path / "aud1.wav"
    aud2 = tmp_path / "aud2.wav"
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    aud1.write_bytes(b"fake")
    aud2.write_bytes(b"fake")
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-1", "audio_url"] = str(aud1)
    apd.pairs.loc[apd.pairs["audio_id"] == "aud-2", "audio_url"] = str(aud2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)

    class FakeSoundAQnet:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

        def predict_from_audio(self, audio_files, batch_size, num_workers, show_progress=False):
            return pd.DataFrame(
                [
                    {"clip_id": "aud1", "vibrant": 0.9, "monotonous": 0.1},
                    {"clip_id": "aud2", "vibrant": 0.2, "monotonous": 0.8},
                ]
            )

    def fake_placepulse(paths):
        return [{"lively": 0.9, "boring": 0.1} for _ in paths]

    fake_module = types.ModuleType("soundaqnet")
    fake_module.SoundAQnet = FakeSoundAQnet
    monkeypatch.setitem(sys.modules, "soundaqnet", fake_module)

    pairs = apd.compute_aq_similarity(
        infer_image_placepulse=True,
        placepulse_predictor=fake_placepulse,
        aq_dimensions=["vibrant", "monotonous"],
        output_dir=tmp_path / "work",
    )

    perfect = pairs[
        (pairs["audio_id"] == "aud-1") & (pairs["image_id"] == "img-mapillary")
    ].iloc[0]
    assert perfect["aq_rmse"] == pytest.approx(0.0)
    assert perfect["aq_pearson"] == pytest.approx(1.0)
    assert "image_placepulse_boring" in apd.pairs.columns


def test_infer_image_placepulse_aq_uses_default_huggingface_predictor(
    tmp_path,
    monkeypatch,
):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(img1)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(img2)
    seen = {}

    class FakePlacePulsePredictor:
        def __init__(self, repo_id, cache_dir=None, device=None):
            seen["repo_id"] = repo_id
            seen["cache_dir"] = cache_dir
            seen["device"] = device

        def predict_batch(self, paths):
            return [{"lively": 0.7, "boring": 0.3} for _ in paths]

    monkeypatch.setattr("opensip.placepulse.PlacePulsePredictor", FakePlacePulsePredictor)

    image_aq = apd.infer_image_placepulse_aq(
        output_dir=tmp_path / "work",
        placepulse_cache_dir=tmp_path / "cache",
        placepulse_device="cpu",
    )

    assert seen == {
        "repo_id": "Jiani11/human-perception-place-pulse",
        "cache_dir": tmp_path / "cache",
        "device": "cpu",
    }
    assert image_aq["img-mapillary"] == {"vibrant": 0.7, "monotonous": 0.3}


def test_infer_image_placepulse_aq_scores_panorama_perspective_views(tmp_path):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    pano = tmp_path / "pano.jpg"
    regular = tmp_path / "regular.jpg"
    Image.new("RGB", (1024, 256), color="green").save(pano)
    Image.new("RGB", (256, 256), color="yellow").save(regular)
    apd.pairs.loc[apd.pairs["image_id"] == "img-mapillary", "image_url"] = str(pano)
    apd.pairs.loc[apd.pairs["image_id"] == "img-google", "image_url"] = str(regular)
    calls = {"n": 0}

    def fake_placepulse(paths):
        out = []
        for _ in paths:
            calls["n"] += 1
            out.append({"lively": float(calls["n"]), "boring": float(10 - calls["n"])})
        return out

    image_aq = apd.infer_image_placepulse_aq(
        placepulse_predictor=fake_placepulse,
        output_dir=tmp_path / "work",
        panorama_strategy="auto",
        panorama_headings=[0, 180],
        panorama_size=(128, 128),
        panorama_min_aspect_ratio=1.8,
    )

    assert calls["n"] == 3
    assert image_aq["img-mapillary"] == {"vibrant": 1.5, "monotonous": 8.5}
    assert image_aq["img-google"] == {"vibrant": 3.0, "monotonous": 7.0}
    pano_rows = apd.pairs[apd.pairs["image_id"] == "img-mapillary"]
    assert pano_rows["image_placepulse_n_views"].tolist() == [2, 2]


def test_export_metadata_only(tmp_path):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")

    csv_path = apd.export(tmp_path, data=False)

    assert (tmp_path / "pairs_metadata.csv").exists()
    assert csv_path.endswith("pairs_metadata.csv")
    written = pd.read_csv(csv_path)
    assert len(written) == 4


def test_save_load_round_trip(tmp_path):
    apd = AudioImagePairDataset(_gtd())
    apd.build_pairs(image_source="svi")
    apd.pairs["similarity"] = [0.9, 0.1, 0.8, 0.7]

    apd.save(tmp_path)
    restored = AudioImagePairDataset.load(tmp_path)

    assert len(restored.pairs) == 4
    assert restored._gtd.audios["id"] == ["aud-1", "aud-2"]
