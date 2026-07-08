"""PlacePulse image perception predictor.

This module intentionally downloads PlacePulse checkpoints on demand instead of
bundling large pickle-based ``.pth`` files in the package.
"""

from __future__ import annotations

import sys
import types
import urllib.request
from pathlib import Path

DEFAULT_PLACEPULSE_REPO_ID = "Jiani11/human-perception-place-pulse"
DEFAULT_PLACEPULSE_CACHE_DIR = Path.home() / ".cache" / "opensip" / "placepulse"


class Net:  # pragma: no cover - real class is built lazily with torch imports
    """Placeholder for pickle module compatibility."""


def _build_placepulse_net_class():
    """Return the architecture class expected by the PlacePulse pickle files."""
    try:
        import torch
        from torch import nn
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:
        raise ImportError(
            "PlacePulse inference requires torch and torchvision. Install with:\n"
            "    pip install \"openSIP[placepulse]\""
        ) from exc

    class _Net(nn.Module):
        def __init__(self, num_class):
            super().__init__()
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            num_fc = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(
                nn.Linear(num_fc, 512, bias=True),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=True),
                nn.ReLU(True),
                nn.Linear(256, num_class, bias=True),
            )
            nn.init.xavier_uniform_(self.model.heads.head[0].weight)
            nn.init.xavier_uniform_(self.model.heads.head[2].weight)
            nn.init.xavier_uniform_(self.model.heads.head[4].weight)

        def forward(self, x):
            return self.model(x)

    _Net.__name__ = "Net"
    _Net.__qualname__ = "Net"
    _Net.__module__ = "Model_01"
    return _Net


def _install_model_01_shim() -> None:
    """Install a minimal ``Model_01`` module for loading upstream pickles."""
    net_cls = _build_placepulse_net_class()
    module = types.ModuleType("Model_01")
    module.Net = net_cls
    sys.modules["Model_01"] = module


def _download_hf_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / filename
    if out.exists() and out.stat().st_size > 0:
        return out

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        urllib.request.urlretrieve(url, out)
        return out

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    )


class PlacePulsePredictor:
    """Predict PlacePulse lively and boring scores for image paths.

    The default model source is
    ``Jiani11/human-perception-place-pulse`` on Hugging Face. Only
    ``lively.pth`` and ``boring.pth`` are downloaded. Scores follow the upstream
    convention: each value is on a 0-10 scale.
    """

    model_files = {"lively": "lively.pth", "boring": "boring.pth"}

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_PLACEPULSE_REPO_ID,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_PLACEPULSE_CACHE_DIR
        self.device = device
        self._models = None
        self._transform = None

    def _resolve_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        if self._models is not None:
            return self._models, self._transform, self._resolve_device()

        try:
            import torch
            from PIL import Image  # noqa: F401
            from torchvision import transforms as T
        except ImportError as exc:
            raise ImportError(
                "PlacePulse inference requires torch, torchvision, and Pillow. Install with:\n"
                "    pip install \"openSIP[placepulse]\""
            ) from exc

        _install_model_01_shim()
        device = self._resolve_device()
        models = {}
        for name, filename in self.model_files.items():
            path = _download_hf_file(self.repo_id, filename, self.cache_dir)
            model = torch.load(path, map_location=device, weights_only=False)
            model = model.to(device)
            model.eval()
            models[name] = model

        transform = T.Compose(
            [
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._models = models
        self._transform = transform
        return models, transform, device

    def predict_batch(self, image_paths: list[str]) -> list[dict[str, float]]:
        import torch
        from PIL import Image

        models, transform, device = self._load()
        batch = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch.append(transform(img))
        if not batch:
            return []

        tensor = torch.stack(batch, dim=0).to(device)
        out: dict[str, list[float]] = {}
        with torch.no_grad():
            for name, model in models.items():
                pred = torch.softmax(model(tensor), dim=1)[:, 1] * 10.0
                out[name] = [round(float(x), 2) for x in pred.detach().cpu().tolist()]

        return [
            {"lively": out["lively"][i], "boring": out["boring"][i]}
            for i in range(len(image_paths))
        ]

    def predict(self, image_paths: list[str]) -> list[dict[str, float]]:
        return self.predict_batch(image_paths)


__all__ = [
    "DEFAULT_PLACEPULSE_REPO_ID",
    "DEFAULT_PLACEPULSE_CACHE_DIR",
    "PlacePulsePredictor",
]
