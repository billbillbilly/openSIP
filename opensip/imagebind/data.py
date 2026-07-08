# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from https://github.com/facebookresearch/ImageBind (Apache 2.0)
# Trimmed to vision + audio only; pytorchvideo dependency removed.

import logging
import math
from importlib.resources import files

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # milliseconds


# ---------------------------------------------------------------------------
# Minimal replacement for pytorchvideo.data.clip_sampling.ConstantClipsPerVideoSampler
# ---------------------------------------------------------------------------

class _ConstantClipsPerVideoSampler:
    """Sample *clips_per_video* non-overlapping clips of *clip_duration* seconds.

    Mimics the pytorchvideo ``ConstantClipsPerVideoSampler`` interface used by
    :func:`get_clip_timepoints`.
    """

    def __init__(self, clip_duration: float, clips_per_video: int) -> None:
        self._clip_duration = clip_duration
        self._clips_per_video = clips_per_video
        self._clip_index = 0

    def __call__(self, last_end_time: float, duration: float, annotation=None):
        # Reset when a new audio starts (caller passes last_end_time=0.0)
        if last_end_time == 0.0:
            self._clip_index = 0

        n = self._clips_per_video
        cd = self._clip_duration

        if n <= 1 or duration <= cd:
            self._clip_index = 0
            return 0.0, min(cd, duration), 0, 0, True

        # Space clips evenly: first at t=0, last ending at t=duration
        step = (duration - cd) / max(n - 1, 1)
        start = self._clip_index * step
        end = start + cd
        is_last = self._clip_index >= n - 1
        self._clip_index += 1
        return start, end, 0, 0, is_last


# ---------------------------------------------------------------------------
# Helpers (kept from original data.py)
# ---------------------------------------------------------------------------

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    try:
        import torchaudio as _torchaudio  # noqa: PLC0415
        import torch as _torch  # noqa: PLC0415
    except OSError as exc:
        raise OSError(
            "torchaudio failed to load — this usually means it was compiled against a "
            "different version of torch than the one currently installed.\n"
            "Fix: pip install torchaudio==$(python -c 'import torch; print(torch.__version__)')\n"
            f"Original error: {exc}"
        ) from exc

    waveform -= waveform.mean()
    fbank = _torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    fbank = fbank.transpose(0, 1)
    n_frames = fbank.size(1)
    p = target_length - n_frames
    if abs(p) / max(n_frames, 1) > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and target_length(%d).",
            n_frames,
            target_length,
        )
    if p > 0:
        fbank = _torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, :target_length]
    return fbank.unsqueeze(0)


def get_clip_timepoints(clip_sampler, duration):
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_and_transform_vision_data(
    image_paths,
    device,
    max_image_edge=1024,
    max_image_pixels=None,
):
    """Load and preprocess images for ImageBind vision encoder.

    Parameters
    ----------
    image_paths:
        List of local file paths to load.
    device:
        Torch device string.
    max_image_edge:
        If set, images whose longer edge exceeds this pixel count are
        proportionally downscaled with PIL before the standard 224-px crop
        pipeline.  This avoids decoding huge street-view panoramas at full
        size when ImageBind will later resize/crop to 224 px anyway. ``None``
        keeps the full image.
    max_image_pixels:
        Backward-compatible alias for ``max_image_edge``. The old name was
        misleading: this value has always limited the longer edge, not the
        total pixel count.
    """
    if image_paths is None:
        return None

    if max_image_pixels is not None:
        max_image_edge = max_image_pixels

    import torch as _torch  # noqa: PLC0415
    from PIL import Image as _Image  # noqa: PLC0415
    from PIL import ImageFile as _ImageFile  # noqa: PLC0415
    from torchvision import transforms as _transforms  # noqa: PLC0415

    # Disable PIL's decompression-bomb guard — we resize to 224 px immediately,
    # so even very large panoramas are safe to load.
    _ImageFile.LOAD_TRUNCATED_IMAGES = True
    _Image.MAX_IMAGE_PIXELS = None

    data_transform = _transforms.Compose([
        _transforms.Resize(224, interpolation=_transforms.InterpolationMode.BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    outputs = []
    for path in image_paths:
        with open(path, "rb") as f:
            image = _Image.open(f).convert("RGB")
        # Optional pre-resize before the standard 224-crop transform to save
        # memory on street-view panoramas and large Flickr images.
        if max_image_edge is not None:
            w, h = image.size
            if w > max_image_edge or h > max_image_edge:
                scale = max_image_edge / max(w, h)
                image = image.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    _Image.BICUBIC,
                )
        outputs.append(data_transform(image).to(device))
    return _torch.stack(outputs, dim=0)


def load_and_transform_text_data(texts, device, context_length=77):
    """Tokenize text prompts for the ImageBind text encoder."""
    if texts is None:
        return None

    from opensip.imagebind.models.multimodal_preprocessors import SimpleTokenizer  # noqa: PLC0415

    bpe_path = files("opensip.imagebind") / "bpe" / "bpe_simple_vocab_16e6.txt.gz"
    tokenizer = SimpleTokenizer(str(bpe_path), context_length=context_length)
    return tokenizer(texts, context_length=context_length).to(device)


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    """Load and preprocess audio files for ImageBind audio encoder.

    Each file is resampled to *sample_rate* Hz, converted to a log-mel
    spectrogram, and split into *clips_per_video* clips of *clip_duration*
    seconds.  The result shape is ``(N_files, clips_per_video, 1, mel_bins,
    target_length)``.
    """
    if audio_paths is None:
        return None

    try:
        import torchaudio as _torchaudio  # noqa: PLC0415
        import torch as _torch  # noqa: PLC0415
        from torchvision import transforms as _transforms  # noqa: PLC0415
    except OSError as exc:
        raise OSError(
            "torchaudio failed to load — this usually means it was compiled against a "
            "different version of torch than the one currently installed.\n"
            "Fix: pip install torchaudio==$(python -c 'import torch; print(torch.__version__)')\n"
            f"Original error: {exc}"
        ) from exc

    audio_outputs = []
    for audio_path in audio_paths:
        clip_sampler = _ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        try:
            waveform, sr = _torchaudio.load(audio_path)
        except RuntimeError:
            # macOS may need an explicit backend when the default (AVFoundation)
            # doesn't recognise the file format (e.g. plain WAV sub-clips).
            # Try soundfile first, then sox.
            loaded = False
            for _backend in ("soundfile", "sox"):
                try:
                    waveform, sr = _torchaudio.load(audio_path, backend=_backend)
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                raise
        if sr != sample_rate:
            waveform = _torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        clips = []
        for t0, t1 in timepoints:
            chunk = waveform[
                :,
                int(t0 * sample_rate): int(t1 * sample_rate),
            ]
            clips.append(waveform2melspec(chunk, sample_rate, num_mel_bins, target_length))

        normalize = _transforms.Normalize(mean=mean, std=std)
        clips = _torch.stack([normalize(c).to(device) for c in clips], dim=0)
        audio_outputs.append(clips)

    return _torch.stack(audio_outputs, dim=0)
