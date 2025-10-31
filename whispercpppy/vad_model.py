from __future__ import annotations

from pathlib import Path

from whispercpppy.model import DownloadResult, default_download_path, stream_download

AVAILABLE_VAD_MODELS: list[str] = ["silero-v5.1.2"]
VAD_BASE_URL = "https://huggingface.co/ggml-org/whisper-vad"
VAD_URL_PREFIX = "resolve/main/ggml"


def is_valid_vad_model(model: str) -> bool:
    return model in AVAILABLE_VAD_MODELS


def build_vad_model_url(model: str) -> str:
    return f"{VAD_BASE_URL}/{VAD_URL_PREFIX}-{model}.bin"


def prepare_vad_download(
    model: str,
    models_dir: Path | str | None = None,
) -> tuple[str, Path]:
    if not is_valid_vad_model(model):
        raise ValueError(f"Invalid model: {model}")

    if isinstance(models_dir, str):
        models_dir = Path(models_dir)

    download_dir = models_dir or default_download_path()
    download_dir.mkdir(parents=True, exist_ok=True)
    url = build_vad_model_url(model)
    savepath = download_dir / model
    return url, savepath


def download_vad_model(
    model: str,
    models_dir: Path | str | None = None,
    overwrite: bool = False,
    timeout: float | None = None,
) -> DownloadResult:
    url, savepath = prepare_vad_download(model, models_dir)
    existed = savepath.is_file()
    if existed and not overwrite:
        return DownloadResult(model=model, url=url, filepath=savepath, existed=True)

    print(f"downloading VAD model {model} to {savepath.resolve()}")
    stream_download(url, savepath, timeout=timeout)
    return DownloadResult(model=model, url=url, filepath=savepath, existed=existed)


__all__ = ["download_vad_model", "AVAILABLE_VAD_MODELS"]
