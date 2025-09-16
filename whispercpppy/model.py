from dataclasses import dataclass
from pathlib import Path

import requests

AVAILABLE_MODELS: list[str] = [
    "tiny",
    "tiny.en",
    "tiny-q5_1",
    "tiny.en-q5_1",
    "tiny-q8_0",
    "base",
    "base.en",
    "base-q5_1",
    "base.en-q5_1",
    "base-q8_0",
    "small",
    "small.en",
    "small.en-tdrz",
    "small-q5_1",
    "small.en-q5_1",
    "small-q8_0",
    "medium",
    "medium.en",
    "medium-q5_0",
    "medium.en-q5_0",
    "medium-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
]


@dataclass(frozen=True)
class DownloadResult:
    model: str
    url: str
    dest: Path
    existed: bool


def get_script_path() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def default_download_path(
    script_dir: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    sd = script_dir or get_script_path()
    wd = cwd or Path.cwd()
    return wd if sd.name == "bin" else sd


def is_valid_model(model: str) -> bool:
    return model in AVAILABLE_MODELS


def model_filename(model: str) -> str:
    return f"ggml-{model}.bin"


def build_model_url(model: str) -> str:
    if "tdrz" in model:
        base_url = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
        prefix = "resolve/main/ggml"
    else:
        base_url = "https://huggingface.co/ggerganov/whisper.cpp"
        prefix = "resolve/main/ggml"
    return f"{base_url}/{prefix}-{model}.bin"


def prepare_download(model: str, models_dir: Path | None = None) -> tuple[str, Path]:
    if not is_valid_model(model):
        raise ValueError(f"Invalid model: {model}")
    download_dir = models_dir or default_download_path()
    download_dir.mkdir(parents=True, exist_ok=True)
    url = build_model_url(model)
    savepath = download_dir / model_filename(model)
    return url, savepath


def stream_download(
    url: str,
    savepath: Path,
    timeout: float | None = None,
) -> None:
    tmp = savepath.with_suffix(savepath.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content():
                    if chunk:
                        f.write(chunk)
        tmp.replace(savepath)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def download_model(
    model: str,
    models_dir: Path | None = None,
    overwrite: bool = False,
    timeout: float | None = None,
) -> DownloadResult:
    url, savepath = prepare_download(model, models_dir)
    existed = savepath.is_file()
    if existed and not overwrite:
        return DownloadResult(model=model, url=url, dest=savepath, existed=True)
    stream_download(url, savepath, timeout=timeout)
    return DownloadResult(model=model, url=url, dest=savepath, existed=existed)


__all__ = ["download_model", "AVAILABLE_MODELS"]
