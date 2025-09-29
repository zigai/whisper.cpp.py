import os
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
    filepath: Path
    existed: bool


def get_env_models_dir() -> str | None:
    return os.environ.get("WHISPERCPP_MODELS_DIR")


def default_download_path(cwd: Path | None = None) -> Path:
    if env_models_dir := get_env_models_dir():
        download_path = Path(env_models_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        return download_path
    return cwd or Path.cwd()


def is_valid_model(model: str) -> bool:
    return model in AVAILABLE_MODELS


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
    savepath = download_dir / model
    return url, savepath


def stream_download(
    url: str,
    savepath: Path,
    timeout: float | None = None,
) -> None:
    tmp = savepath.with_suffix(savepath.suffix + ".tmp")
    progress_printed = False
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0) or 0)
            chunk_size = 64 * 1024
            report_every = max(total // 100, chunk_size) if total else 1_000_000
            next_report = report_every
            downloaded = 0
            mb = 1_000_000

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        if downloaded >= next_report or downloaded >= total:
                            percent = min(int(downloaded * 100 / total), 100)
                            current_mb = downloaded / mb
                            total_mb = total / mb
                            print(
                                f"  {percent:3d}% ({current_mb:.1f}/{total_mb:.1f} MB)",
                                end="\r",
                                flush=True,
                            )
                            progress_printed = True
                            next_report = min(total, downloaded + report_every)
                    else:
                        if downloaded >= next_report:
                            current_mb = downloaded / mb
                            print(
                                f"  downloaded {current_mb:.1f} MB",
                                end="\r",
                                flush=True,
                            )
                            progress_printed = True
                            next_report = downloaded + report_every

            if total:
                total_mb = total / mb
                downloaded_mb = downloaded / mb
                print(
                    f"  100% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                    flush=True,
                )
            else:
                downloaded_mb = downloaded / mb
                print(f"  downloaded {downloaded_mb:.1f} MB", flush=True)
        tmp.replace(savepath)
    except Exception:
        if progress_printed:
            print()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def download_model(
    model: str,
    models_dir: Path | str | None = None,
    overwrite: bool = False,
    timeout: float | None = None,
) -> DownloadResult:
    if isinstance(models_dir, str):
        models_dir = Path(models_dir)
    url, savepath = prepare_download(model, models_dir)
    existed = savepath.is_file()
    if existed and not overwrite:
        return DownloadResult(model=model, url=url, filepath=savepath, existed=True)
    print(f"downloading {model} to {savepath.resolve()}")
    stream_download(url, savepath, timeout=timeout)
    return DownloadResult(model=model, url=url, filepath=savepath, existed=existed)


__all__ = ["download_model", "AVAILABLE_MODELS"]
