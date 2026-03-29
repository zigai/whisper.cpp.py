from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from whispercpppy.model import (
    AVAILABLE_MODELS,
    DownloadResult,
    default_download_path,
    download_model,
)
from whispercpppy.vad_model import AVAILABLE_VAD_MODELS, download_vad_model


def create_parser(
    available_models: Sequence[str],
    *,
    description_header: str,
    dir_help: str,
    models_arg_help: str,
) -> argparse.ArgumentParser:
    models_text = [f" - {name}" for name in available_models]
    model_choices = "\n".join(models_text)
    description = f"{description_header}\n\navailable models:\n{model_choices}"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("models", help=models_arg_help, nargs="+")
    parser.add_argument(
        "-d",
        "--dir",
        dest="directory",
        type=Path,
        help=dir_help,
        metavar="\b",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite the file if it already exists.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=None,
        help="timeout for the download request in seconds.",
        metavar="\b",
    )

    return parser


def run_cli(
    parser: argparse.ArgumentParser,
    downloader: Callable[[str, Path | str | None, bool, float | None], DownloadResult],
) -> int:
    args = parser.parse_args()
    download_dir = args.directory or default_download_path()
    for model_name in args.models:
        result = downloader(
            model_name,
            download_dir,
            args.overwrite,
            args.timeout,
        )

        if result.existed and not args.overwrite:
            print(f"{result.filepath} already exists. Skipping download.")

    return os.EX_OK


def build_model_download_parser() -> argparse.ArgumentParser:
    dir_help = "save directory for models. defaults to the path set by WHISPERCPP_MODELS_DIR env var or the current directory if not set"
    return create_parser(
        AVAILABLE_MODELS,
        description_header="download whisper.cpp GGUF models",
        dir_help=dir_help,
        models_arg_help="model names",
    )


def build_vad_download_parser() -> argparse.ArgumentParser:
    dir_help = "save directory for VAD models. defaults to the path set by WHISPERCPP_MODELS_DIR env var or the current directory if not set"
    return create_parser(
        AVAILABLE_VAD_MODELS,
        description_header="download whisper.cpp VAD GGUF models",
        dir_help=dir_help,
        models_arg_help="VAD model names",
    )


def model_cli() -> int:
    return run_cli(build_model_download_parser(), download_model)


def vad_model_cli() -> int:
    return run_cli(build_vad_download_parser(), download_vad_model)
