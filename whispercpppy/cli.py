from __future__ import annotations

import argparse
import os
from pathlib import Path

from whispercpppy.model import AVAILABLE_MODELS, default_download_path, download_model


def build_parser() -> argparse.ArgumentParser:
    models_text = [f" - {name}" for name in AVAILABLE_MODELS]
    model_choices = "\n".join(models_text)
    description = f"download whisper.cpp GGUF models\n\navailable models:\n{model_choices}"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("models", help="model names", nargs="+")
    parser.add_argument(
        "-d",
        "--dir",
        dest="directory",
        type=Path,
        help=(
            "save directory for models. "
            "defaults to the path set by WHISPERCPP_MODELS_DIR or the current directory."
        ),
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


def cli() -> int:
    parser = build_parser()
    args = parser.parse_args()
    download_dir = args.directory or default_download_path()
    for model_name in args.models:
        result = download_model(
            model_name,
            models_dir=download_dir,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )

        if result.existed and not args.overwrite:
            print(f"{result.filepath} already exists. Skipping download.")
    return os.EX_OK


if __name__ == "__main__":
    cli()
