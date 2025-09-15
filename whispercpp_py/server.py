from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class VoiceActivityDetectionOptions(BaseModel):
    enable: bool = Field(default=False, description="--vad")
    model: str | None = Field(default=None, description="--vad-model")
    threshold: float = Field(default=0.50, description="--vad-threshold")
    min_speech_duration_ms: int = Field(
        default=250, description="--vad-min-speech-duration-ms"
    )
    min_silence_duration_ms: int = Field(
        default=100, description="--vad-min-silence-duration-ms"
    )
    max_speech_duration_s: float = Field(
        default=float("inf"), description="--vad-max-speech-duration-s"
    )
    speech_pad_ms: int = Field(default=30, description="--vad-speech-pad-ms")
    samples_overlap_s: float = Field(default=0.10, description="--vad-samples-overlap")


class WhisperCppServerOptions(BaseModel):
    help: bool = Field(default=False, description="--help")
    threads: int = Field(default=4, description="--threads")
    processors: int = Field(default=1, description="--processors")
    offset_t_ms: int = Field(default=0, description="--offset-t")
    offset_n: int = Field(default=0, description="--offset-n")
    duration_ms: int = Field(default=0, description="--duration")
    max_context: int = Field(default=-1, description="--max-context")
    max_len_chars: int = Field(default=0, description="--max-len")
    split_on_word: bool = Field(default=False, description="--split-on-word")
    best_of: int = Field(default=2, description="--best-of")
    beam_size: int = Field(default=-1, description="--beam-size")
    audio_ctx: int = Field(default=0, description="--audio-ctx")
    word_thold: float = Field(default=0.01, description="--word-thold")
    entropy_thold: float = Field(default=2.40, description="--entropy-thold")
    logprob_thold: float = Field(default=-1.00, description="--logprob-thold")
    debug_mode: bool = Field(default=False, description="--debug-mode")
    translate: bool = Field(default=False, description="--translate")
    diarize: bool = Field(default=False, description="--diarize")
    tinydiarize: bool = Field(default=False, description="--tinydiarize")
    no_fallback: bool = Field(default=False, description="--no-fallback")
    print_special: bool = Field(default=False, description="--print-special")
    print_colors: bool = Field(default=False, description="--print-colors")
    print_realtime: bool = Field(default=False, description="--print-realtime")
    print_progress: bool = Field(default=False, description="--print-progress")
    no_timestamps: bool = Field(default=False, description="--no-timestamps")
    language: str = Field(default="en", description="--language")
    detect_language: bool = Field(default=False, description="--detect-language")
    prompt: str = Field(default="", description="--prompt")
    model: str = Field(default="models/ggml-base.en.bin", description="--model")
    ov_e_device: str = Field(default="CPU", description="--ov-e-device")
    dtw_model: str | None = Field(default=None, description="--dtw")
    host: str = Field(default="127.0.0.1", description="--host")
    port: int = Field(default=8080, description="--port")
    public_path: str = Field(default="examples/server/public", description="--public")
    request_path: str = Field(default="", description="--request-path")
    inference_path: str = Field(default="/inference", description="--inference-path")
    convert_audio: bool = Field(default=False, description="--convert")
    suppress_nst: bool = Field(default=False, description="--suppress-nst")
    no_speech_thold: float = Field(default=0.60, description="--no-speech-thold")
    no_context: bool = Field(default=False, description="--no-context")
    no_gpu: bool = Field(default=False, description="--no-gpu")
    flash_attn: bool = Field(default=False, description="--flash-attn")


def field_to_cli_arg(flag: str, value) -> list[str] | None:
    if value is None:
        return None
    if type(value) is bool:
        if value:
            return [flag]
        else:
            return None
    if type(value) is str and value == "":
        return None
    return [flag, str(value)]


ResponseFormat = Literal["json"]


class InferenceRequest(BaseModel):
    file: Path
    temperature: float = Field(default=0.0)
    temperature_inc: float = Field(default=0.2)
    response_format: ResponseFormat = Field(default="json")


class LoadRequest(BaseModel):
    model: Path


def generate_start_server_command(
    server_opts: WhisperCppServerOptions,
    vad_opts: VoiceActivityDetectionOptions | None = None,
    binary: str = "whisper-server",
) -> list[str]:
    command: list[str] = [binary]
    for name, info in WhisperCppServerOptions.model_fields.items():
        desc = info.description
        assert desc is not None
        arg = field_to_cli_arg(desc, getattr(server_opts, name))
        if arg is None:
            continue
        command.extend(arg)

    if vad_opts is not None and vad_opts.enable:
        for name, info in VoiceActivityDetectionOptions.model_fields.items():
            desc = info.description
            assert desc is not None
            arg = field_to_cli_arg(desc, getattr(server_opts, name))
            if arg is None:
                continue
            command.extend(arg)
    return command
