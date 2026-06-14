from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from whispercpppy.response_types import InferenceJSONVerbose


class VoiceActivityDetectionOptions(BaseModel):
    enable: bool = Field(default=False, description="--vad")
    model: str | None = Field(default=None, description="--vad-model")
    threshold: float = Field(default=0.50, description="--vad-threshold")
    min_speech_duration_ms: int = Field(default=250, description="--vad-min-speech-duration-ms")
    min_silence_duration_ms: int = Field(default=100, description="--vad-min-silence-duration-ms")
    max_speech_duration_s: float = Field(
        default=float("inf"),
        description="--vad-max-speech-duration-s",
    )
    speech_pad_ms: int = Field(default=30, description="--vad-speech-pad-ms")
    samples_overlap_s: float = Field(default=0.10, description="--vad-samples-overlap")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def hash(self) -> int:
        return hash(str(self.to_dict()))


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
    tmp_dir: str = Field(default=".", description="--tmp-dir")
    suppress_nst: bool = Field(default=False, description="--suppress-nst")
    no_speech_thold: float = Field(default=0.60, description="--no-speech-thold")
    no_gpu: bool = Field(default=False, description="--no-gpu")
    device: int = Field(default=0, description="--device")
    flash_attn: bool = Field(default=True, description="--flash-attn")
    no_flash_attn: bool = Field(default=False, description="--no-flash-attn")
    no_language_probabilities: bool = Field(
        default=False,
        description="--no-language-probabilities",
    )


CLIArgValue = bool | int | float | str | None


def field_to_cli_arg(flag: str, value: CLIArgValue) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, bool):
        if value:
            return [flag]

        return None

    if isinstance(value, str) and value == "":
        return None

    return [flag, str(value)]


ResponseFormat = Literal["json", "verbose_json", "srt", "vtt", "text", "tsv"]

VIDEO_EXT = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
}


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXT


def resolve_executable(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path():
        if not candidate.exists():
            raise RuntimeError(f"Executable not found: {command}")
        if not os.access(candidate, os.X_OK):
            raise RuntimeError(f"Executable is not executable: {command}")

        return str(candidate.resolve())

    resolved_command = shutil.which(command)
    if resolved_command is None:
        raise RuntimeError(f"Executable not found in PATH: {command}")

    return resolved_command


def video_to_mono16k_wav(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = Path(temp_file.name)

    try:
        ffmpeg_command = resolve_executable("ffmpeg")
    except RuntimeError as exc:
        audio_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required to convert video files to audio but was not found"
        ) from exc

    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]

    try:
        subprocess.run(  # noqa: S603 - resolved executable with arg list
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        audio_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required to convert video files to audio but was not found"
        ) from exc
    except subprocess.CalledProcessError as exc:
        audio_path.unlink(missing_ok=True)
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(
            "ffmpeg failed to convert video file to audio"
            + (f": {stderr.strip()}" if stderr else "")
        ) from exc

    return audio_path


def generate_start_server_command(
    server_opts: WhisperCppServerOptions,
    vad_opts: VoiceActivityDetectionOptions | None = None,
    binary: str = "whisper-server",
) -> list[str]:
    command: list[str] = [binary]

    for name, info in WhisperCppServerOptions.model_fields.items():
        if name == "flash_attn" and server_opts.no_flash_attn:
            continue

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
            arg = field_to_cli_arg(desc, getattr(vad_opts, name))
            if arg is None:
                continue

            command.extend(arg)

    return command


class WhisperCppServer:
    def __init__(
        self,
        server_options: WhisperCppServerOptions,
        vad_options: VoiceActivityDetectionOptions | None = None,
        binary: str = "whisper-server",
        autostart: bool = False,
        ready_timeout_s: float | None = 30.0,
        ready_check_interval_s: float = 0.25,
        ready_probe_timeout_s: float = 1.0,
        request_timeout_s: float | None = None,
    ) -> None:
        if ready_check_interval_s <= 0:
            raise ValueError("'ready_check_interval_s' must be positive")
        if ready_probe_timeout_s <= 0:
            raise ValueError("'ready_probe_timeout_s' must be positive")
        if ready_timeout_s is not None and ready_timeout_s < 0:
            raise ValueError("'ready_timeout_s' must be non-negative or None")
        if request_timeout_s is not None and request_timeout_s <= 0:
            raise ValueError("'request_timeout_s' must be positive or None")

        self._ready_timeout = ready_timeout_s
        self._ready_interval = ready_check_interval_s
        self._ready_probe_timeout = ready_probe_timeout_s
        self._request_timeout = request_timeout_s
        self._server_options = server_options
        self._vad_options = vad_options
        self._binary = binary
        self._process: subprocess.Popen[bytes] | None = None
        self._base_url = f"http://{server_options.host}:{server_options.port}"

        if autostart:
            self.start()

    def start(self) -> None:
        if self._process is not None and self.is_running():
            return

        command = generate_start_server_command(
            self._server_options,
            self._vad_options,
            resolve_executable(self._binary),
        )
        self._process = subprocess.Popen(  # noqa: S603 - resolved executable with arg list
            command
        )

    def __del__(self) -> None:
        try:
            self.stop()
        except (AttributeError, OSError, subprocess.SubprocessError):
            return

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

        self._process = None

    def is_running(self) -> bool:
        if self._process is None:
            return False

        return self._process.poll() is None

    def is_ready(self) -> bool:
        if not self.is_running():
            return False

        url = self._get_url("")
        try:
            response = requests.head(url, timeout=self._ready_probe_timeout)
        except requests.RequestException:
            return False

        if 200 <= response.status_code < 400:
            return True

        if response.status_code in (405, 501):
            try:
                response = requests.get(url, timeout=self._ready_probe_timeout)
            except requests.RequestException:
                return False

            return 200 <= response.status_code < 400

        return False

    def inference(
        self,
        file: Path | str,
        temperature: float = 0.0,
        temperature_inc: float = 0.2,
    ) -> InferenceJSONVerbose:
        self._wait_until_ready()

        url = self._get_url(self._server_options.inference_path)

        if isinstance(file, str):
            file = Path(file)

        upload_path = file
        temp_audio_path: Path | None = None
        if is_video_file(file):
            temp_audio_path = video_to_mono16k_wav(file)
            upload_path = temp_audio_path

        try:
            with upload_path.open("rb") as file_handle:
                response = requests.post(
                    url,
                    files={"file": (upload_path.name, file_handle)},
                    data={
                        "temperature": str(temperature),
                        "temperature_inc": str(temperature_inc),
                        "response_format": "verbose_json",
                    },
                    timeout=self._request_timeout,
                )
            response.raise_for_status()
            response_json = response.json()

            return InferenceJSONVerbose(**response_json)
        finally:
            if temp_audio_path is not None:
                temp_audio_path.unlink(missing_ok=True)

    def load(self, model: Path | str) -> requests.Response:
        self._wait_until_ready()
        url = self._get_url("load")
        response = requests.post(
            url,
            files={"model": (None, str(model))},
            timeout=self._request_timeout,
        )
        response.raise_for_status()

        return response

    def _get_url(self, path: str) -> str:
        segments: list[str] = []

        base_path = self._server_options.request_path
        if base_path:
            segments.append(base_path.strip("/"))

        normalized = path.strip("/")
        if normalized:
            segments.append(normalized)

        if not segments:
            return self._base_url

        return f"{self._base_url}/{'/'.join(segments)}"

    def _resolve_poll_interval(self, poll_interval_s: float | None) -> float:
        interval = self._ready_interval if poll_interval_s is None else poll_interval_s
        if interval <= 0:
            raise ValueError("'poll_interval_s' must be positive")

        return interval

    def _resolve_deadline(self, timeout_s: float | None) -> float | None:
        timeout = self._ready_timeout if timeout_s is None else timeout_s
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout_s' must be non-negative or None")
        if timeout is None:
            return None

        return time.monotonic() + timeout

    def _wait_until_ready(
        self,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        poll_interval = self._resolve_poll_interval(poll_interval_s)
        deadline = self._resolve_deadline(timeout_s)

        if not self.is_running():
            self.start()

        while not self.is_ready():
            if not self.is_running():
                raise RuntimeError("WhisperCPP server process exited before it became ready")

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for WhisperCPP server to become ready")

            time.sleep(poll_interval)


__all__ = ["VoiceActivityDetectionOptions", "WhisperCppServer", "WhisperCppServerOptions"]
