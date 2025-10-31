# whisper.cpp.py

Python wrapper around [whisper.cpp](https://github.com/ggml-org/whisper.cpp) server.

## Installation

This package requires [whisper.cpp](https://github.com/ggml-org/whisper.cpp) to be installed manually.

```bash
uv add "git+https://github.com/zigai/whisper.cpp.py.git"
```

```bash
pip install git+https://github.com/zigai/whisper.cpp.py
```

## Usage

```python
from whispercpppy import WhisperCppServer, WhisperCppServerOptions

opts = WhisperCppServerOptions()
whisper = WhisperCppServer(opts)

transcript = whisper.inference("/path/to/audio/or/video/file")
```

## GGUF model download helper

```text
whispercpp-download --help
usage: whisper-cpp-download [-h] [-d] [-o] [-t] models [models ...]

download whisper.cpp GGUF models

available models:
 - tiny
 - tiny.en
 - tiny-q5_1
 - tiny.en-q5_1
 - tiny-q8_0
 - base
 - base.en
 - base-q5_1
 - base.en-q5_1
 - base-q8_0
 - small
 - small.en
 - small.en-tdrz
 - small-q5_1
 - small.en-q5_1
 - small-q8_0
 - medium
 - medium.en
 - medium-q5_0
 - medium.en-q5_0
 - medium-q8_0
 - large-v1
 - large-v2
 - large-v2-q5_0
 - large-v2-q8_0
 - large-v3
 - large-v3-q5_0
 - large-v3-turbo
 - large-v3-turbo-q5_0
 - large-v3-turbo-q8_0

positional arguments:
  models             model names

options:
  -h, --help         show this help message and exit
  -d, --dir      save directory for models. defaults to the path set by WHISPERCPP_MODELS_DIR env var or the current directory if not set.
  -o, --overwrite    overwrite the file if it already exists.
  -t, --timeout  timeout for the download request in seconds
```

## License

[MIT License](https://github.com/zigai/whisper.cpp.py/blob/master/LICENSE)
