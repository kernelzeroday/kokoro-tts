"""Kokoro on the GPU through llama.cpp's llama-server (/tts endpoint).

LlamaKokoro mimics the parts of kokoro_onnx.Kokoro that kokoro-tts uses
(create, create_stream, get_voices), so the rest of the CLI is unchanged.
It starts a private llama-server on a free localhost port and stops it on
close() / exit. llama.cpp runs the model on whatever GPU backend it was built
with (Vulkan, Metal, CUDA...).

English goes in as text (llama.cpp's own text-to-phoneme step). Other
languages are phonemized with espeak-ng via kokoro_onnx's tokenizer and sent
with raw_prompt, which is what the llama.cpp docs prescribe.

Config (environment):
  KOKORO_LLAMA_SERVER  llama-server binary (default: llama-server on PATH)
  KOKORO_LLAMA_MODEL   model args (default: -hf ggml-org/Kokoro-82M-GGUF)
  KOKORO_LLAMA_ARGS    extra llama-server args (e.g. "-dev Vulkan0")
"""

import atexit
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

DEFAULT_MODEL = "-hf ggml-org/Kokoro-82M-GGUF"
ENGLISH = {"en-us", "en-gb"}


def server_binary():
    path = os.environ.get("KOKORO_LLAMA_SERVER") or shutil.which("llama-server")
    if not path or not os.access(path, os.X_OK):
        raise RuntimeError(
            "llama backend needs llama-server (a llama.cpp build with Kokoro "
            "support); set KOKORO_LLAMA_SERVER or put it on PATH")
    return path


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaKokoro:

    def __init__(self, startup_timeout=180.0):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = tempfile.NamedTemporaryFile(
            prefix="kokoro-llama-", suffix=".log", delete=False)
        cmd = [server_binary(), "--host", "127.0.0.1", "--port", str(self.port),
               "-ngl", "99", *shlex.split(os.environ.get("KOKORO_LLAMA_MODEL", DEFAULT_MODEL)),
               *shlex.split(os.environ.get("KOKORO_LLAMA_ARGS", ""))]
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.close)
        deadline = time.monotonic() + startup_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited at startup; see {self.log.name}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"llama-server did not become ready; see {self.log.name}")
            time.sleep(0.2)
        with urllib.request.urlopen(self.base + "/props", timeout=10) as r:
            props = json.load(r)
        self._voices = list(props.get("voices") or [])
        self._phonemizer = None

    def close(self):
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log = getattr(self, "log", None)
        if log and not log.closed:
            log.close()
            if proc and proc.returncode in (0, -15):
                os.unlink(log.name)  # keep the log only if something went wrong

    # --- kokoro_onnx.Kokoro-compatible surface -------------------------------

    def get_voices(self):
        return list(self._voices)

    def get_voice_style(self, name):
        raise ValueError("voice blending is not supported by the llama (GPU) backend")

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us", **_):
        if not isinstance(voice, str):
            raise ValueError("voice blending is not supported by the llama (GPU) backend")
        body = {"input": text, "voice": voice, "speed": float(speed),
                "response_format": "pcm"}
        if lang in ENGLISH:
            body["lang"] = lang
        else:
            if self._phonemizer is None:
                from kokoro_onnx.tokenizer import Tokenizer
                self._phonemizer = Tokenizer()
            body["input"] = self._phonemizer.phonemize(text, lang)
            body["raw_prompt"] = True
        req = urllib.request.Request(
            self.base + "/tts", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"llama-server /tts {e.code}: {e.read().decode(errors='replace')[:300]}")
        params = dict(p.strip().split("=", 1) for p in ctype.split(";")[1:] if "=" in p)
        sr = int(params.get("rate", 24000))
        return np.frombuffer(data, dtype="<f4").copy(), sr

    async def create_stream(self, text, voice="af_heart", speed=1.0, lang="en-us", **kw):
        import asyncio
        yield await asyncio.to_thread(self.create, text, voice, speed, lang)
