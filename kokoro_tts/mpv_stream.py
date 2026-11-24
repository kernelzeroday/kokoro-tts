"""Stream synthesized speech through mpv.

Chunks are rendered in a background thread to FLAC files in a per-document
cache directory and appended to a running mpv over its JSON IPC socket, so
playback starts after the first (short) chunk while the rest keeps rendering.
mpv then provides pause, seek, and chunk skipping (< / >) for free, and the
cache doubles as a replayable playlist (playlist.m3u) and makes reruns instant.
"""

import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

FIRST_CHUNK_CHARS = 250  # small first chunk so audio starts quickly
CHUNK_CHARS = 500  # ~30s of audio; bounds the wait after skipping ahead
SAMPLE_RATE = 24000

KEYS = ("space pause · ←/→ 5s · ↑/↓ 1min · < > prev/next chunk · "
        "PgUp/PgDn 10 chunks · q quit")
LUA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpv_kokoro.lua")


def mpv_available():
    return os.name == "posix" and shutil.which("mpv") is not None


def cache_root():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "kokoro-tts")


class MpvIPC:
    """Minimal synchronous client for mpv's --input-ipc-server socket."""

    def __init__(self, path, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except OSError:
                self.sock.close()
                if time.monotonic() > deadline:
                    raise RuntimeError("mpv IPC socket did not come up")
                time.sleep(0.05)
        self.file = self.sock.makefile("rb")
        self.next_id = 0

    def command(self, *args):
        self.next_id += 1
        rid = self.next_id
        self.sock.sendall(json.dumps({"command": list(args), "request_id": rid}).encode() + b"\n")
        while True:  # skip async event lines until our reply arrives
            line = self.file.readline()
            if not line:
                raise ConnectionError("mpv closed the IPC socket")
            reply = json.loads(line)
            if reply.get("request_id") == rid:
                if reply.get("error") not in (None, "success"):
                    raise RuntimeError(f"mpv {args[0]}: {reply['error']}")
                return reply.get("data")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def plan_chunks(chapters, chunk_text):
    """Flatten chapters into (title, text) chunks; the very first is short."""
    items = []
    for chapter in chapters:
        chunks = chunk_text(chapter["content"], initial_chunk_size=CHUNK_CHARS)
        if not items and chunks:
            chunks = chunk_text(chunks[0], initial_chunk_size=FIRST_CHUNK_CHARS) + chunks[1:]
        items.extend((chapter["title"], c) for c in chunks if c.strip())
    return items


def doc_key(items, voice, speed, lang, model_path):
    h = hashlib.sha256()
    if isinstance(voice, np.ndarray):
        h.update(voice.tobytes())
    else:
        h.update(str(voice).encode())
    h.update(f"|{speed}|{lang}|{os.path.basename(model_path)}|".encode())
    for title, text in items:
        h.update(title.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()[:16]


def write_m3u(path, entries):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for title, fname in entries:
            f.write(f"#EXTINF:-1,{title}\n{fname}\n")
    os.replace(tmp, path)


def silence_file():
    path = os.path.join(cache_root(), "silence.flac")
    if not os.path.exists(path):
        os.makedirs(cache_root(), exist_ok=True)
        part = path[:-5] + f".{os.getpid()}.part.flac"
        sf.write(part, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        os.replace(part, path)
    return path


def stream_with_mpv(kokoro, chapters, chunk_text, voice, speed, lang,
                    model_path, debug=False):
    items = plan_chunks(chapters, chunk_text)
    if not items:
        print("Nothing to read.")
        return
    total = len(items)
    doc_dir = os.path.join(cache_root(), doc_key(items, voice, speed, lang, model_path))
    os.makedirs(doc_dir, exist_ok=True)
    paths = [os.path.join(doc_dir, f"{i + 1:04d}.flac") for i in range(total)]
    done = [os.path.exists(p) for p in paths]
    m3u = os.path.join(doc_dir, "playlist.m3u")
    # Every chunk goes in the playlist now, so mpv can skip anywhere; the Lua
    # script stands in silence for chunks that are not rendered yet.
    write_m3u(m3u, [(f"{title} — part {i + 1}/{total}", paths[i])
                    for i, (title, _) in enumerate(items)])

    print(f"{total} chunks, {sum(done)} cached · {KEYS}")
    print(f"Playlist: {m3u}")

    want = [0]          # playlist index the listener is on
    wake = threading.Event()
    stop = threading.Event()
    error = []

    def next_index():
        start = want[0]
        for i in range(start, total):
            if not done[i]:
                return i
        return None

    def render():
        try:
            while not stop.is_set():
                i = next_index()
                if i is None:
                    wake.wait(0.5)
                    wake.clear()
                    continue
                samples, sr = kokoro.create(items[i][1], voice=voice, speed=speed, lang=lang)
                part = paths[i][:-5] + ".part.flac"
                sf.write(part, samples, sr)
                os.replace(part, paths[i])
                done[i] = True
        except Exception as e:  # synthesis failure: report after mpv exits
            error.append(e)

    sock_dir = tempfile.mkdtemp(prefix="kokoro-mpv-")
    sock_path = os.path.join(sock_dir, "ipc")
    mpv = subprocess.Popen([
        "mpv", "--no-video", f"--input-ipc-server={sock_path}",
        f"--script={LUA}",
        f"--script-opts=kokoro-dir={doc_dir},kokoro-silence={silence_file()}",
        "--term-playing-msg=  ${media-title}",
        *shlex.split(os.environ.get("KOKORO_MPV_ARGS", "")),
        m3u,
    ])
    worker = threading.Thread(target=render, daemon=True)
    worker.start()
    try:
        ipc = MpvIPC(sock_path)
        while mpv.poll() is None:
            time.sleep(0.25)
            try:
                pos = ipc.command("get_property", "playlist-pos")
            except (ConnectionError, OSError, RuntimeError):
                continue  # mpv is exiting
            if isinstance(pos, int) and 0 <= pos != want[0]:
                want[0] = pos
                wake.set()
    finally:
        stop.set()
        if mpv.poll() is None:
            mpv.terminate()
        mpv.wait()
        shutil.rmtree(sock_dir, ignore_errors=True)

    rendered = sum(done)
    print()
    if error:
        print(f"Rendering failed: {error[0]!r}")
    if rendered < total:
        print(f"{rendered}/{total} chunks rendered; rerun to keep rendering from cache.")
    else:
        print(f"Replay: mpv {m3u}")
