from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "scripts" / "render.py"


def load_render():
    spec = importlib.util.spec_from_file_location("comfy_render", RENDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeComfy:
    """Minimal stand-in for the ComfyUI HTTP API used by render.py."""

    def __init__(self) -> None:
        self.prompt: dict | None = None
        self.uploads: list[str] = []
        self.upload_name = "uploaded.wav"
        self.transcript = "hello from the fake server"
        self.outputs: dict = {"4": {"text": [self.transcript]}}
        self.device_type = "cpu"
        self.device_name = "CPU"
        self.error_once = False
        self.history_error_served = False
        self.freed = 0
        self.prompt_count = 0
        self.empty_after = None
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # silence test output
                pass

            def _json(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/system_stats":
                    self._json({"system": {}, "devices": [{"name": app.device_name, "type": app.device_type}]})
                elif self.path.startswith("/history/"):
                    if app.error_once and not app.history_error_served:
                        app.history_error_served = True
                        self._json(
                            {
                                "p1": {
                                    "status": {
                                        "status_str": "error",
                                        "completed": False,
                                        "messages": [
                                            [
                                                "execution_error",
                                                {
                                                    "node_type": "Qwen3ASRTranscribe",
                                                    "exception_type": "torch.OutOfMemoryError",
                                                    "exception_message": "Allocation on device 0 would exceed allowed memory. (out of memory)",
                                                },
                                            ]
                                        ],
                                    },
                                    "outputs": {},
                                }
                            }
                        )
                        return
                    outputs = app.outputs
                    if app.empty_after is not None and app.prompt_count > app.empty_after:
                        outputs = {"4": {"text": [""]}}
                    self._json(
                        {
                            "p1": {
                                "status": {"status_str": "success", "completed": True},
                                "outputs": outputs,
                            }
                        }
                    )
                elif self.path.startswith("/object_info/CheckpointLoaderSimple"):
                    self._json(
                        {
                            "CheckpointLoaderSimple": {
                                "input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"]]}}
                            }
                        }
                    )
                elif self.path.startswith("/view?"):
                    body = b"fake-bytes"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/upload/image":
                    match = re.search(rb'filename="([^"]+)"', body)
                    app.uploads.append(match.group(1).decode() if match else "")
                    self._json({"name": app.upload_name, "subfolder": "", "type": "input"})
                elif self.path == "/prompt":
                    app.prompt = json.loads(body)
                    app.prompt_count += 1
                    self._json({"prompt_id": "p1"})
                elif self.path == "/free":
                    app.freed += 1
                    self._json({"ok": True})
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class RenderAsrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.audio = self.dir / "interview.wav"
        self.audio.write_bytes(b"RIFF....WAVEfmt ")
        self.render = load_render()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_asr_dry_run_patches_audio_and_options(self) -> None:
        argv = [
            "render.py",
            "asr",
            "--audio",
            str(self.audio),
            "--language",
            "English",
            "--context",
            "OpenCode, ComfyUI",
            "--timestamps",
            "--asr-model",
            "Qwen/Qwen3-ASR-1.7B",
            "--device",
            "cpu",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out:
            self.render.main()
        graph = json.loads(out.getvalue())
        self.assertEqual(graph["1"]["class_type"], "LoadAudio")
        self.assertEqual(graph["1"]["inputs"]["audio"], "interview.wav")
        self.assertEqual(graph["2"]["inputs"]["repo_id"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(graph["2"]["inputs"]["precision"], "fp32")
        transcribe = graph["3"]["inputs"]
        self.assertEqual(transcribe["language"], "English")
        self.assertEqual(transcribe["context"], "OpenCode, ComfyUI")
        self.assertTrue(transcribe["return_timestamps"])

    def test_asr_requires_audio(self) -> None:
        argv = ["render.py", "asr", "--dry-run", "--device", "cpu"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                self.render.main()
        self.assertIn("needs --audio", str(ctx.exception.code))

    def test_asr_end_to_end_uploads_submits_and_writes_transcript(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        output = self.dir / "transcript.txt"
        argv = [
            "render.py",
            "asr",
            "--audio",
            str(self.audio),
            "--language",
            "German",
            "--server",
            server.url,
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.render.main()

        self.assertEqual(server.uploads, ["interview.wav"])
        self.assertIsNotNone(server.prompt)
        graph = server.prompt["prompt"]
        self.assertEqual(graph["1"]["inputs"]["audio"], "uploaded.wav")
        self.assertEqual(graph["3"]["inputs"]["language"], "German")
        self.assertEqual(output.read_text(encoding="utf-8"), server.transcript + "\n")

    def test_server_device_reads_system_stats(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        server.device_type = "cuda"
        server.device_name = "NVIDIA GeForce RTX 4090"
        self.assertEqual(self.render.server_device(server.url), "cuda")
        server.device_type = "cuda"
        server.device_name = "AMD Radeon RX 7900 XTX"
        self.assertEqual(self.render.server_device(server.url), "rocm")
        self.assertIsNone(self.render.server_device("http://127.0.0.1:1"))

    def test_collect_text_and_text_output(self) -> None:
        entry = {"outputs": {"4": {"text": ["  transcribed line  "]}}}
        self.assertEqual(self.render.collect_text(entry), "transcribed line")
        self.assertIsNone(self.render.collect_text({"outputs": {}}))
        self.assertEqual(Path("out.txt").suffix, ".txt")

    def test_cpu_backend_selects_low_memory_defaults(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        argv = [
            "render.py",
            "t2v",
            "a paper boat",
            "--server",
            server.url,
            "--dry-run",
        ]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(stderr):
            self.render.main()
        graph = json.loads(out.getvalue())
        self.assertEqual(graph["3"]["inputs"]["clip_name"], self.render.QWEN_LOW)
        self.assertEqual(graph["6"]["inputs"]["width"], 416)
        self.assertEqual(graph["6"]["inputs"]["height"], 256)
        self.assertEqual(graph["6"]["inputs"]["length"], 73)
        self.assertIn("low-memory", stderr.getvalue())

    def test_tts_dry_run_patches_voice_loader_and_seed(self) -> None:
        argv = [
            "render.py",
            "tts",
            "A calm test sentence.",
            "--speaker",
            "Ryan",
            "--language",
            "German",
            "--seed",
            "7",
            "--tts-model",
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "--precision",
            "fp32",
            "--max-new-tokens",
            "512",
            "--device",
            "cpu",
            "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out:
            self.render.main()
        graph = json.loads(out.getvalue())
        self.assertEqual(graph["1"]["class_type"], "Qwen3Loader")
        self.assertEqual(graph["1"]["inputs"]["repo_id"], "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.assertEqual(graph["1"]["inputs"]["precision"], "fp32")
        voice = graph["2"]["inputs"]
        self.assertEqual(voice["text"], "A calm test sentence.")
        self.assertEqual(voice["speaker"], "Ryan")
        self.assertEqual(voice["language"], "German")
        self.assertEqual(voice["seed"], 7)
        self.assertEqual(voice["max_new_tokens"], 512)
        self.assertEqual(graph["3"]["class_type"], "SaveAudio")

    def test_tts_end_to_end_saves_audio(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        server.outputs = {
            "3": {"audio": [{"filename": "qwen3_tts_00001_.flac", "subfolder": "audio", "type": "output"}]}
        }
        output = self.dir / "speech.flac"
        argv = [
            "render.py",
            "tts",
            "Hello from TTS.",
            "--speaker",
            "Aiden",
            "--language",
            "English",
            "--seed",
            "11",
            "--server",
            server.url,
            "--output",
            str(output),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.render.main()

        self.assertEqual(output.read_bytes(), b"fake-bytes")
        graph = server.prompt["prompt"]
        self.assertEqual(graph["2"]["inputs"]["text"], "Hello from TTS.")
        self.assertEqual(graph["2"]["inputs"]["speaker"], "Aiden")
        self.assertEqual(graph["2"]["inputs"]["seed"], 11)

    def test_tts_then_asr_roundtrip_through_cli(self) -> None:
        """TTS output feeds ASR; the returned transcript is written to --output."""
        sentence = "The quick brown fox jumps over the lazy dog."
        server = FakeComfy()
        self.addCleanup(server.close)
        speech = self.dir / "roundtrip.flac"
        server.outputs = {
            "3": {"audio": [{"filename": "qwen3_tts_00002_.flac", "subfolder": "audio", "type": "output"}]}
        }
        tts_argv = [
            "render.py",
            "tts",
            sentence,
            "--speaker",
            "Ryan",
            "--language",
            "English",
            "--server",
            server.url,
            "--output",
            str(speech),
        ]
        with mock.patch.object(sys, "argv", tts_argv), contextlib.redirect_stdout(io.StringIO()):
            self.render.main()
        self.assertTrue(speech.is_file())

        server.outputs = {"4": {"text": [sentence]}}
        transcript = self.dir / "roundtrip.txt"
        asr_argv = [
            "render.py",
            "asr",
            "--audio",
            str(speech),
            "--language",
            "English",
            "--server",
            server.url,
            "--output",
            str(transcript),
        ]
        with mock.patch.object(sys, "argv", asr_argv), contextlib.redirect_stdout(io.StringIO()):
            self.render.main()

        self.assertEqual(server.uploads, ["roundtrip.flac"])
        self.assertEqual(transcript.read_text(encoding="utf-8").strip(), sentence)


def make_tone(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ar", "16000", "-ac", "1", str(path),
        ],
        check=True,
        capture_output=True,
    )


class RenderAsrLongTests(unittest.TestCase):
    """Long-audio ASR: chunking, progress, VRAM release and OOM recovery."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.audio = self.dir / "episode.wav"
        if shutil.which("ffmpeg"):
            make_tone(self.audio, 2.4)
        self.render = load_render()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_asr_long_requires_audio(self) -> None:
        with mock.patch.object(sys, "argv", ["render.py", "asr-long", "--device", "cpu"]):
            with self.assertRaises(SystemExit) as ctx:
                self.render.main()
        self.assertIn("needs --audio", str(ctx.exception.code))

    def test_asr_long_dry_run_keeps_the_documented_graph(self) -> None:
        with mock.patch.object(sys, "argv", ["render.py", "asr-long", "--audio", str(self.audio), "--dry-run", "--device", "cpu"]), contextlib.redirect_stdout(io.StringIO()) as out:
            self.render.main()
        graph = json.loads(out.getvalue())
        self.assertEqual(graph["1"]["class_type"], "LoadAudio")
        self.assertEqual(graph["3"]["class_type"], "Qwen3ASRTranscribe")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to chunk audio")
    def test_asr_long_chunks_joins_and_writes_review_sidecar(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        output = self.dir / "transcript.txt"
        stderr = io.StringIO()
        argv = [
            "render.py", "asr-long",
            "--audio", str(self.audio),
            "--chunk-seconds", "1",
            "--work-dir", str(self.dir),
            "--server", server.url,
            "--output", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.render.main()

        self.assertEqual(len(server.uploads), 3)
        self.assertEqual(len(set(server.uploads)), 3, "each chunk must have a unique ComfyUI cache key")
        self.assertEqual(output.read_text(encoding="utf-8").strip(), "\n\n".join([server.transcript] * 3))
        sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["chunks"], 3)
        self.assertEqual(sidecar["chunk_seconds"], 1)
        self.assertEqual(sidecar["model"], "Qwen/Qwen3-ASR-0.6B")
        self.assertGreaterEqual(server.freed, 1, "the pack should release VRAM before starting")
        self.assertIn("event=prepared chunks=3", stderr.getvalue())
        self.assertIn("event=done", stderr.getvalue())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to chunk audio")
    def test_asr_long_keeps_good_chunks_when_a_trailing_chunk_is_silent(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        server.empty_after = 2
        make_tone(self.audio, 2.4)
        output = self.dir / "transcript.txt"
        stderr = io.StringIO()
        argv = [
            "render.py", "asr-long",
            "--audio", str(self.audio),
            "--chunk-seconds", "1",
            "--work-dir", str(self.dir),
            "--server", server.url,
            "--output", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.render.main()

        self.assertEqual(output.read_text(encoding="utf-8").strip(), "\n\n".join([server.transcript] * 2))
        sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertTrue(any("returned no text" in warning for warning in sidecar["warnings"]))
        self.assertIn("returned no text", stderr.getvalue())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to read the audio")
    def test_asr_long_frees_vram_and_retries_after_an_oom(self) -> None:
        server = FakeComfy()
        self.addCleanup(server.close)
        server.error_once = True
        output = self.dir / "transcript.txt"
        stderr = io.StringIO()
        argv = [
            "render.py", "asr-long",
            "--audio", str(self.audio),
            "--chunk-seconds", "30",
            "--work-dir", str(self.dir),
            "--server", server.url,
            "--output", str(output),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.render.main()

        self.assertEqual(output.read_text(encoding="utf-8").strip(), server.transcript)
        self.assertGreaterEqual(server.freed, 2, "an OOM must trigger another /free before the retry")
        self.assertIn("event=oom", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
