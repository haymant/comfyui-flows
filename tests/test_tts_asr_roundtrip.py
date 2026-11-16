"""Live TTS -> ASR round-trip test.

This needs a running ComfyUI server with the optional Qwen3-TTS and Qwen3-ASR
custom node packs plus their model weights. It is skipped by default:

    COMFY_ROUNDTRIP=1 python3 -m unittest tests.test_tts_asr_roundtrip -v

Environment:
    COMFYUI_SERVER   server URL (default http://127.0.0.1:8188)
    COMFY_ROUNDTRIP_TEXT     sentence to synthesize (default: a pangram)
    COMFY_ROUNDTRIP_SPEAKER  Qwen3-TTS preset voice (default: Ryan)
    COMFY_ROUNDTRIP_DIR      where to keep the artifacts (default .test/tts_asr_roundtrip)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "scripts" / "render.py"

SENTENCE = os.environ.get(
    "COMFY_ROUNDTRIP_TEXT",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
)
SERVER = os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188")
SPEAKER = os.environ.get("COMFY_ROUNDTRIP_SPEAKER", "Ryan")
ARTIFACT_DIR = Path(os.environ.get("COMFY_ROUNDTRIP_DIR", ROOT / ".test" / "tts_asr_roundtrip")).expanduser()
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = ROOT / ARTIFACT_DIR


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def similarity(expected: str, actual: str) -> float:
    return difflib.SequenceMatcher(None, normalize(expected), normalize(actual)).ratio()


@unittest.skipUnless(
    os.environ.get("COMFY_ROUNDTRIP") == "1",
    "set COMFY_ROUNDTRIP=1 with a live ComfyUI plus Qwen3-TTS/Qwen3-ASR models",
)
class TtsAsrRoundTripTests(unittest.TestCase):
    def run_render(self, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(RENDER), *args, "--server", SERVER],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def test_tts_speech_is_recognized_as_the_input_text(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        speech = ARTIFACT_DIR / "speech.flac"
        transcript = ARTIFACT_DIR / "transcript.txt"
        (ARTIFACT_DIR / "input.txt").write_text(SENTENCE + "\n", encoding="utf-8")
        for stale in (speech, transcript):
            stale.unlink(missing_ok=True)

        tts_out = self.run_render(
            "tts",
            SENTENCE,
            "--speaker",
            SPEAKER,
            "--language",
            "English",
            "--output",
            str(speech),
        )
        (ARTIFACT_DIR / "tts.log").write_text(tts_out, encoding="utf-8")
        self.assertGreater(speech.stat().st_size, 10_000, "TTS produced an empty audio file")

        duration = None
        if shutil.which("ffprobe"):
            duration = float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                        str(speech),
                    ],
                    text=True,
                ).strip()
            )
            self.assertGreater(duration, 2.0, "TTS speech is shorter than two seconds")

        asr_out = self.run_render(
            "asr",
            "--audio",
            str(speech),
            "--language",
            "English",
            "--output",
            str(transcript),
        )
        (ARTIFACT_DIR / "asr.log").write_text(asr_out, encoding="utf-8")
        recognized = transcript.read_text(encoding="utf-8").strip()
        score = similarity(SENTENCE, recognized)
        (ARTIFACT_DIR / "result.json").write_text(
            json.dumps(
                {
                    "input": SENTENCE,
                    "recognized": recognized,
                    "similarity": score,
                    "speaker": SPEAKER,
                    "duration_seconds": duration,
                    "server": SERVER,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.assertTrue(recognized, "ASR returned an empty transcript")
        self.assertGreaterEqual(
            score,
            0.8,
            f"transcript differs too much from the TTS input (similarity {score:.2f})\n"
            f"  input:      {SENTENCE}\n"
            f"  recognized: {recognized}\n"
            f"  artifacts:  {ARTIFACT_DIR}",
        )


if __name__ == "__main__":
    unittest.main()
