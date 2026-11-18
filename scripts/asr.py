#!/usr/bin/env python3
"""Local speech recognition without a ComfyUI graph.

ComfyUI is a good home for diffusion, but a Whisper-family model is a single
forward pass over the audio, not a sampling graph. Running it through a server
adds a queue, an upload, a node graph and a process per call for no benefit.
This script calls the model directly and reuses it across cues, which is what a
QA loop over hundreds of short clips actually needs.

Backends are auto-detected:

- ``mlx`` on Apple Silicon (``mlx-audio`` + MLX Whisper weights);
- ``faster-whisper`` (CTranslate2) on NVIDIA CUDA or CPU-only hosts.

Both return the same record, so callers see one format regardless of host.
Nothing here uploads audio; the model runs in this process.

Examples
--------
    python3 scripts/asr.py --audio episode.mp3 --out out/episode.txt
    python3 scripts/asr.py --audio cue.wav --format json --out out/cue.json
    python3 scripts/asr.py --audio cue.wav --format srt --out out/cue.srt
    python3 scripts/asr.py --audio zh.mp3 --language zh --format vtt --out out/zh.vtt

    # Reuse one loaded model from Python (avoids paying model load per cue):
    from asr import Transcriber
    engine = Transcriber(model="mlx-community/whisper-large-v3-turbo-asr-fp16")
    for cue in cues:
        print(engine.transcribe(cue, language="en")["text"])
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

DEFAULT_MLX_MODEL = os.environ.get(
    "ASR_MODEL", "mlx-community/whisper-large-v3-turbo-asr-fp16"
)
DEFAULT_FASTER_WHISPER_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")

# MLX community names -> faster-whisper (CTranslate2) model names. A caller can
# pass either name; the argument is translated for the backend that runs it.
FASTER_WHISPER_MODELS = {
    "mlx-community/whisper-large-v3-turbo-asr-fp16": "large-v3-turbo",
    "mlx-community/whisper-large-v3-mlx": "large-v3",
    "mlx-community/whisper-large-v3-mlx-4bit": "large-v3",
    "mlx-community/whisper-large-v3-turbo": "large-v3-turbo",
    "mlx-community/whisper-medium-mlx": "medium",
    "mlx-community/whisper-small-mlx": "small",
    "mlx-community/whisper-base-mlx": "base",
    "mlx-community/whisper-tiny-mlx": "tiny",
    "whisper-large-v3-turbo-asr-fp16": "large-v3-turbo",
}

FORMATS = ("txt", "json", "srt", "vtt")


def normalize_language(value: str | None) -> str | None:
    """Return a Whisper language code, or None to auto-detect.

    ``auto`` is deliberate: forcing ``en`` on Mandarin audio hurts accuracy, and
    the NIP store already tracks the detected language separately.
    """
    if value is None:
        return None
    value = value.strip().lower()
    if not value or value in {"auto", "detect", "unknown"}:
        return None
    if value.startswith(("zh", "cmn", "mandarin", "chinese")):
        return "zh"
    return value.split("-")[0].split("_")[0]


def _has_mlx() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        import mlx_audio.stt  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def resolve_backend(requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", "mlx", "faster-whisper"}:
        raise ValueError("backend must be auto, mlx or faster-whisper")
    if requested == "auto":
        return "mlx" if _has_mlx() else "faster-whisper"
    if requested == "mlx" and not _has_mlx():
        raise RuntimeError(
            "mlx backend needs macOS on Apple Silicon with mlx-audio installed"
        )
    return requested


def faster_whisper_name(model: str) -> str:
    return FASTER_WHISPER_MODELS.get(model, model)


def probe_duration(path: Path) -> float:
    """Best-effort audio duration via ffprobe; 0.0 when it is unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return round(float(out.decode().strip()), 3)
    except Exception:  # noqa: BLE001
        return 0.0


class Transcriber:
    """One loaded ASR model, reused across calls.

    A transcript record is ``{"text", "language", "model", "backend",
    "duration", "segments": [{"start", "end", "text"}]}``. Timestamps are
    seconds and may be ``None`` when the backend cannot align text to audio.
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str = "auto",
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.backend = resolve_backend(backend)
        if self.backend == "mlx":
            self.model_name = model or DEFAULT_MLX_MODEL
            self.device = "mps"
        else:
            self.model_name = faster_whisper_name(
                model or DEFAULT_FASTER_WHISPER_MODEL
            )
            self.device = (device or os.environ.get("ASR_DEVICE") or "auto").lower()
            self.compute_type = compute_type or os.environ.get("ASR_COMPUTE_TYPE")
        self._model = None
        self._load_seconds = 0.0

    # -- loading -----------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        started = time.time()
        if self.backend == "mlx":
            from mlx_audio.stt.utils import load_model  # noqa: PLC0415

            self._model = load_model(self.model_name)
        else:
            from faster_whisper import WhisperModel  # noqa: PLC0415

            device = self.device
            if device == "auto":
                device = "cuda" if _cuda_available() else "cpu"
            compute = self.compute_type or (
                "int8" if device == "cpu" else "float16"
            )
            self.device = device
            self.compute_type = compute
            print(
                f"[asr] loading faster-whisper {self.model_name} "
                f"on {device} ({compute})",
                file=sys.stderr,
            )
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
        self._load_seconds = round(time.time() - started, 2)
        return self._model

    # -- transcription -----------------------------------------------------
    def transcribe(self, audio: str | Path, language: str | None = None) -> dict:
        audio = Path(audio)
        if not audio.is_file():
            raise FileNotFoundError(f"audio not found: {audio}")
        language = normalize_language(language)
        model = self._ensure_model()
        duration = probe_duration(audio)
        started = time.time()
        if self.backend == "mlx":
            record = self._mlx(model, audio, language)
        else:
            record = self._faster_whisper(model, audio, language)
        elapsed = round(time.time() - started, 2)
        record.update(
            {
                "model": self.model_name,
                "backend": self.backend,
                "duration": duration or record.get("duration") or 0.0,
                "elapsed": elapsed,
                "load_seconds": self._load_seconds,
            }
        )
        if record.get("duration"):
            record["realtime_factor"] = round(
                record["duration"] / max(elapsed, 0.001), 2
            )
        record["text"] = re.sub(r"\s+", " ", record.get("text") or "").strip()
        return record

    def _mlx(self, model, audio: Path, language: str | None) -> dict:
        kwargs: dict = {"verbose": False}
        if language:
            kwargs["language"] = language
        output = model.generate(str(audio), **kwargs)
        segments = []
        for segment in getattr(output, "segments", None) or []:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": text,
                }
            )
        return {
            "text": getattr(output, "text", "") or "",
            "language": getattr(output, "language", None) or language,
            "segments": segments,
            "duration": 0.0,
        }

    def _faster_whisper(self, model, audio: Path, language: str | None) -> dict:
        segments_iter, info = model.transcribe(
            str(audio),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        segments = []
        parts = []
        for segment in segments_iter:
            text = (segment.text or "").strip()
            if not text:
                continue
            parts.append(text)
            segments.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                }
            )
        return {
            "text": " ".join(parts),
            "language": getattr(info, "language", language),
            "segments": segments,
            "duration": round(float(getattr(info, "duration", 0.0)), 3),
            "language_probability": round(
                float(getattr(info, "language_probability", 0.0)), 4
            ),
        }


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# -- formatting -------------------------------------------------------------
def _timestamp(seconds: float | None, *, comma: bool) -> str:
    if seconds is None:
        return "00:00:00" + (",000" if comma else ".000")
    millis = int(round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render(record: dict, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    text = (record.get("text") or "").strip()
    if fmt == "txt":
        return text + "\n" if text else ""
    if fmt == "json":
        return json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    segments = [s for s in record.get("segments") or [] if (s.get("text") or "").strip()]
    timed = [s for s in segments if s.get("start") is not None and s.get("end") is not None]
    if not timed:
        raise ValueError(
            f"{fmt} needs segment timestamps; use txt or json for this audio"
        )
    if fmt == "srt":
        lines = []
        for index, segment in enumerate(timed, 1):
            lines.append(str(index))
            lines.append(
                f"{_timestamp(segment['start'], comma=True)} --> "
                f"{_timestamp(segment['end'], comma=True)}"
            )
            lines.append(segment["text"].strip())
            lines.append("")
        return "\n".join(lines)
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(timed, 1):
        lines.append(str(index))
        lines.append(
            f"{_timestamp(segment['start'], comma=False)} --> "
            f"{_timestamp(segment['end'], comma=False)}"
        )
        lines.append(segment["text"].strip())
        lines.append("")
    return "\n".join(lines)


def write(record: dict, out: Path, fmt: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(record, fmt), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe local audio with MLX Whisper or faster-whisper.",
    )
    parser.add_argument("--audio", required=True, help="input audio or video file")
    parser.add_argument("--out", required=True, help="output path")
    parser.add_argument("--format", choices=FORMATS, default="txt")
    parser.add_argument(
        "--language",
        default="auto",
        help="language code (en, zh, …); 'auto' detects it",
    )
    parser.add_argument(
        "--model", default=None, help=f"default MLX: {DEFAULT_MLX_MODEL}"
    )
    parser.add_argument(
        "--backend", choices=["auto", "mlx", "faster-whisper"], default="auto"
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (faster-whisper)")
    parser.add_argument("--compute-type", default=None, help="float16 / int8 / …")
    parser.add_argument(
        "--quiet", action="store_true", help="do not print the JSON summary"
    )
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"error: audio not found: {audio}", file=sys.stderr)
        return 1
    try:
        engine = Transcriber(
            model=args.model,
            backend=args.backend,
            device=args.device,
            compute_type=args.compute_type,
        )
        record = engine.transcribe(audio, language=args.language)
        out = Path(args.out)
        write(record, out, args.format)
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        summary = {
            "out": str(out),
            "format": args.format,
            "backend": record["backend"],
            "model": record["model"],
            "language": record.get("language"),
            "duration": record.get("duration"),
            "elapsed": record.get("elapsed"),
            "realtime_factor": record.get("realtime_factor"),
            "chars": len(record.get("text") or ""),
            "segments": len(record.get("segments") or []),
        }
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
