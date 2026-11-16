#!/usr/bin/env python3
"""Run the workflows in comfy/workflows against a local ComfyUI server.

No third-party dependencies: standard library only. Works with ComfyUI Desktop
on macOS (MPS) and Linux (CUDA, ROCm or CPU); the compute backend is read from
the running server.

Examples
--------
    python3 scripts/render.py t2v "a paper boat on a rain puddle" --output out/boat.mp4
    python3 scripts/render.py i2v --image photo.jpg --prompt "slow push-in" --output out/photo.mp4
    python3 scripts/render.py r2v --image face.png --prompt "Use <Picture 1> ..." --output out/face.mp4
    python3 scripts/render.py t2i "a reading corner" --output out/room.png
    python3 scripts/render.py i2i --image room.jpg --prompt "make it winter" --denoise 0.6
    python3 scripts/render.py tts "The quick brown fox jumps over the lazy dog." --output out/speech.flac
    python3 scripts/render.py asr --audio out/speech.flac --output out/speech.txt
    python3 scripts/render.py frame --video out/boat.mp4 --at 1.5 --output out/poster.png
    python3 scripts/render.py --list
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKFLOWS = ROOT / "workflows" / "api"
DEFAULT_SERVER = os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188")
DEFAULT_OUTPUT = ROOT / "output"

MODES = {
    "t2v": ("minimax_h3_t2v_32gb.api.json", "video"),
    "i2v": ("minimax_h3_i2v_32gb.api.json", "video"),
    "flf2v": ("minimax_h3_flf2v_32gb.api.json", "video"),
    "r2v": ("minimax_h3_r2v_32gb.api.json", "video"),
    "t2i": ("sdxl_text_to_image.api.json", "image"),
    "i2i": ("sdxl_image_to_image.api.json", "image"),
    "tts": ("qwen3_tts_local.api.json", "audio"),
    "asr": ("qwen3_asr_local.api.json", "text"),
}

VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
TEXT_EXT = {".txt", ".md"}
DEFAULT_SUFFIX = {"video": ".mp4", "image": ".png", "audio": ".flac", "text": ".txt"}
QWEN_LOW = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"

DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
CPU_TIMEOUT = 24 * 60 * 60
DEFAULT_TIMEOUT = 2 * 60 * 60


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def request_json(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def upload_input(server: str, path: Path) -> str:
    """Upload any input file (image or audio) and return its Load* name."""
    boundary = "----comfy" + uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    field("type", "input")
    field("overwrite", "true")
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        server + "/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.load(resp)
    name = result["name"]
    sub = result.get("subfolder") or ""
    return f"{sub}/{name}" if sub else name


def upload_image(server: str, path: Path) -> str:
    return upload_input(server, path)


def upload_audio(server: str, path: Path) -> str:
    return upload_input(server, path)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def server_device(server: str) -> str | None:
    """Read the backend from a running ComfyUI (/system_stats), if reachable."""
    try:
        stats = request_json(server + "/system_stats", timeout=3)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for device in stats.get("devices") or []:
        kind = str(device.get("type", "")).lower()
        name = str(device.get("name", "")).lower()
        if kind == "cuda" and any(word in name for word in ("amd", "radeon", "rocm", "gfx")):
            return "rocm"
        if kind in {"cuda", "rocm", "mps", "cpu"}:
            return kind
    return None


def host_device() -> str:
    """Fallback when the server is not running yet."""
    if sys.platform == "darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if os.path.exists("/dev/kfd") or shutil.which("rocm-smi"):
        return "rocm"
    return "cpu"


def resolve_device(server: str, requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "flag"
    found = server_device(server)
    if found:
        return found, "server"
    return host_device(), "host"


# ---------------------------------------------------------------------------
# Graph patching
# ---------------------------------------------------------------------------
def nodes_of(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in graph.items() if v.get("class_type") == class_type]


def by_title(graph: dict, title: str) -> dict | None:
    for node in graph.values():
        if node.get("_meta", {}).get("title") == title:
            return node
    return None


def first_node(graph: dict, *class_types: str) -> dict | None:
    for class_type in class_types:
        found = nodes_of(graph, class_type)
        if found:
            return found[0][1]
    return None


def patch_prompt(graph: dict, prompt: str) -> None:
    h3 = nodes_of(graph, "MiniMaxH3ImageToVideo") or nodes_of(graph, "MiniMaxH3ReferenceToVideo")
    if h3:
        h3[0][1]["inputs"]["prompt"] = prompt
        return
    node = by_title(graph, "Positive prompt")
    if node is None:
        nodes = nodes_of(graph, "CLIPTextEncode")
        if not nodes:
            raise SystemExit("workflow has no prompt input")
        node = nodes[0][1]
    node["inputs"]["text"] = prompt


def patch_negative(graph: dict, negative: str) -> None:
    node = by_title(graph, "Negative prompt")
    if node is not None:
        node["inputs"]["text"] = negative


def patch_seed(graph: dict, seed: int) -> None:
    if nodes_of(graph, "RandomNoise"):
        nodes_of(graph, "RandomNoise")[0][1]["inputs"]["noise_seed"] = seed
    if nodes_of(graph, "KSampler"):
        nodes_of(graph, "KSampler")[0][1]["inputs"]["seed"] = seed
    for class_type in ("Qwen3CustomVoice", "Qwen3VoiceDesign", "Qwen3VoiceClone"):
        found = nodes_of(graph, class_type)
        if found:
            found[0][1]["inputs"]["seed"] = seed


def patch_steps(graph: dict, steps: int) -> None:
    if nodes_of(graph, "BasicScheduler"):
        nodes_of(graph, "BasicScheduler")[0][1]["inputs"]["steps"] = steps
    if nodes_of(graph, "KSampler"):
        nodes_of(graph, "KSampler")[0][1]["inputs"]["steps"] = steps


def patch_denoise(graph: dict, denoise: float) -> None:
    if nodes_of(graph, "KSampler"):
        nodes_of(graph, "KSampler")[0][1]["inputs"]["denoise"] = denoise


def patch_size(graph: dict, width: int | None, height: int | None) -> None:
    if width is None and height is None:
        return
    for _, node in nodes_of(graph, "MiniMaxH3ImageToVideo") + nodes_of(graph, "MiniMaxH3ReferenceToVideo"):
        if width is not None:
            node["inputs"]["width"] = width
        if height is not None:
            node["inputs"]["height"] = height
    for _, node in nodes_of(graph, "EmptyLatentImage"):
        if width is not None:
            node["inputs"]["width"] = width
        if height is not None:
            node["inputs"]["height"] = height


def patch_length(graph: dict, seconds: float | None, frames: int | None) -> None:
    if seconds is None and frames is None:
        return
    if frames is None:
        frames = max(5, round(seconds * 24))
        while frames % 17 != 5:
            frames += 1
    for _, node in nodes_of(graph, "MiniMaxH3ImageToVideo") + nodes_of(graph, "MiniMaxH3ReferenceToVideo"):
        node["inputs"]["length"] = frames


def add_node(graph: dict, class_type: str, inputs: dict, title: str) -> str:
    ids = [int(k) for k in graph if k.isdigit()]
    new_id = str(max(ids, default=0) + 1)
    graph[new_id] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
    return new_id


def patch_images(graph: dict, images: list[Path], last_image: Path | None, dry_run: bool, server: str) -> None:
    """Wire uploaded images into the right LoadImage nodes."""
    upload = (lambda p: p.name) if dry_run else (lambda p: upload_image(server, p))

    h3_i2v = nodes_of(graph, "MiniMaxH3ImageToVideo")
    h3_r2v = nodes_of(graph, "MiniMaxH3ReferenceToVideo")

    if h3_i2v and by_title(graph, "First frame") is not None:
        if not images:
            raise SystemExit("i2v/flf2v needs --image (the start frame)")
        load = by_title(graph, "First frame")
        load["inputs"]["image"] = upload(images[0])
        last_loader = by_title(graph, "Last frame")
        if last_loader is not None:
            if last_image is None:
                raise SystemExit("flf2v needs --last-image (the end frame)")
            last_loader["inputs"]["image"] = upload(last_image)
        elif last_image is not None:
            last_id = add_node(graph, "LoadImage", {"image": upload(last_image)}, "Last frame")
            scale_id = add_node(
                graph,
                "ImageScaleToTotalPixels",
                {"image": [last_id, 0], "upscale_method": "lanczos", "megapixels": 0.2, "resolution_steps": 32},
                "Fit last frame to 0.2 MP",
            )
            h3_i2v[0][1]["inputs"]["last_frame"] = [scale_id, 0]
        return

    if h3_r2v:
        if not images:
            raise SystemExit("r2v needs at least one --image")
        node = h3_r2v[0][1]
        first = by_title(graph, "Reference image 1")
        if first is None:
            raise SystemExit("r2v workflow has no reference loader")
        first["inputs"]["image"] = upload(images[0])
        for index, path in enumerate(images[1:], start=1):
            if index >= 9:
                raise SystemExit("H3 R2V supports at most 9 reference images")
            node_id = add_node(graph, "LoadImage", {"image": upload(path)}, f"Reference image {index + 1}")
            node["inputs"][f"ref_images.ref_image_{index}"] = [node_id, 0]
        return

    # SDXL image to image
    load = by_title(graph, "Source image")
    if load is not None:
        if not images:
            raise SystemExit("i2i needs --image")
        load["inputs"]["image"] = upload(images[0])


def patch_audio(graph: dict, audio: Path | None, dry_run: bool, server: str) -> None:
    """Wire the uploaded audio file into the ASR LoadAudio node."""
    node = by_title(graph, "Input audio") or first_node(graph, "LoadAudio")
    if node is None:
        raise SystemExit("workflow has no LoadAudio input")
    if audio is None:
        raise SystemExit("asr needs --audio (a local audio file)")
    node["inputs"]["audio"] = audio.name if dry_run else upload_audio(server, audio)


def patch_asr(
    graph: dict,
    language: str | None,
    context: str | None,
    timestamps: bool,
    model: str | None,
    precision: str | None = None,
) -> None:
    transcribe = by_title(graph, "Transcribe locally") or first_node(graph, "Qwen3ASRTranscribe")
    if transcribe is None:
        raise SystemExit("workflow has no Qwen3ASRTranscribe node")
    if language is not None:
        transcribe["inputs"]["language"] = language
    if context is not None:
        transcribe["inputs"]["context"] = context
    transcribe["inputs"]["return_timestamps"] = timestamps
    if model is not None or precision is not None:
        loader = by_title(graph, "Local Qwen3-ASR model") or first_node(graph, "Qwen3ASRLoader")
        if loader is None:
            raise SystemExit("workflow has no Qwen3ASRLoader node")
        if model is not None:
            loader["inputs"]["repo_id"] = model
        if precision is not None:
            loader["inputs"]["precision"] = precision


def patch_tts(
    graph: dict,
    text: str,
    language: str | None,
    speaker: str | None,
    instruct: str | None,
    model: str | None,
    precision: str | None,
    max_new_tokens: int | None,
) -> None:
    voice = by_title(graph, "Speak text") or first_node(graph, "Qwen3CustomVoice")
    if voice is None:
        raise SystemExit("workflow has no Qwen3CustomVoice node")
    voice["inputs"]["text"] = text
    if language is not None:
        voice["inputs"]["language"] = language
    if speaker is not None:
        voice["inputs"]["speaker"] = speaker
    if instruct is not None:
        voice["inputs"]["instruct"] = instruct
    if max_new_tokens is not None:
        voice["inputs"]["max_new_tokens"] = max_new_tokens

    loader = by_title(graph, "Local Qwen3-TTS model") or first_node(graph, "Qwen3Loader")
    if model is not None or precision is not None:
        if loader is None:
            raise SystemExit("workflow has no Qwen3Loader node")
        if model is not None:
            loader["inputs"]["repo_id"] = model
        if precision is not None:
            loader["inputs"]["precision"] = precision


def check_checkpoint(server: str, graph: dict) -> None:
    """Fail early with a useful message when an SDXL checkpoint is missing."""
    loaders = nodes_of(graph, "CheckpointLoaderSimple")
    if not loaders:
        return
    info = request_json(server + "/object_info/CheckpointLoaderSimple")
    options = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    for _, node in loaders:
        name = node["inputs"]["ckpt_name"]
        if name not in options:
            raise SystemExit(
                f"ComfyUI cannot find checkpoint {name!r}.\n"
                "Install it with:\n"
                "  bash scripts/download_models.sh sdxl\n"
                "Then wait for the download to finish (about 7 GB)."
            )


# ---------------------------------------------------------------------------
# Queue and download
# ---------------------------------------------------------------------------
def submit(server: str, graph: dict) -> str:
    payload = {"prompt": graph, "client_id": uuid.uuid4().hex}
    try:
        result = request_json(server + "/prompt", payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"ComfyUI rejected the workflow ({exc.code}):\n{detail}") from exc
    if result.get("node_errors"):
        raise SystemExit("ComfyUI node errors:\n" + json.dumps(result["node_errors"], indent=2))
    return result["prompt_id"]


def wait_for(server: str, prompt_id: str, timeout: float) -> dict:
    started = time.time()
    last_note = 0.0
    while time.time() - started < timeout:
        history = request_json(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise SystemExit("ComfyUI execution failed:\n" + json.dumps(messages, indent=2)[-6000:])
            return entry
        now = time.time()
        if now - last_note >= 15:
            print(f"  ... running for {int(now - started)}s", flush=True)
            last_note = now
        time.sleep(1.5)
    raise SystemExit(f"timed out after {int(timeout)}s; the job is still in the ComfyUI queue")


def collect_files(entry: dict) -> list[dict]:
    found: list[dict] = []
    for output in entry.get("outputs", {}).values():
        for value in output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("filename") and item.get("type", "output") == "output":
                    found.append(item)
    # de-duplicate by filename
    seen = set()
    unique = []
    for item in found:
        key = (item.get("subfolder", ""), item["filename"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def collect_text(entry: dict) -> str | None:
    """Pull the transcript out of a PreviewAny (or text-returning) node."""
    outputs = entry.get("outputs", {})
    candidates: list[str] = []
    for output in outputs.values():
        for key in ("text", "string", "value"):
            value = output.get(key)
            if isinstance(value, list):
                candidates.extend(str(item) for item in value if item is not None)
            elif isinstance(value, str):
                candidates.append(value)
    for text in candidates:
        if text.strip():
            return text.strip()
    return None


def pick_file(files: list[dict], kind: str) -> dict:
    allowed = VIDEO_EXT if kind == "video" else IMAGE_EXT if kind == "image" else AUDIO_EXT if kind == "audio" else TEXT_EXT
    for item in files:
        if Path(item["filename"]).suffix.lower() in allowed:
            return item
    return files[0]


def download(server: str, item: dict, destination: Path) -> Path:
    query = urllib.parse.urlencode(
        {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
    )
    data = get_bytes(server + "/view?" + query)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def ffprobe_duration(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    )
    return float(out.strip())


def install_hint(tool: str) -> str:
    if sys.platform == "darwin":
        return f"brew install {tool}"
    return f"sudo apt install {tool}   # or: sudo dnf install {tool}"


def extract_frame(video: Path, at: str, output: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"ffmpeg is required for `frame`; install it with: {install_hint('ffmpeg')}")
    seconds = ffprobe_duration(video) * float(at[:-1]) / 100.0 if at.endswith("%") else float(at)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render MiniMax H3 video, SDXL images, Qwen3-TTS speech or Qwen3-ASR transcripts through a local ComfyUI server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", nargs="?", choices=[*MODES, "frame"], help="workflow to run")
    parser.add_argument("prompt", nargs="?", help="prompt text (or use --prompt-file)")
    parser.add_argument("--prompt", dest="prompt_opt", metavar="TEXT", help="prompt text (same as the positional argument)")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--negative", help="negative prompt for SDXL")
    parser.add_argument("--image", action="append", type=Path, default=[], help="input image; repeat for R2V references")
    parser.add_argument("--last-image", type=Path, help="optional last frame for i2v")
    parser.add_argument("--audio", type=Path, help="input audio file for the asr mode")
    parser.add_argument(
        "--language",
        help="language name for tts (Auto, English, German, ...) or asr (English, German, ...; default auto)",
    )
    parser.add_argument("--context", help="optional ASR context/hotword text")
    parser.add_argument("--timestamps", action="store_true", help="ask the ASR node for word/character timestamps")
    parser.add_argument("--asr-model", metavar="REPO", help="override the Qwen3-ASR model repo id")
    parser.add_argument("--speaker", help="Qwen3-TTS preset voice (Vivian, Ryan, Aiden, ...)")
    parser.add_argument("--instruct", help="optional Qwen3-TTS style instruction, e.g. 'calm narration'")
    parser.add_argument("--tts-model", metavar="REPO", help="override the Qwen3-TTS CustomVoice repo id")
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        help="TTS/ASR model precision (default: workflow dtype; fp32 on CPU)",
    )
    parser.add_argument("--max-new-tokens", type=int, help="Qwen3-TTS generation cap (default: 1024)")
    parser.add_argument("--video", type=Path, help="source video for `frame`")
    parser.add_argument("--at", default="0.0", help="frame timestamp in seconds or percent, e.g. 1.5 or 50%%")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--workflow", type=Path, help="override the API workflow JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration", type=float, help="video duration in seconds (snapped to 17k+5 frames)")
    parser.add_argument("--length", type=int, help="video length in frames (overrides --duration)")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--denoise", type=float)
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use the Q2 text encoder and 3s / 0.1 MP defaults (download with PROFILE=cpu); "
        "automatic on a CPU backend, disable with --no-low-memory",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="compute backend override; auto reads it from /system_stats (default: auto)",
    )
    parser.add_argument("--timeout", type=float, default=None, help="job timeout in seconds (default: 2h, 24h on CPU)")
    parser.add_argument("--dry-run", action="store_true", help="print the patched API graph and exit")
    parser.add_argument("--list", action="store_true", help="list modes and workflow files")
    return parser


def run_asr(args: argparse.Namespace, graph: dict, device: str) -> None:
    precision = args.precision or ("fp32" if device == "cpu" else None)
    patch_asr(graph, args.language, args.context, args.timestamps, args.asr_model, precision)
    try:
        patch_audio(graph, args.audio, args.dry_run, args.server)
    except SystemExit:
        raise
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach ComfyUI at {args.server}: {exc}") from exc

    if args.dry_run:
        print(json.dumps(graph, ensure_ascii=False, indent=2))
        return

    timeout = args.timeout or (CPU_TIMEOUT if device == "cpu" else DEFAULT_TIMEOUT)
    try:
        prompt_id = submit(args.server, graph)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach ComfyUI at {args.server}: {exc}\n"
            "Open Comfy Desktop (or run scripts/open_comfyui.sh) and try again."
        ) from exc

    print(f"queued {prompt_id}  (asr, device {device})")
    entry = wait_for(args.server, prompt_id, timeout)
    transcript = collect_text(entry)
    if not transcript:
        raise SystemExit("ComfyUI finished but returned no transcript:\n" + json.dumps(entry, indent=2)[-4000:])

    destination = args.output or (DEFAULT_OUTPUT / f"asr_{int(time.time())}.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(transcript + "\n", encoding="utf-8")
    print(destination)


def finish_job(args: argparse.Namespace, graph: dict, kind: str, device: str, seed: int, label: str) -> None:
    timeout = args.timeout or (CPU_TIMEOUT if device == "cpu" else DEFAULT_TIMEOUT)
    try:
        check_checkpoint(args.server, graph)
        prompt_id = submit(args.server, graph)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"cannot reach ComfyUI at {args.server}: {exc}\n"
            "Open Comfy Desktop (or run scripts/open_comfyui.sh) and try again."
        ) from exc

    print(f"queued {prompt_id}  ({label}, seed {seed}, device {device})")
    entry = wait_for(args.server, prompt_id, timeout)
    files = collect_files(entry)
    if not files:
        raise SystemExit("ComfyUI finished but returned no output files:\n" + json.dumps(entry, indent=2)[-4000:])

    chosen = pick_file(files, kind)
    suffix = Path(chosen["filename"]).suffix or DEFAULT_SUFFIX.get(kind, ".bin")
    destination = args.output or (DEFAULT_OUTPUT / f"{label}_{int(time.time())}_{seed}{suffix}")
    saved = download(args.server, chosen, destination)
    print(saved)
    for item in files:
        if item is not chosen:
            extra = destination.with_name(f"{destination.stem}_{Path(item['filename']).name}")
            download(args.server, item, extra)
            print(extra)


def main() -> None:
    args = build_parser().parse_args()

    if args.list:
        print("mode  workflow")
        for mode, (name, kind) in MODES.items():
            print(f"{mode:4}  {WORKFLOWS / name}  ({kind})")
        print("frame source video -> still image (ffmpeg)")
        return

    if not args.mode:
        raise SystemExit("choose a mode: " + ", ".join(MODES) + ", frame")

    if args.mode == "frame":
        if not args.video or not args.output:
            raise SystemExit("frame needs --video and --output")
        print(extract_frame(args.video, args.at, args.output))
        return

    workflow_name, kind = MODES[args.mode]
    workflow_path = args.workflow or (WORKFLOWS / workflow_name)
    graph = json.loads(workflow_path.read_text())

    device, device_source = resolve_device(args.server, args.device)

    if args.mode == "asr":
        run_asr(args, graph, device)
        return

    prompt = args.prompt or args.prompt_opt
    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
    if not prompt:
        raise SystemExit("a prompt is required (positional or --prompt-file)")

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)

    if args.mode == "tts":
        patch_tts(
            graph,
            prompt,
            args.language,
            args.speaker,
            args.instruct,
            args.tts_model,
            args.precision or ("fp32" if device == "cpu" else None),
            args.max_new_tokens,
        )
        patch_seed(graph, seed)
        if args.dry_run:
            print(json.dumps(graph, ensure_ascii=False, indent=2))
            return
        finish_job(args, graph, "audio", device, seed, "tts")
        return

    low_memory = args.low_memory
    if low_memory is None:
        low_memory = device == "cpu" and device_source in ("server", "flag")
        if low_memory:
            print(
                "note: CPU backend detected; using the low-memory profile "
                "(Q2 text encoder, 3s, 0.1 MP).\n"
                "      Download it with: PROFILE=cpu bash scripts/download_models.sh core\n"
                "      Pass --no-low-memory to keep the workflow defaults.",
                file=sys.stderr,
            )

    patch_prompt(graph, prompt)
    if args.negative:
        patch_negative(graph, args.negative)
    patch_seed(graph, seed)
    if args.steps:
        patch_steps(graph, args.steps)
    if args.denoise is not None:
        patch_denoise(graph, args.denoise)

    width, height = args.width, args.height
    duration, length = args.duration, args.length
    if low_memory:
        for _, node in nodes_of(graph, "CLIPLoaderGGUF"):
            node["inputs"]["clip_name"] = QWEN_LOW
        if width is None:
            width = 416
        if height is None:
            height = 256
        if kind == "video" and duration is None and length is None:
            duration = 3.0

    patch_size(graph, width, height)
    if kind == "video":
        patch_length(graph, duration, length)

    try:
        patch_images(graph, args.image, args.last_image, args.dry_run, args.server)
    except SystemExit:
        raise
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach ComfyUI at {args.server}: {exc}") from exc

    if args.dry_run:
        print(json.dumps(graph, ensure_ascii=False, indent=2))
        return

    finish_job(args, graph, kind, device, seed, args.mode)


if __name__ == "__main__":
    main()
