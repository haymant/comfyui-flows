# comfy — standalone ComfyUI media workflows

A self-contained ComfyUI workflow pack for **ComfyUI Desktop on macOS (Apple
Silicon / MPS) and Linux (NVIDIA CUDA, AMD ROCm, or x86-64 CPU-only)**. It
generates video with native stereo audio from text or sample images, still
images from text or sample images, and optional local speech with Qwen3-TTS and
Qwen3-ASR. The speech workflows are a separate custom-node/model dependency and
are not required by the image/video workflows.

Nothing here imports the rest of this repository. The only contract is:

- ComfyUI Desktop is installed.
- The model files live in `ComfyUI/models/` (the scripts find it for you).
- The CLI talks to the Desktop server at `http://127.0.0.1:8188` (or the port
  recorded in `comfy-procs/`).

The default profile targets a **32 GB machine**: an M2 Pro/Max Mac, an NVIDIA
GPU with 24 GB+ VRAM, or a Linux host with enough system RAM for CPU inference.
It uses MiniMax H3 GGUF weights sized to fit in 32 GB, with a Turbo LoRA so a
5-second clip is a coffee break rather than an afternoon.

| Platform | Backend | Status |
|---|---|---|
| macOS 13+ on Apple Silicon | MPS | primary; timings below were measured here |
| Linux x86-64 + NVIDIA | CUDA | supported; default workflows, GGUF encoders |
| Linux x86-64 + AMD | ROCm | supported where ComfyUI installs a ROCm PyTorch build |
| Linux x86-64, no GPU | CPU | loads and runs, but H3 video is hours per clip |

ComfyUI Desktop on Linux ships as a `.deb` or AppImage. The scripts look in the
XDG app-data directories, `~/ComfyUI-Installs/*/ComfyUI`, and `~/ComfyUI`,
and also honour `COMFY_ROOT` for manual installs.

## What you get

| Mode | Workflow | Input | Output | Model |
|---|---|---|---|---|
| `t2v` | `workflows/minimax_h3_t2v_32gb.json` | text | 5 s video + stereo audio | H3 FL2VA |
| `i2v` | `workflows/minimax_h3_i2v_32gb.json` | one image (+ optional last frame) | 5 s video + audio | H3 FL2VA |
| `flf2v` | `workflows/minimax_h3_flf2v_32gb.json` | start image + end image | 5 s video + audio | H3 FL2VA |
| `r2v` | `workflows/minimax_h3_r2v_32gb.json` | 1–9 reference images | 5 s video + audio | H3 Ref2VA |
| `t2i` | `workflows/sdxl_text_to_image.json` | text | 1024×1024 still | SDXL |
| `i2i` | `workflows/sdxl_image_to_image.json` | one image + text | restyled still | SDXL |
| `frame` | — | a generated video | PNG still | ffmpeg |
| `tts` | `workflows/qwen3_tts_local.json` | text | several seconds of speech | Qwen3-TTS 0.6B |
| `asr` | `workflows/qwen3_asr_local.json` | short local audio file | transcript text, detected language; optional timestamps | Qwen3-ASR 0.6B |
| `asr-long` | `workflows/qwen3_asr_local.json` | long local audio file | chunked transcript + review sidecar | Qwen3-ASR 0.6B |

`workflows/api/` contains API-format graphs. `scripts/render.py` submits the
image, video, TTS and ASR graphs; `frame` uses ffmpeg. The files in
`workflows/` are UI graphs — drop them into Comfy Desktop or open them from the
workflow browser. The `tts` and `asr` modes are the building blocks for a
speech round-trip test: synthesize a sentence, transcribe the result, and
compare the two.

MiniMax H3 is a video model. Still images use SDXL; you can also pull a PNG from
any H3 clip with the `frame` mode.

Two speech helpers run without ComfyUI at all: `scripts/asr.py` (MLX Whisper on
Apple Silicon, faster-whisper on CUDA/CPU) and `scripts/qa_audio.py` (per-cue
quality gate over synthesized speech). They are the recommended path for
long-form transcription and cue checks; the Qwen3 speech graphs remain optional.
See [Standalone ASR](#standalone-asr-scriptsasrpy--preferred-for-episodes-and-cue-checks).

## 1. Install the one custom node

MiniMax H3's GGUF loaders are not in core ComfyUI. Everything else is core.

```bash
bash scripts/setup_desktop.sh --check     # shows paths, device and server state
bash scripts/setup_desktop.sh --nodes     # installs ComfyUI-GGUF
```

Quit and reopen Comfy Desktop afterwards. If you prefer the GUI, install
**ComfyUI-GGUF** from ComfyUI Manager instead.

On Linux, install ComfyUI Desktop from the official `.deb` or AppImage first,
create a local instance, then run the commands above. `--check` prints the
detected backend and the recommended startup arguments; `--args` prints just
the arguments if you want to paste them into Desktop's settings:

```bash
bash scripts/setup_desktop.sh --args
```

On a CPU-only host the arguments include `--cpu`.

## 2. Download models

```bash
bash scripts/download_models.sh core      # ~31 GB, T2V + I2V
bash scripts/download_models.sh r2v       # +11 GB, reference-to-video
bash scripts/download_models.sh sdxl      # +7 GB, still images
```

`PROFILE=24gb` swaps the 14.6 GB Qwen encoder for the 8.5 GB Q2 version.
`PROFILE=cpu` is an alias for `24gb`; use it on CPU-only or low-RAM hosts.
`PROFILE=quality` uses Q5 H3 weights; only sensible on 64 GB or more.

The exact files, URLs and sizes are in [MODELS.md](MODELS.md).

## 3. Install the workflows in Desktop

```bash
bash scripts/setup_desktop.sh --workflows
```

They appear under **Workflows → linkzag-minimax-h3**. Or open a JSON directly:
`Workflows → Open` and pick a file from `workflows/`. The installer also seeds
`ComfyUI/input/example.png` so the image loaders have a valid placeholder.

Install the optional speech workflows separately:

```bash
bash scripts/setup_desktop.sh --speech-workflows   # TTS + ASR
# or individually:
bash scripts/setup_desktop.sh --tts-workflow
bash scripts/setup_desktop.sh --asr-workflow
```

## Optional: local speech (TTS + ASR)

The speech graphs use two community node packs by the same author:

- [ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) — text to speech
- [ComfyUI-Qwen3-ASR](https://github.com/DarioFT/ComfyUI-Qwen3-ASR) — speech to text

Install them through ComfyUI Manager (search for `Qwen3-TTS` and `Qwen3-ASR`)
or clone them into `ComfyUI/custom_nodes/` and install their `requirements.txt`
with the Desktop Python. Their Python requirements include PyTorch,
Transformers, `qwen-tts` / `qwen-asr`, `soundfile`, and other packages. Check
compatibility with the installed ComfyUI Desktop Python/Torch versions before
installing; this pack deliberately does not run an unpinned dependency install
or change the existing ComfyUI environment automatically. Restart ComfyUI after
the nodes are installed.

The two packs currently pin different Transformers releases (`qwen-tts`
4.57.3, `qwen-asr` 4.57.6), so a normal combined `pip install` cannot resolve.
Both run with 4.57.6; if pip refuses, install the two `qwen-*` wheels with
`--no-deps` after installing the shared requirements. The safest option is a
separate ComfyUI venv/base directory for speech experiments, as used by the
live round-trip test, rather than altering the Desktop environment.

Both node packs are optional and independent of the image/video workflows. The
TTS graph defaults to `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` with the `Ryan`
preset voice, English, `bf16` and a 1024-token generation cap. The ASR graph
defaults to `Qwen/Qwen3-ASR-0.6B` with automatic language detection and no
word/character timestamps. Models auto-download into
`ComfyUI/models/Qwen3-TTS/` and `ComfyUI/models/Qwen3-ASR/` on first use; you can
pre-provision those directories for offline operation.

```bash
# Several seconds of speech; writes a FLAC file.
python3 scripts/render.py tts "The quick brown fox jumps over the lazy dog." \
  --speaker Ryan --language English --output out/speech.flac

# Transcribe it again; writes plain text.
python3 scripts/render.py asr --audio out/speech.flac --language English \
  --output out/speech.txt

# A whole podcast episode: split, transcribe, join, keep a review sidecar.
python3 scripts/render.py asr-long --audio episode.mp3 --language English \
  --chunk-seconds 45 --output out/episode.txt
```

### Long audio (`asr-long`)

`asr` sends one file as one prompt. That is fine for a sentence or a short cue,
but not for an episode: the `Qwen3ASRLoader` node hard-codes
`max_new_tokens=256`, so anything past roughly one minute of speech is cut
mid-sentence without an error. A 30–60 minute file also exhausts an 8 GB GPU
because the acoustic encoder's self-attention grows with the input length.

`asr-long` is the episode-scale mode:

- splits the input into mono 16 kHz WAV chunks (`--chunk-seconds`, default 45)
  using ffmpeg, with a unique upload name per run so ComfyUI's execution cache
  can never replay a previous transcript;
- asks ComfyUI to release cached VRAM (`POST /free`, disable with `--no-free`)
  before the first chunk;
- prints machine-readable progress on stderr:
  `[asr-long] event=chunk_start 3/64`, `event=chunk_done 3/64 chars=812`,
  `event=warning …`, `event=oom …`;
- on `torch.OutOfMemoryError`, frees VRAM, retries the chunk, and if it still
  fails halves the chunk and retries once (then fails with an actionable
  message);
- warns when a chunk still looks truncated so you can lower `--chunk-seconds`;
- treats an empty chunk (music, silence, no speech) as a warning and keeps the
  other chunks instead of failing the whole episode;
- writes the joined transcript to `--output` and a sidecar JSON next to it
  (`chunks`, `chunk_seconds`, `duration_seconds`, `warnings`, `device`, `model`).

```bash
python3 scripts/render.py asr-long --audio episode.mp3 \
  --chunk-seconds 45 --work-dir /tmp/asr-episode --output out/episode.txt
```

Chunk files are deleted after the run; pass `--keep-chunks` (or reuse
`--work-dir`) when you want to inspect them. The sidecar is intentionally kept:
a downstream wrapper can show which model/device produced the text and which
chunks were flagged.

`--timestamps` enables the ASR node's timestamp output; it needs the optional
Qwen forced aligner in the model loader, whose weights are additional. On a
CPU-only host `render.py` selects `fp32` automatically; override it with
`--precision bf16` if you prefer. In the UI, open `qwen3_tts_local.json` or
`qwen3_asr_local.json`, edit the text or pick input audio, then press **Run**.

The round-trip test synthesizes speech with the `tts` mode, feeds the file to
`asr`, and checks that the recognized text matches the input. It needs a live
server and both model packs:

```bash
COMFY_ROUNDTRIP=1 python3 -m unittest tests.test_tts_asr_roundtrip -v
```

It deliberately requires the models: the default test run covers the mocked CLI
plumbing in `tests/test_render_cli.py`, not a real generation.

These workflows send audio and text to a local ComfyUI process; the graphs have
no remote-upload step. Model weights may still be downloaded from the selected
Hugging Face or ModelScope source. Keep the ComfyUI server bound to loopback and
do not expose its unauthenticated local API to an untrusted network.

### Standalone ASR (`scripts/asr.py`) — preferred for episodes and cue checks

`asr`/`asr-long` are a ComfyUI graph. That is useful for a demo, but a
Whisper-family model is one forward pass over the audio, not a sampling graph:
through ComfyUI every call pays a queue, an upload, a node graph and a fresh
process. Measured on an M2 Pro, `whisper-large-v3-turbo` transcribes at roughly
**5–8× realtime**, while the graph round trip dominates a short cue.

`scripts/asr.py` calls the model directly and reuses it across calls. It needs
neither ComfyUI nor a running server, and it writes the same text for every
host:

| Host | Backend | Packages |
|---|---|---|
| macOS on Apple Silicon | `mlx` (MLX Whisper) | `mlx-audio`, `mlx` |
| Linux + NVIDIA | `faster-whisper` (CTranslate2) | `faster-whisper` |
| CPU-only | `faster-whisper` (`int8`) | `faster-whisper` |

```bash
# One whole episode. Auto-detects the language and prints a JSON summary.
python3 scripts/asr.py --audio podcast.mp3 --out out/podcast.txt

# Timestamped output for captions or a transcript sidecar.
python3 scripts/asr.py --audio cue.wav --out out/cue.srt --format srt --language en
python3 scripts/asr.py --audio episode.mp3 --out out/episode.json --format json

# Force a backend or a model.
python3 scripts/asr.py --audio cue.wav --out out/cue.txt --backend faster-whisper --device cpu
```

- `--format` is `txt`, `json`, `srt` or `vtt`. Subtitle formats need timestamps;
  `txt`/`json` still work when a backend cannot align.
- `json` records `text`, `language`, `segments[]`, `model`, `backend`,
  `duration`, `elapsed` and `realtime_factor`, so a caller can show provenance.
- `ASR_MODEL` selects the model (default
  `mlx-community/whisper-large-v3-turbo-asr-fp16`, translated to
  `large-v3-turbo` for faster-whisper). Set `HF_HUB_OFFLINE=1` once the weights
  are cached to skip the Hub check.
- Nothing leaves the machine. The model runs in the same process as the script.

Then load it from Python to keep one model across a cue loop:

```python
from asr import Transcriber

engine = Transcriber()                 # resolves mlx or faster-whisper
for cue in cues:
    print(engine.transcribe(cue, language="en")["text"])
```

### Cue QA (`scripts/qa_audio.py`)

`qa_audio.py` is the gate for synthesized speech: it transcribes each cue with
`asr.py` (one loaded model for the whole run) and flags truncation, dropped or
extra words, stutters, extreme pace and bad levels. WER is used for English,
CER for Mandarin, and spelled-out numbers are normalised before scoring so
"seventy-five percent" matches "75%".

```bash
python3 scripts/qa_audio.py --manifest out/cues.json --out out/qa.json -v
```

```json
{
  "language": "en",
  "cues": [
    {"id": "shot-1", "text": "The expected narration.", "audio": "out/shot-1.wav"},
    {"id": "shot-2", "text": "…", "audio": "out/shot-2.wav"}
  ]
}
```

`audio` resolves relative to the manifest (or `--audio-dir`); with no `audio`
field the gate reads `<audio-dir>/<id>.wav`. The report lists `status`, `wer`,
`duration`, `wpm`/`cps`, levels, the expected and heard text, and exact errors
and warnings per cue. The exit code is non-zero when any cue fails.

This is a gate, not a pronunciation oracle. Accent, timbre and naturalness still
need a human. Both scripts are covered by `tests/test_asr_qa_cli.py`; the model
backends are exercised only when `mlx-audio` or `faster-whisper` is installed.

## 4. Run

macOS ships `python3`, not `python`; use `python3` in the commands below. Linux
distributions usually provide both, and this pack uses `python3` throughout.
`bash scripts/open_comfyui.sh` launches Desktop on both platforms.

### From the ComfyUI window

Open a workflow, edit the prompt, press **Run**. `i2v`, `r2v` and `i2i` have a
`LoadImage` node with `example.png`; choose your file there. The default canvas
is **0.2 MP** (about 608×352 landscape) and **5 seconds**. Raise megapixels
after the first successful render.

### From the command line

```bash
bash scripts/open_comfyui.sh

python3 scripts/render.py t2v "a paper boat on a rain puddle at golden hour"
python3 scripts/render.py i2v --image photo.jpg --prompt "slow push-in, natural motion"
python3 scripts/render.py flf2v --image start.png --last-image end.png --prompt "smooth transition"
python3 scripts/render.py r2v --image face.png --prompt "Use <Picture 1> as the identity reference"
python3 scripts/render.py t2i "a sunlit reading corner" --output out/room.png
python3 scripts/render.py i2i --image room.jpg --prompt "the same room in winter" --denoise 0.6
python3 scripts/render.py tts "The quick brown fox jumps over the lazy dog." --output out/speech.flac
python3 scripts/render.py asr --audio out/speech.flac --output out/speech.txt

python3 scripts/render.py frame --video out/boat.mp4 --at 1.5 --output out/poster.png
python3 scripts/render.py --list
```

Useful flags: `--seed`, `--width/--height`, `--duration`, `--steps`,
`--denoise`, `--output`, `--dry-run`, `--device`. `--dry-run` prints the
patched API graph without submitting it. `--device auto` (the default) reads
the backend from `/system_stats`; use `--device cuda`, `rocm`, `mps` or `cpu`
to override it. Speech modes add `--language`, `--speaker`, `--instruct`,
`--tts-model`, `--asr-model`, `--precision` and `--max-new-tokens`.

On a CPU backend, `render.py` automatically switches the H3 video modes to the
low-memory profile (Q2 text encoder, 3 s, 0.1 MP) and uses a 24-hour timeout.
Pass `--no-low-memory` to keep the workflow defaults.

Example:

```bash
python3 scripts/render.py t2v \
  "A five-second shot of rain on a neon street, slow tracking camera. Audio: rain, distant traffic, one low synth note." \
  --duration 5 --width 608 --height 352 --seed 7 --output out/rain.mp4
```

## Why this fits 32 GB

The 32 GB profile stores about 31 GB of weights on disk, but ComfyUI loads the
text encoder and the diffusion transformer in separate stages. The resident
working set is roughly 23–26 GB at 0.2 MP, leaving headroom for the OS. On
Apple Silicon that is unified memory; on CUDA/ROCm the text encoder is usually
offloaded to system RAM between stages, so keep at least 32 GB of RAM as well.
On a CPU-only host every tensor is in system RAM.

| Part | File | Size |
|---|---|---:|
| H3 diffusion, Q3_K_M | `MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf` | 8.9 GB |
| H3 text encoder, Q4_K_M | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |
| Turbo LoRA | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB |

The Q4_K_M diffusion file (11.6 GB) also fits 32 GB if you keep other apps
closed; swap the loader filename after downloading it. Q3_K_M is the default so
the first run does not immediately hit swap.

If Desktop exposes **extra server arguments**, add the line for your backend
(`bash scripts/setup_desktop.sh --args` prints it):

| Backend | Startup arguments |
|---|---|
| Apple MPS | `--use-pytorch-cross-attention --cache-none --disable-smart-memory` |
| NVIDIA CUDA | `--cache-none --disable-smart-memory` |
| AMD ROCm | `--use-pytorch-cross-attention --cache-none --disable-smart-memory` |
| CPU-only | `--cpu --cache-none --disable-smart-memory` |

Do **not** use `--force-fp16` with H3 on MPS; it has produced black video.
On CUDA/ROCm `--force-fp16` is not needed either: the GGUF loaders and the
Turbo LoRA already select suitable dtypes.
`--cache-none --disable-smart-memory` is the important part on a 32 GB machine:
it makes ComfyUI unload the text encoder before the transformer needs memory.

### Speed expectations

These are estimates, not promises. A 48 GB M5 Pro measured 6m54s for a 5.17 s
124-frame clip at 0.2 MP with Turbo 8. An M2 with 32 GB is slower: expect
roughly **15–30 minutes** for the same 0.2 MP/5 s job once models are warm, and
roughly double at 0.4 MP. The first run also pays model-load time. Generate a
short clip at 0.2 MP first, then raise resolution.

CUDA is materially faster for this graph than MPS on a 24 GB-class card, but
this pack ships no measured CUDA numbers yet — benchmark the low-memory job
above before committing to a long render. CPU-only is a different category:
expect **hours per sampling step** for H3 at any useful canvas. On CPU, prefer
SDXL stills, the speech modes, and 3 s / 0.1 MP H3 clips.

Measured on a 32 GB M2 during the first real render:

- 0.2 MP, 124 frames, 6 steps: **~235 s per step** for the first three steps.
- Step 4 rose to **~480 s** once the machine ran out of free memory.
- The text encoder (15.4 GB) and diffusion model (8.8 GB) were both resident,
  plus the VAEs; macOS had **16.5 GB of swap in use**.

If you see more than a few GB of swap, stop and restart with the startup
arguments below. A 5 s H3 clip is genuinely a ~25–40 minute job on this class of
machine; it is not comparable to one SDXL cover image.

### Desktop startup arguments (important on 32 GB)

In Comfy Desktop, open the installation settings → **Startup Arguments** and
add the line for your backend:

```text
# Apple Silicon
--use-pytorch-cross-attention --cache-none --disable-smart-memory

# NVIDIA CUDA
--cache-none --disable-smart-memory

# AMD ROCm
--use-pytorch-cross-attention --cache-none --disable-smart-memory

# CPU-only
--cpu --cache-none --disable-smart-memory
```

Restart Desktop. `--cache-none --disable-smart-memory` makes ComfyUI release the
15.4 GB text encoder before loading the diffusion model. Without them the log
says `0 models unloaded` and the machine swaps.

### Low-memory first render

Before trying 0.2 MP / 5 s, prove the pipeline with a smaller job:

```bash
python3 scripts/render.py t2v "a paper boat on a rain puddle at golden hour" \
  --low-memory --output out/boat-first.mp4
```

`--low-memory` selects `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` and defaults to 3 s at
416×256. For a 24 GB-class footprint, download that encoder first:

```bash
PROFILE=24gb bash scripts/download_models.sh core   # PROFILE=cpu is the same set
```

Close Chrome, Slack and other large apps before rendering. Comfy Desktop and
the OS need the remaining headroom.

### If memory runs out

1. Use `PROFILE=cpu` or `PROFILE=24gb` (Q2 text encoder) and re-download just
   that file.
2. Drop the canvas to `0.1 MP` and duration to `3 s`.
3. Quit Chrome, Slack and other large apps.
4. Restart Comfy Desktop so it releases cached models.
5. Keep `--disable-smart-memory` on.
6. On CUDA/ROCm, check the log for tensors left in system RAM; a 32 GB host
   needs the same headroom as a 32 GB Mac.

## Prompting H3

MiniMax H3 generates picture and audio together. Write one block with:

1. **Scene and look** — subject, place, light, lens, texture.
2. **Timeline** — `[0s–1s]`, `[1s–3s]`, … for a 5-second clip.
3. **Camera** — one continuous take or hard cuts; no dissolves unless asked.
4. **Audio** — dialogue, SFX and music. Quote spoken lines exactly.
5. **Negatives** — no text, logos, subtitles, watermarks.

Duration snaps to the model's 17k+5 frame grid at 24 fps. `124` frames is about
5.17 s. The CLI converts `--duration` to a valid frame count.

Canvas sizes (multiple of 32):

| Megapixels | 16:9 | 9:16 |
|---:|---|---|
| 0.1 | 416×256 | 256×416 |
| 0.2 | 608×352 | 352×608 |
| 0.3 | 736×416 | 416×736 |
| 0.4 | 864×480 | 480×864 |
| 0.98 | 1344×768 | 768×1344 |

For R2V, refer to inputs as `<Picture 1>`, `<Picture 2>`, `<Video 1>`,
`<Audio 1>`. Say which reference controls identity, style, motion or voice.

### Turbo on / off

The checked-in H3 workflows use a Turbo LoRA:

- FL2V (T2V/I2V): 6 steps, `res_multistep` / `simple`
- Ref2V (R2V): 4 steps, `res_multistep` / `simple`

For maximum quality, bypass the `LoraLoaderModelOnly` node in the UI, reconnect
the diffusion loader directly to the guider/scheduler, and use 20–25 steps.
Expect several times the render time. For R2V, official guidance prefers 20–25
steps when identity must hold.

## Troubleshooting

**`Missing node type: UnetLoaderGGUF` / `CLIPLoaderGGUF`**
Run `bash scripts/setup_desktop.sh --nodes`, then quit and reopen Desktop.

**`Value not in list: ...gguf`**
The file is missing or has a different name. Run
`bash scripts/download_models.sh check`, then compare with [MODELS.md](MODELS.md).

**`cannot reach ComfyUI at http://127.0.0.1:8188`**
Open Comfy Desktop. If it uses another port, set
`COMFYUI_SERVER=http://127.0.0.1:PORT`. `scripts/open_comfyui.sh` re-reads the
port that Desktop records in `comfy-procs/` while it waits.

**Black video / NaN audio**
Remove `--force-fp16` from Desktop's server arguments. Update ComfyUI Desktop.
Stay at or below about 0.4 MP per pass on MPS.

**`ERROR lora ... adaln_proj` in the log**
The Turbo LoRA contains dense AdaLN keys that a pruned checkpoint cannot apply.
ComfyUI skips them; generation still works. If the log bothers you, install
`PlagueKind/ComfyUI-PlagueKind-Nodes` and put `H3AdaLNLoRAFix` after the LoRA
loader. This pack does not require it.

**Render appears to take hours, or the progress bar jumps**
The machine went to sleep mid-render. The ComfyUI log may say
`Prompt executed in 262 seconds` while the progress bar shows 17 minutes.
Prevent sleep during long jobs:

```bash
# macOS
caffeinate -dimsu python3 scripts/render.py t2v "..." --output out/clip.mp4

# Linux (systemd)
systemd-inhibit --what=sleep:idle --why="ComfyUI render" \
  python3 scripts/render.py t2v "..." --output out/clip.mp4
```

**Very slow first run**
Model load dominates. The second run in the same Desktop session is faster.
Keep `--cache-none` for memory safety, not speed.

**ffmpeg not found for `frame`**
`brew install ffmpeg` on macOS; `sudo apt install ffmpeg` or
`sudo dnf install ffmpeg` on Linux.

**Desktop will not open on Linux**
Launch it from the application menu, or run the AppImage directly with
`--no-sandbox` if the Electron sandbox is blocked. To point the helper script
at a specific binary:
`COMFY_DESKTOP_CMD="/path/to/comfyui-desktop" bash scripts/open_comfyui.sh`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The default run covers graph structure, the CLI (including ASR/TTS against a
fake ComfyUI server), the standalone ASR/QA scripts' formatting, normalisation,
backend selection and report logic, shell-script syntax and helper detection,
and checks that the checked-in JSON matches `scripts/build_workflows.py`. The
live TTS → ASR round trip is opt-in and keeps its speech, transcript, logs and
`result.json` under `.test/tts_asr_roundtrip/` (gitignored):

```bash
COMFY_ROUNDTRIP=1 python3 -m unittest tests.test_tts_asr_roundtrip -v
```

Set `COMFYUI_SERVER` if ComfyUI uses a port other than 8188 and
`COMFY_ROUNDTRIP_DIR` to keep the artifacts elsewhere.

## Licensing and attribution

MiniMax H3 weights are released under the **MiniMax H3 Community License**.
Local commercial use requires a separate license from Comfy, the official
reseller. Generated output must retain the required **“Powered by MiniMax H3”**
attribution. SDXL base 1.0 is under the CreativeML Open RAIL++-M license.
Review all of them before publishing. The optional Qwen3-TTS and Qwen3-ASR node
packs and weights have their own licenses; the 0.6B model cards currently
identify Apache-2.0. Verify upstream terms before redistribution or commercial
deployment.

## Layout

```text
comfy/
├── README.md
├── MODELS.md
├── NOTICE.md
├── assets/                   # example.png placeholder for image loaders
├── prompts/                  # copy-paste prompt examples
├── scripts/
│   ├── _comfy_root.sh        # finds Desktop on macOS/Linux, backend + models
│   ├── setup_desktop.sh      # check / nodes / workflows / speech graphs / models
│   ├── download_models.sh    # resumable model downloads and profile aliases
│   ├── open_comfyui.sh       # open Desktop and wait for the API
│   ├── render.py             # CLI: t2v, i2v, flf2v, r2v, t2i, i2i, tts, asr, frame
│   └── build_workflows.py    # regenerates the checked-in JSON graphs
├── tests/                    # CLI, graph and shell-helper tests
└── workflows/
    ├── *.json                # UI graphs for Desktop, including speech
    └── api/*.api.json        # API graphs used by render.py
```
