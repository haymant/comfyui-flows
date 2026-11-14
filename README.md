# comfy — standalone MiniMax H3 + SDXL workflows

A self-contained ComfyUI workflow pack for **ComfyUI Desktop on Apple Silicon**.
It generates video with native stereo audio from text or sample images, and
still images from text or sample images.

Nothing here imports the rest of this repository. The only contract is:

- ComfyUI Desktop is installed.
- The model files live in `ComfyUI/models/` (the scripts find it for you).
- The CLI talks to the Desktop server at `http://127.0.0.1:8188`.

The default profile targets a **MacBook with 32 GB unified memory (M2 Pro/Max)**.
It uses MiniMax H3 GGUF weights sized to fit in 32 GB, with a Turbo LoRA so a
5-second clip is a coffee break rather than an afternoon.

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

`workflows/api/` contains the same graphs in ComfyUI API format, which is what
`scripts/render.py` submits. The files in `workflows/` are UI graphs — drop them
into Comfy Desktop or open them from the workflow browser.

MiniMax H3 is a video model. Still images use SDXL; you can also pull a PNG from
any H3 clip with the `frame` mode.

## 1. Install the one custom node

MiniMax H3's GGUF loaders are not in core ComfyUI. Everything else is core.

```bash
bash scripts/setup_desktop.sh --check     # shows the detected paths
bash scripts/setup_desktop.sh --nodes     # installs ComfyUI-GGUF
```

Quit and reopen Comfy Desktop afterwards. If you prefer the GUI, install
**ComfyUI-GGUF** from ComfyUI Manager instead.

## 2. Download models

```bash
bash scripts/download_models.sh core      # ~31 GB, T2V + I2V
bash scripts/download_models.sh r2v       # +11 GB, reference-to-video
bash scripts/download_models.sh sdxl      # +7 GB, still images
```

`PROFILE=24gb` swaps the 14.6 GB Qwen encoder for the 8.5 GB Q2 version.
`PROFILE=quality` uses Q5 H3 weights; only sensible on 64 GB or more.

The exact files, URLs and sizes are in [MODELS.md](MODELS.md).

## 3. Install the workflows in Desktop

```bash
bash scripts/setup_desktop.sh --workflows
```

They appear under **Workflows → linkzag-minimax-h3**. Or open a JSON directly:
`Workflows → Open` and pick a file from `workflows/`. The installer also seeds
`ComfyUI/input/example.png` so the image loaders have a valid placeholder.

## 4. Run

macOS ships `python3`, not `python`; use `python3` in the commands below.

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

python3 scripts/render.py frame --video out/boat.mp4 --at 1.5 --output out/poster.png
python3 scripts/render.py --list
```

Useful flags: `--seed`, `--width/--height`, `--duration`, `--steps`,
`--denoise`, `--output`, `--dry-run`. `--dry-run` prints the patched API graph
without submitting it.

Example:

```bash
python3 scripts/render.py t2v \
  "A five-second shot of rain on a neon street, slow tracking camera. Audio: rain, distant traffic, one low synth note." \
  --duration 5 --width 608 --height 352 --seed 7 --output out/rain.mp4
```

## Why this fits 32 GB

The 32 GB profile stores about 31 GB of weights on disk, but ComfyUI loads the
text encoder and the diffusion transformer in separate stages. The resident
working set is roughly 23–26 GB at 0.2 MP, which leaves room for macOS.

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

If Desktop exposes **extra server arguments**, add:

```text
--use-pytorch-cross-attention --reserve-vram 6 --cache-none --disable-smart-memory
```

Do **not** use `--force-fp16` with H3 on MPS; it has produced black video.
`--cache-none --disable-smart-memory` is the important part on a 32 GB machine:
it makes ComfyUI unload the text encoder before the transformer needs memory.

### Speed expectations

These are estimates, not promises. A 48 GB M5 Pro measured 6m54s for a 5.17 s
124-frame clip at 0.2 MP with Turbo 8. An M2 with 32 GB is slower: expect
roughly **15–30 minutes** for the same 0.2 MP/5 s job once models are warm, and
roughly double at 0.4 MP. The first run also pays model-load time. Generate a
short clip at 0.2 MP first, then raise resolution.

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
add:

```text
--use-pytorch-cross-attention --cache-none --disable-smart-memory
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
PROFILE=24gb bash scripts/download_models.sh core
```

Close Chrome, Slack and other large apps before rendering. Comfy Desktop and
macOS need the remaining headroom.

### If memory runs out

1. Use `PROFILE=24gb` (Q2 text encoder) and re-download just that file.
2. Drop the canvas to `0.1 MP` and duration to `3 s`.
3. Quit Chrome, Slack and other large apps.
4. Restart Comfy Desktop so it releases cached models.
5. Keep `--disable-smart-memory` on.

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
`COMFYUI_SERVER=http://127.0.0.1:PORT`.

**Black video / NaN audio**
Remove `--force-fp16` from Desktop's server arguments. Update ComfyUI Desktop.
Stay at or below about 0.4 MP per pass on MPS.

**`ERROR lora ... adaln_proj` in the log**
The Turbo LoRA contains dense AdaLN keys that a pruned checkpoint cannot apply.
ComfyUI skips them; generation still works. If the log bothers you, install
`PlagueKind/ComfyUI-PlagueKind-Nodes` and put `H3AdaLNLoRAFix` after the LoRA
loader. This pack does not require it.

**Render appears to take hours, or the progress bar jumps**
The Mac went to sleep mid-render. The ComfyUI log may say
`Prompt executed in 262 seconds` while the progress bar shows 17 minutes. Run
long jobs under `caffeinate`:

```bash
caffeinate -dimsu python3 scripts/render.py t2v "..." --output out/clip.mp4
```

**Very slow first run**
Model load dominates. The second run in the same Desktop session is faster.
Keep `--cache-none` for memory safety, not speed.

**ffmpeg not found for `frame`**
`brew install ffmpeg`.

## Licensing and attribution

MiniMax H3 weights are released under the **MiniMax H3 Community License**.
Local commercial use requires a separate license from Comfy, the official
reseller. Generated output must retain the required **“Powered by MiniMax H3”**
attribution. SDXL base 1.0 is under the CreativeML Open RAIL++-M license.
Review both before publishing.

## Layout

```text
comfy/
├── README.md
├── MODELS.md
├── NOTICE.md
├── assets/                   # example.png placeholder for image loaders
├── prompts/                  # copy-paste prompt examples
├── scripts/
│   ├── _comfy_root.sh        # finds ComfyUI Desktop
│   ├── setup_desktop.sh      # check / nodes / workflows / models
│   ├── download_models.sh    # resumable model downloads
│   ├── open_comfyui.sh       # open Desktop and wait for the API
│   ├── render.py             # CLI: t2v, i2v, r2v, t2i, i2i, frame
│   └── build_workflows.py    # regenerates the checked-in JSON graphs
└── workflows/
    ├── *.json                # UI graphs for Comfy Desktop
    └── api/*.api.json        # API graphs used by render.py
```
