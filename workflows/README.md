# Workflows

Open a `.json` UI graph directly in ComfyUI Desktop, or install the image/video graphs with:

```bash
bash scripts/setup_desktop.sh --workflows
```

Install the optional ASR graph separately:

```bash
bash scripts/setup_desktop.sh --asr-workflow
```

| File | Mode | Needs |
|---|---|---|
| `minimax_h3_t2v_32gb.json` | text → video + audio | MiniMax H3 core models and ComfyUI-GGUF |
| `minimax_h3_i2v_32gb.json` | image → video + audio | MiniMax H3 core models and ComfyUI-GGUF |
| `minimax_h3_flf2v_32gb.json` | start + end image → video + audio | MiniMax H3 core models and ComfyUI-GGUF |
| `minimax_h3_r2v_32gb.json` | reference image(s) → video | core + R2V models and ComfyUI-GGUF |
| `sdxl_text_to_image.json` | text → still | SDXL checkpoint |
| `sdxl_image_to_image.json` | image + text → still | SDXL checkpoint |
| `qwen3_asr_local.json` | local audio → transcript text | optional ComfyUI-Qwen3-ASR node and Qwen3-ASR model |

The `api/` folder holds API-format equivalents. The existing `scripts/render.py`
command supports the image/video graphs, not ASR. The ASR graph defaults to the
0.6B model with automatic language detection. Word/character timestamps are
optional and require the separate Qwen forced-aligner model. See the root README
for install and runtime-compatibility notes.

Defaults are deliberately small for a 32 GB Mac:

- H3 canvas: 0.2 MP (about 608×352 at 16:9), 124 frames (~5.17 s at 24 fps)
- H3 sampling: 6 steps for T2V/I2V (Turbo LoRA), 4 steps for R2V (Ref2V LoRA)
- SDXL: 1024×1024, 24 steps, CFG 6, `dpmpp_2m` / `karras`

Raise megapixels only after the first successful render.
