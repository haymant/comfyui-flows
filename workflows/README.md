# Workflows

Open the `.json` files directly in ComfyUI Desktop, or install them with:

```bash
bash scripts/setup_desktop.sh --workflows
```

| File | Mode | Needs |
|---|---|---|
| `minimax_h3_t2v_32gb.json` | text → video + audio | core models |
| `minimax_h3_i2v_32gb.json` | image → video + audio | core models |
| `minimax_h3_flf2v_32gb.json` | start + end image → video + audio | core models |
| `minimax_h3_r2v_32gb.json` | reference image(s) → video | core + R2V models |
| `sdxl_text_to_image.json` | text → still | SDXL checkpoint |
| `sdxl_image_to_image.json` | image + text → still | SDXL checkpoint |

The `api/` folder holds the API-format equivalents used by `scripts/render.py`.
Edit the UI versions in Desktop, then use **Save (API format)** if you want the
CLI to use your changes — or regenerate everything with
`python3 scripts/build_workflows.py` after editing the builder.

Defaults are deliberately small for a 32 GB Mac:

- H3 canvas: 0.2 MP (about 608×352 at 16:9), 124 frames (~5.17 s at 24 fps)
- H3 sampling: 6 steps for T2V/I2V (Turbo LoRA), 4 steps for R2V (Ref2V LoRA)
- SDXL: 1024×1024, 24 steps, CFG 6, `dpmpp_2m` / `karras`

Raise megapixels only after the first successful render.
