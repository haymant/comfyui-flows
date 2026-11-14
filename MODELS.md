# Model manifest

All paths are relative to `ComfyUI/models/` inside ComfyUI Desktop.
`download_models.sh` writes them there and resumes partial downloads.

## Core — T2V and I2V (~31 GB)

| Destination | File | Size | Source |
|---|---|---:|---|
| `diffusion_models/` | `MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf` | 8.9 GB | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| `text_encoders/` | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `vae/` | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `loras/` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

## R2V — reference-to-video (+11 GB)

| Destination | File | Size | Source |
|---|---|---:|---|
| `diffusion_models/` | `MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf` | 8.9 GB | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| `loras/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 2.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

## SDXL — still images (+7 GB)

| Destination | File | Size | Source |
|---|---|---:|---|
| `checkpoints/` | `sd_xl_base_1.0.safetensors` | 6.9 GB | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |

## Profile alternatives

| Profile | Diffusion | Text encoder | Intended machine |
|---|---|---|---|
| `32gb` (default) | FL2VA/Ref2VA Q3_K_M | Q4_K_M | 32 GB unified memory |
| `24gb` | FL2VA/Ref2VA Q3_K_M | Q2_K | 24 GB unified memory |
| `quality` | FL2VA/Ref2VA Q5_K_M | Q4_K_M | 64 GB+ unified memory |

The `24gb` and `quality` profiles change only the filenames in the downloader;
the workflows stay the same. If you switch profiles, update the loader widgets
in the UI (or run `render.py --workflow` against a copy of the API graph with the
new names).

## Notes

- Use the **GGUF** text encoder, not the NVFP4-AWQ one. NVFP4 is CUDA-only.
- `ComfyUI-GGUF` needs `gguf==0.18.0`; `setup_desktop.sh --nodes` installs it.
- The H3 GGUF files are pruned checkpoints. The ComfyUI log may skip incompatible
  AdaLN LoRA keys; that is expected and harmless with the Turbo LoRA.
- MiniMax H3 weights are governed by the MiniMax H3 Community License. Local
  commercial use needs a license from Comfy, and output must carry the
  “Powered by MiniMax H3” attribution.
