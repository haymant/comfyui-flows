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

## Optional speech — local TTS and ASR

The Qwen3-TTS and Qwen3-ASR custom nodes download model weights on first use
unless a local model directory is supplied. These optional models are separate
from the H3/SDXL profiles and are not fetched by `download_models.sh`.

| ComfyUI models path | Model | Source | License |
|---|---|---|---|
| `Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice/` | TTS 0.6B CustomVoice (workflow default) | [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | Check current model card |
| `Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice/` | TTS 1.7B CustomVoice (optional quality alternative) | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Check current model card |
| `Qwen3-ASR/Qwen3-ASR-0.6B/` | Qwen3-ASR 0.6B (workflow default) | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Apache-2.0 per model card |
| `Qwen3-ASR/Qwen3-ASR-1.7B/` | Qwen3-ASR 1.7B (optional quality alternative) | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Check current model card |
| `Qwen3-ASR/Qwen3-ForcedAligner-0.6B/` | Optional forced aligner for timestamps | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Check current model card |

Model downloads are user-triggered by running the speech workflows. Pre-provision
the weights for offline operation. Do not install speech packages as part of the
default image/video setup. On CPU-only hosts `render.py` selects `fp32`
automatically (the workflow defaults are `bf16` for TTS and `fp16` for ASR).

### Standalone Whisper (no ComfyUI)

`scripts/asr.py` and `scripts/qa_audio.py` use a Whisper model directly instead
of the ComfyUI graph. They do not touch `ComfyUI/models/`; the weights live in
the Hugging Face cache (`~/.cache/huggingface/hub`).

| Host | Default model | Cache size | Source |
|---|---|---:|---|
| macOS Apple Silicon (MLX) | `mlx-community/whisper-large-v3-turbo-asr-fp16` | ~1.6 GB | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |
| Linux CUDA / CPU (faster-whisper) | `large-v3-turbo` (CTranslate2) | ~1.6 GB | [Systran/faster-whisper-large-v3-turbo](https://huggingface.co/Systran/faster-whisper-large-v3-turbo) |

Override with `ASR_MODEL`. Whisper large-v3-turbo is the same model family the
Qwen3-ASR graph does not replace: Whisper emits sentence timestamps natively and
runs one pass over long audio, which is why the pack recommends it for
transcripts and cue QA. Keep Qwen3-ASR for short interactive clips in the graph.

## Profile alternatives

| Profile | Diffusion | Text encoder | Intended machine |
|---|---|---|---|
| `32gb` (default) | FL2VA/Ref2VA Q3_K_M | Q4_K_M | 32 GB unified memory or 32 GB RAM + 24 GB VRAM |
| `24gb` | FL2VA/Ref2VA Q3_K_M | Q2_K | 24 GB unified memory / VRAM |
| `cpu` | FL2VA/Ref2VA Q3_K_M | Q2_K | CPU-only or low-RAM host (alias for `24gb`) |
| `quality` | FL2VA/Ref2VA Q5_K_M | Q4_K_M | 64 GB+ unified memory |

The `24gb` and `quality` profiles change only the filenames in the downloader;
the workflows stay the same. If you switch profiles, update the loader widgets
in the UI (or run `render.py --workflow` against a copy of the API graph with the
new names).

## Notes

- Use the **GGUF** text encoder, not the NVFP4-AWQ one. NVFP4 needs a CUDA
  build with that quantisation support; the GGUF encoder works on MPS, CUDA,
  ROCm and CPU.
- `ComfyUI-GGUF` needs `gguf==0.18.0`; `setup_desktop.sh --nodes` installs it.
- The H3 GGUF files are pruned checkpoints. The ComfyUI log may skip incompatible
  AdaLN LoRA keys; that is expected and harmless with the Turbo LoRA.
- MiniMax H3 weights are governed by the MiniMax H3 Community License. Local
  commercial use needs a license from Comfy, and output must carry the
  “Powered by MiniMax H3” attribution.
