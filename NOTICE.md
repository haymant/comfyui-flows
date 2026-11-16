# Notices

This repository contains workflow JSON, shell scripts and Python scripts. It
does **not** contain model weights.

## MiniMax H3

MiniMax H3 model weights are released under the **MiniMax H3 Community
License**. Review the license before downloading or using the weights.

- Local commercial use of generated output requires a commercial license from
  Comfy, the official reseller.
- Published output must retain the required **“Powered by MiniMax H3”**
  attribution.
- The MiniMax H3 open release is H3-Base (FL2VA and Ref2VA); 2K output is a
  hosted-service feature and needs a separate upscale pass locally.
- The GGUF quantizations used here are community conversions of the pruned H3
  checkpoints. Their licenses follow the upstream MiniMax H3 license.

Upstream: <https://github.com/MiniMax-AI/MiniMax-H3>

## SDXL

`sd_xl_base_1.0.safetensors` is Stable Diffusion XL Base 1.0, released under the
CreativeML Open RAIL++-M license. Review it before publishing generated images.

Upstream: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>

## Optional Qwen3-ASR

The optional ASR workflow depends on the separately installed
<https://github.com/DarioFT/ComfyUI-Qwen3-ASR> custom-node project. It is not
bundled here; consult its current license and dependency terms before installing
or redistributing it. The default Qwen3-ASR 0.6B weights are hosted at
<https://huggingface.co/Qwen/Qwen3-ASR-0.6B> and the model card identifies them
as Apache-2.0. Other model and forced-aligner cards may have separate terms.

## ComfyUI and custom nodes

ComfyUI is licensed under GPL-3.0. `ComfyUI-GGUF` is installed separately from
<https://github.com/city96/ComfyUI-GGUF> and keeps its own license.
