#!/bin/bash
# Download model files into the ComfyUI Desktop models/ directory.
#
#   bash scripts/download_models.sh core    # MiniMax H3 T2V / I2V (~31 GB)
#   bash scripts/download_models.sh r2v     # add reference-to-video (~11 GB)
#   bash scripts/download_models.sh sdxl    # add still-image generation (~7 GB)
#   bash scripts/download_models.sh all     # everything (~49 GB)
#   bash scripts/download_models.sh check
#
# Profiles (memory, not operating system; they apply to macOS and Linux):
#   PROFILE=32gb     default: Q3 diffusion + Q4 text encoder, fits 32 GB
#   PROFILE=24gb     Q3 diffusion + Q2 text encoder, for 24 GB machines
#   PROFILE=cpu      alias for 24gb; use on CPU-only or low-RAM x86-64 hosts
#   PROFILE=quality  Q5 diffusion + Q4 text encoder, for 64 GB+ machines
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_comfy_root.sh"

ROOT="$(comfy_root || true)"
if [ -z "$ROOT" ]; then
  echo "Could not find ComfyUI Desktop. Set COMFY_ROOT or open Comfy Desktop once." >&2
  exit 1
fi

MODELS="$(comfy_models "$ROOT")"
PROFILE="${PROFILE:-32gb}"
DEVICE="$(comfy_device)"

case "$PROFILE" in
  32gb)
    FL2V_FILE="MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf"
    FL2V_MIN=8300
    REF2V_FILE="MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf"
    REF2V_MIN=8300
    QWEN_FILE="qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
    QWEN_MIN=13600
    ;;
  24gb|cpu)
    PROFILE=cpu
    FL2V_FILE="MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf"
    FL2V_MIN=8300
    REF2V_FILE="MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf"
    REF2V_MIN=8300
    QWEN_FILE="qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
    QWEN_MIN=7900
    ;;
  quality)
    FL2V_FILE="MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf"
    FL2V_MIN=13000
    REF2V_FILE="MiniMax-H3-Ref2VA-Pruned-Q5_K_M.gguf"
    REF2V_MIN=13000
    QWEN_FILE="qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
    QWEN_MIN=13600
    ;;
  *)
    echo "unknown PROFILE=$PROFILE (use 32gb, 24gb, cpu or quality)" >&2
    exit 2
    ;;
esac

file_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || wc -c <"$1" | tr -d ' '
}

download() {
  local url="$1" dest="$2" min_mb="$3"
  local min_bytes=$((min_mb * 1000000))
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    local have
    have="$(file_size "$dest")"
    if [ "$have" -ge "$min_bytes" ]; then
      echo "have   $(basename "$dest")"
      return
    fi
    echo "remove incomplete $(basename "$dest")"
    rm -f "$dest"
  fi
  echo "get    $(basename "$dest")  (~$((min_mb / 1000)) GB)"
  local attempt
  for attempt in $(seq 1 20); do
    if [ -f "$dest.part" ]; then
      curl -L --fail --connect-timeout 30 --speed-limit 1024 --speed-time 60 -C - -o "$dest.part" "$url" && break
    else
      curl -L --fail --connect-timeout 30 --speed-limit 1024 --speed-time 60 -o "$dest.part" "$url" && break
    fi
    echo "  network stalled; resuming ($attempt/20)"
    sleep 5
  done
  if [ ! -f "$dest.part" ]; then
    echo "download failed: $dest" >&2
    exit 1
  fi
  local got
  got="$(file_size "$dest.part")"
  if [ "$got" -lt "$min_bytes" ]; then
    echo "download too small ($got bytes): $dest.part" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

core() {
  download "https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/resolve/main/$FL2V_FILE" \
    "$MODELS/diffusion_models/$FL2V_FILE" "$FL2V_MIN"
  download "https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/$QWEN_FILE" \
    "$MODELS/text_encoders/$QWEN_FILE" "$QWEN_MIN"
  download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
    "$MODELS/vae/minimax_h3_video_vae_fp16.safetensors" 4900
  download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
    "$MODELS/vae/minimax_h3_audio_vae_fp32.safetensors" 550
  download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
    "$MODELS/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" 1800
}

r2v() {
  download "https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/resolve/main/$REF2V_FILE" \
    "$MODELS/diffusion_models/$REF2V_FILE" "$REF2V_MIN"
  download "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
    "$MODELS/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" 1800
}

sdxl() {
  download "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "$MODELS/checkpoints/sd_xl_base_1.0.safetensors" 6400
}

check_all() {
  local missing=0
  local files=(
    "diffusion_models/$FL2V_FILE"
    "text_encoders/$QWEN_FILE"
    "vae/minimax_h3_video_vae_fp16.safetensors"
    "vae/minimax_h3_audio_vae_fp32.safetensors"
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    "diffusion_models/$REF2V_FILE"
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    "checkpoints/sd_xl_base_1.0.safetensors"
  )
  for f in "${files[@]}"; do
    if [ -s "$MODELS/$f" ]; then
      echo "ok   $f"
    else
      echo "MISS $f"
      missing=$((missing + 1))
    fi
  done
  [ "$missing" -eq 0 ]
}

ACTION="${1:-core}"
echo "ComfyUI models: $MODELS  (profile: $PROFILE, device: $DEVICE)"
if [ "$DEVICE" = "cpu" ] && [ "$PROFILE" != "cpu" ]; then
  echo "note: no GPU detected. Consider PROFILE=cpu for a smaller text encoder." >&2
fi
case "$ACTION" in
  core) core ;;
  r2v) r2v ;;
  sdxl) sdxl ;;
  all) core; r2v; sdxl ;;
  check) check_all ;;
  *)
    echo "usage: $0 [core|r2v|sdxl|all|check]" >&2
    exit 2
    ;;
esac

if [ "$ACTION" != "check" ]; then
  echo
  echo "Done. MiniMax H3 weights are under the MiniMax H3 Community License."
  echo "Keep the required 'Powered by MiniMax H3' attribution on published output."
fi
