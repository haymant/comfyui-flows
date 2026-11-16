#!/bin/bash
# Install/check the pieces this project needs inside ComfyUI Desktop.
#
# Works with ComfyUI Desktop on macOS (Apple Silicon / MPS) and Linux
# (NVIDIA CUDA, AMD ROCm, or CPU-only x86-64).
#
#   bash scripts/setup_desktop.sh --check
#   bash scripts/setup_desktop.sh --args
#   bash scripts/setup_desktop.sh --nodes
#   bash scripts/setup_desktop.sh --workflows
#   bash scripts/setup_desktop.sh --asr-workflow
#   bash scripts/setup_desktop.sh --tts-workflow
#   bash scripts/setup_desktop.sh --speech-workflows
#   bash scripts/setup_desktop.sh --models core
#   bash scripts/setup_desktop.sh --all
#
# --all installs the ComfyUI-GGUF node pack, copies the workflows into the
# Desktop workflow browser, and downloads the 32 GB core model set. It does not
# download the optional R2V or SDXL models; use --models all for those.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_comfy_root.sh"

ROOT="$(comfy_root || true)"
if [ -z "$ROOT" ]; then
  cat >&2 <<'EOF'
Could not find ComfyUI Desktop.

Open Comfy Desktop once and create a local instance, then run this again.
If it lives somewhere unusual:
  COMFY_ROOT="/path/to/ComfyUI" bash scripts/setup_desktop.sh --check

Linux standalone instances default to ~/ComfyUI-Installs/<name>/ComfyUI.
On a manual install, point COMFY_ROOT at the directory that contains main.py.
EOF
  exit 1
fi

PY="$(comfy_python "$ROOT")"
SERVER="$(comfy_server)"
MODELS="$(comfy_models "$ROOT")"
PLATFORM="$(comfy_platform)"
DEVICE="$(comfy_device)"
STARTUP_ARGS="$(comfy_startup_args)"

check_models() {
  local missing=0
  local files=(
    "diffusion_models/MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf"
    "vae/minimax_h3_video_vae_fp16.safetensors"
    "vae/minimax_h3_audio_vae_fp32.safetensors"
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
  )
  local encoders=(
    "text_encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
    "text_encoders/qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
  )
  local encoder_found=0
  echo "models ($MODELS):"
  for f in "${files[@]}"; do
    if [ -s "$MODELS/$f" ]; then
      echo "  ok   $f"
    else
      echo "  MISS $f"
      missing=$((missing + 1))
    fi
  done
  for f in "${encoders[@]}"; do
    if [ -s "$MODELS/$f" ]; then
      echo "  ok   $f"
      encoder_found=1
    fi
  done
  if [ "$encoder_found" -eq 0 ]; then
    echo "  MISS text_encoders/qwen3vl-32B-MiniMax-H3-Q{4,2}_K_M.gguf"
    missing=$((missing + 1))
  fi
  if [ "$missing" -gt 0 ]; then
    echo "  -> run: bash scripts/download_models.sh core"
  fi
}

check() {
  echo "ComfyUI Desktop"
  echo "  platform: $PLATFORM"
  echo "  device:   $DEVICE (auto-detected; override with COMFY_DEVICE)"
  echo "  root:     $ROOT"
  echo "  python:   $PY"
  echo "  server:   $SERVER"
  echo "  version:  $(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/comfyui_version.py" 2>/dev/null || echo unknown)"
  if [ -d "$ROOT/custom_nodes/ComfyUI-GGUF" ]; then
    echo "  GGUF node: installed"
  else
    echo "  GGUF node: MISSING -> bash scripts/setup_desktop.sh --nodes"
  fi
  if curl -sf -m 2 "$SERVER/system_stats" >/dev/null 2>&1; then
    echo "  server:   running"
  else
    echo "  server:   not running -> bash scripts/open_comfyui.sh"
  fi
  check_models
  echo "recommended Desktop startup arguments for $DEVICE:"
  echo "  $STARTUP_ARGS"
  case "$DEVICE" in
    cpu)
      echo
      echo "CPU-only: use the low-memory profile and small jobs."
      echo "  PROFILE=cpu bash scripts/download_models.sh core"
      echo "  python3 scripts/render.py t2v \"...\" --low-memory"
      echo "Expect minutes per sampling step, not seconds."
      ;;
    cuda|rocm)
      echo
      echo "Use the GGUF text encoder (already the default). Do not add --force-fp16."
      echo "Keep ComfyUI on loopback; the local API is unauthenticated."
      ;;
  esac
}

install_nodes() {
  local cn="$ROOT/custom_nodes"
  local dest="$cn/ComfyUI-GGUF"
  [ -d "$cn" ] || { echo "no custom_nodes at $cn" >&2; exit 1; }
  if [ -d "$dest/.git" ]; then
    echo "ComfyUI-GGUF already cloned; leaving it as-is"
  else
    echo "cloning ComfyUI-GGUF into $dest"
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "$dest"
  fi
  echo "installing ComfyUI-GGUF requirements into the Desktop environment"
  if [ -f "$dest/requirements.txt" ]; then
    "$PY" -m pip install -q -r "$dest/requirements.txt"
  fi
  "$PY" -m pip install -q --force-reinstall --no-deps "gguf==0.18.0"
  echo
  echo "Quit and reopen Comfy Desktop so it loads the GGUF nodes."
}

install_workflows() {
  local dest="$ROOT/user/default/workflows/linkzag-minimax-h3"
  mkdir -p "$dest"
  cp "$HERE/../workflows/minimax_h3_"*.json "$dest/"
  cp "$HERE/../workflows/sdxl_"*.json "$dest/"
  mkdir -p "$ROOT/input"
  if [ ! -f "$ROOT/input/example.png" ]; then
    cp "$HERE/../assets/example.png" "$ROOT/input/example.png"
  fi
  echo "copied workflows to $dest"
  echo "seeded $ROOT/input/example.png for the image loaders"
  echo "In Comfy Desktop: Workflows -> linkzag-minimax-h3 -> choose a workflow."
}

install_asr_workflow() {
  local dest="$ROOT/user/default/workflows/linkzag-asr"
  mkdir -p "$dest"
  cp "$HERE/../workflows/qwen3_asr_local.json" "$dest/"
  echo "copied the local ASR workflow to $dest"
  echo "Install the optional Qwen3-ASR custom node pack before opening it."
  echo "In Comfy Desktop: Workflows -> linkzag-asr -> qwen3_asr_local."
}

install_tts_workflow() {
  local dest="$ROOT/user/default/workflows/linkzag-tts"
  mkdir -p "$dest"
  cp "$HERE/../workflows/qwen3_tts_local.json" "$dest/"
  echo "copied the local TTS workflow to $dest"
  echo "Install the optional Qwen3-TTS custom node pack before opening it."
  echo "In Comfy Desktop: Workflows -> linkzag-tts -> qwen3_tts_local."
}

install_speech_workflows() {
  install_asr_workflow
  install_tts_workflow
}

usage() {
  cat <<'EOF'
Install/check the pieces this project needs inside ComfyUI Desktop.

  bash scripts/setup_desktop.sh --check
  bash scripts/setup_desktop.sh --args
  bash scripts/setup_desktop.sh --nodes
  bash scripts/setup_desktop.sh --workflows
  bash scripts/setup_desktop.sh --asr-workflow
  bash scripts/setup_desktop.sh --tts-workflow
  bash scripts/setup_desktop.sh --speech-workflows
  bash scripts/setup_desktop.sh --models core
  bash scripts/setup_desktop.sh --all

--all installs the ComfyUI-GGUF node pack, copies the workflows into the
Desktop workflow browser, and downloads the 32 GB core model set. It does not
download the optional R2V or SDXL models; use --models all for those.

Environment:
  COMFY_ROOT       path to the ComfyUI directory if auto-detection fails
  COMFY_MODELS     models directory (default: $COMFY_ROOT/models or ~/ComfyUI-Shared)
  COMFY_DEVICE     mps | cuda | rocm | cpu
  PROFILE          32gb (default) | 24gb | cpu | quality
EOF
}

ACTION="${1:---check}"
case "$ACTION" in
  --check|check) check ;;
  --args|args) echo "$STARTUP_ARGS" ;;
  --nodes|nodes) install_nodes ;;
  --workflows|workflows) install_workflows ;;
  --asr-workflow|asr-workflow) install_asr_workflow ;;
  --tts-workflow|tts-workflow) install_tts_workflow ;;
  --speech-workflows|speech-workflows) install_speech_workflows ;;
  --models|models)
    PROFILE="${PROFILE:-32gb}" COMFY_ROOT="$ROOT" COMFY_MODELS="$MODELS" \
      bash "$HERE/download_models.sh" "${2:-core}"
    ;;
  --all|all)
    install_nodes
    install_workflows
    PROFILE="${PROFILE:-32gb}" COMFY_ROOT="$ROOT" COMFY_MODELS="$MODELS" \
      bash "$HERE/download_models.sh" core
    echo
    echo "Done. Quit and reopen Comfy Desktop, then run:"
    echo "  python3 scripts/render.py t2v \"a paper boat on a rain puddle\""
    ;;
  -h|--help|help) usage ;;
  *) echo "unknown option: $ACTION" >&2; usage >&2; exit 2 ;;
esac
