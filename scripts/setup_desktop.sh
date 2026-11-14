#!/bin/bash
# Install/check the pieces this project needs inside ComfyUI Desktop.
#
#   bash scripts/setup_desktop.sh --check
#   bash scripts/setup_desktop.sh --nodes
#   bash scripts/setup_desktop.sh --workflows
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

Open Comfy Desktop once, then run this again. If it lives somewhere unusual:
  COMFY_ROOT="/path/to/ComfyUI" bash scripts/setup_desktop.sh --check
EOF
  exit 1
fi

PY="$(comfy_python "$ROOT")"
SERVER="$(comfy_server)"

check_models() {
  local models="$ROOT/models" missing=0
  local files=(
    "diffusion_models/MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf"
    "text_encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
    "vae/minimax_h3_video_vae_fp16.safetensors"
    "vae/minimax_h3_audio_vae_fp32.safetensors"
    "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
  )
  echo "models:"
  for f in "${files[@]}"; do
    if [ -s "$models/$f" ]; then
      echo "  ok   $f"
    else
      echo "  MISS $f"
      missing=$((missing + 1))
    fi
  done
  if [ "$missing" -gt 0 ]; then
    echo "  -> run: bash scripts/download_models.sh core"
  fi
}

check() {
  echo "ComfyUI Desktop"
  echo "  root:    $ROOT"
  echo "  python:  $PY"
  echo "  server:  $SERVER"
  echo "  version: $(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/comfyui_version.py" 2>/dev/null || echo unknown)"
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
  cp "$HERE/../workflows/"*.json "$dest/"
  mkdir -p "$ROOT/input"
  if [ ! -f "$ROOT/input/example.png" ]; then
    cp "$HERE/../assets/example.png" "$ROOT/input/example.png"
  fi
  echo "copied workflows to $dest"
  echo "seeded $ROOT/input/example.png for the image loaders"
  echo "In Comfy Desktop: Workflows -> linkzag-minimax-h3 -> choose a workflow."
}

usage() {
  cat <<'EOF'
Install/check the pieces this project needs inside ComfyUI Desktop.

  bash scripts/setup_desktop.sh --check
  bash scripts/setup_desktop.sh --nodes
  bash scripts/setup_desktop.sh --workflows
  bash scripts/setup_desktop.sh --models core
  bash scripts/setup_desktop.sh --all

--all installs the ComfyUI-GGUF node pack, copies the workflows into the
Desktop workflow browser, and downloads the 32 GB core model set. It does not
download the optional R2V or SDXL models; use --models all for those.
EOF
}

ACTION="${1:---check}"
case "$ACTION" in
  --check|check) check ;;
  --nodes|nodes) install_nodes ;;
  --workflows|workflows) install_workflows ;;
  --models|models)
    PROFILE="${PROFILE:-32gb}" COMFY_ROOT="$ROOT" bash "$HERE/download_models.sh" "${2:-core}"
    ;;
  --all|all)
    install_nodes
    install_workflows
    PROFILE="${PROFILE:-32gb}" COMFY_ROOT="$ROOT" bash "$HERE/download_models.sh" core
    echo
    echo "Done. Quit and reopen Comfy Desktop, then run:"
    echo "  python3 scripts/render.py t2v \"a paper boat on a rain puddle\""
    ;;
  -h|--help|help) usage ;;
  *) echo "unknown option: $ACTION" >&2; usage >&2; exit 2 ;;
esac
