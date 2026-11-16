#!/bin/bash
# Shared helpers for the comfy/ project. Source this file; do not execute it.
#
# Finds ComfyUI Desktop on macOS or Linux, detects the compute backend, and
# locates the models directory. Every helper can be overridden with an
# environment variable, so manual ComfyUI installs and remote servers work too:
#
#   COMFY_ROOT            path to the ComfyUI directory (the one with main.py)
#   COMFY_MODELS          path to models/ (default: $COMFY_ROOT/models)
#   COMFYUI_SERVER        http://host:port of the running server
#   COMFY_DEVICE          mps | cuda | rocm | cpu (default: auto-detected)
#   COMFY_DESKTOP_DATA    app-data dir that holds comfy-procs/ (default: below)
#   COMFY_DESKTOP_CMD     path to the Desktop executable, used by open_comfyui.sh

set -u

comfy_platform() {
  case "$(uname -s)" in
    Darwin) printf 'macos\n' ;;
    Linux) printf 'linux\n' ;;
    *) printf 'unknown\n' ;;
  esac
}

# Application-data directories that may contain comfy-procs/*.json.
# Comfy Desktop follows the XDG spec on Linux and Application Support on macOS.
# The dev build is called comfyui-desktop-2; the packaged app "Comfy Desktop".
_comfy_data_dirs() {
  local name
  local cfg="${XDG_CONFIG_HOME:-$HOME/.config}"
  local state="${XDG_STATE_HOME:-$HOME/.local/state}"
  local data="${XDG_DATA_HOME:-$HOME/.local/share}"
  if [ -n "${COMFY_DESKTOP_DATA:-}" ]; then
    printf '%s\n' "$COMFY_DESKTOP_DATA"
  fi
  case "$(comfy_platform)" in
    macos)
      for name in "Comfy Desktop" "comfyui-desktop-2"; do
        printf '%s\n' "$HOME/Library/Application Support/$name"
      done
      ;;
    *)
      for name in "Comfy Desktop" "comfyui-desktop-2"; do
        printf '%s\n' "$cfg/$name" "$state/$name" "$data/$name"
      done
      ;;
  esac
}

_comfy_proc_field() {
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1])).get(sys.argv[2], "")
    print(value if value is not None else "")
except Exception:
    print("")
PY
}

_comfy_from_proc() {
  local dir f install
  while IFS= read -r dir; do
    [ -d "$dir/comfy-procs" ] || continue
    for f in "$dir"/comfy-procs/*.json; do
      [ -f "$f" ] || continue
      install="$(_comfy_proc_field "$f" installPath)"
      if [ -n "$install" ] && [ -f "$install/ComfyUI/main.py" ]; then
        printf '%s\n' "$install/ComfyUI"
        return 0
      fi
      # Some builds record the ComfyUI directory itself instead of the instance.
      if [ -n "$install" ] && [ -f "$install/main.py" ]; then
        printf '%s\n' "$install"
        return 0
      fi
    done
  done < <(_comfy_data_dirs)
  return 1
}

comfy_root() {
  if [ -n "${COMFY_ROOT:-}" ]; then
    printf '%s\n' "$COMFY_ROOT"
    return 0
  fi
  if _comfy_from_proc; then
    return 0
  fi
  local candidate
  # Every Comfy Desktop standalone instance lives under ComfyUI-Installs/.
  for candidate in \
    "$HOME"/ComfyUI-Installs/*/ComfyUI \
    "$HOME"/ComfyUI-Installs/* \
    "$HOME/ComfyUI-Installs/ComfyUI-01/ComfyUI" \
    "$HOME/Documents/ComfyUI" \
    "$HOME/ComfyUI" \
    "$HOME/.local/share/ComfyUI" \
    "/opt/ComfyUI"; do
    if [ -f "$candidate/main.py" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

comfy_models() {
  local root="${1:-}"
  if [ -n "${COMFY_MODELS:-}" ]; then
    printf '%s\n' "$COMFY_MODELS"
    return 0
  fi
  if [ -z "$root" ]; then
    root="$(comfy_root || true)"
  fi
  # Comfy Desktop can share models between instances under ~/ComfyUI-Shared.
  if [ -n "$root" ] && [ -d "$root/models" ]; then
    printf '%s\n' "$root/models"
    return 0
  fi
  if [ -d "$HOME/ComfyUI-Shared" ]; then
    printf '%s\n' "$HOME/ComfyUI-Shared"
    return 0
  fi
  if [ -n "$root" ]; then
    printf '%s\n' "$root/models"
    return 0
  fi
  return 1
}

comfy_server() {
  if [ -n "${COMFYUI_SERVER:-}" ]; then
    printf '%s\n' "$COMFYUI_SERVER"
    return 0
  fi
  local dir f port
  while IFS= read -r dir; do
    [ -d "$dir/comfy-procs" ] || continue
    for f in "$dir"/comfy-procs/*.json; do
      [ -f "$f" ] || continue
      port="$(_comfy_proc_field "$f" port)"
      if [ -n "$port" ]; then
        printf 'http://127.0.0.1:%s\n' "$port"
        return 0
      fi
    done
  done < <(_comfy_data_dirs)
  printf 'http://127.0.0.1:8188\n'
}

# Best-effort compute backend. The running server is the source of truth for
# scripts/render.py; this is for setup/launch/help output.
comfy_device() {
  if [ -n "${COMFY_DEVICE:-}" ]; then
    printf '%s\n' "$COMFY_DEVICE"
    return 0
  fi
  case "$(comfy_platform)" in
    macos)
      printf 'mps\n'
      return 0
      ;;
  esac
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf 'cuda\n'
    return 0
  fi
  if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
    printf 'rocm\n'
    return 0
  fi
  printf 'cpu\n'
}

# Extra server arguments that keep the 8.9 GB diffusion model and the 8-15 GB
# text encoder from being resident at the same time. The flags are per backend:
# --cpu is required for a CPU-only install, and the MPS cross-attention flag is
# not needed or wanted on CUDA.
comfy_startup_args() {
  case "$(comfy_device)" in
    cuda) printf '%s\n' "--cache-none --disable-smart-memory" ;;
    rocm|mps) printf '%s\n' "--use-pytorch-cross-attention --cache-none --disable-smart-memory" ;;
    cpu) printf '%s\n' "--cpu --cache-none --disable-smart-memory" ;;
    *) printf '%s\n' "--cache-none --disable-smart-memory" ;;
  esac
}

comfy_python() {
  local root="$1"
  if [ -x "$root/.venv/bin/python" ]; then
    printf '%s\n' "$root/.venv/bin/python"
  elif [ -x "$root/.venv/bin/python3" ]; then
    printf '%s\n' "$root/.venv/bin/python3"
  elif [ -x "$root/venv/bin/python" ]; then
    printf '%s\n' "$root/venv/bin/python"
  elif [ -x "$root/venv/bin/python3" ]; then
    printf '%s\n' "$root/venv/bin/python3"
  else
    printf 'python3\n'
  fi
}
