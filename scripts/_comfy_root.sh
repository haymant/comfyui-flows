#!/bin/bash
# Shared helpers for the comfy/ project. Source this file; do not execute it.
#
# It finds the ComfyUI Desktop installation on macOS without depending on any
# other file in this repository. Override with COMFY_ROOT if needed.

set -u

_comfy_from_proc() {
  local proc_dir="$HOME/Library/Application Support/Comfy Desktop/comfy-procs"
  local f install
  for f in "$proc_dir"/*.json; do
    [ -f "$f" ] || continue
    install="$(python3 - "$f" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("installPath", ""))
except Exception:
    print("")
PY
)"
    if [ -n "$install" ] && [ -f "$install/ComfyUI/main.py" ]; then
      printf '%s\n' "$install/ComfyUI"
      return 0
    fi
  done
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
  for candidate in \
    "$HOME/ComfyUI-Installs/ComfyUI-01/ComfyUI" \
    "$HOME/Documents/ComfyUI" \
    "$HOME/ComfyUI"; do
    if [ -f "$candidate/main.py" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

comfy_server() {
  if [ -n "${COMFYUI_SERVER:-}" ]; then
    printf '%s\n' "$COMFYUI_SERVER"
    return 0
  fi
  local proc_dir="$HOME/Library/Application Support/Comfy Desktop/comfy-procs"
  local f port
  for f in "$proc_dir"/*.json; do
    [ -f "$f" ] || continue
    port="$(python3 - "$f" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("port", ""))
except Exception:
    print("")
PY
)"
    if [ -n "$port" ]; then
      printf 'http://127.0.0.1:%s\n' "$port"
      return 0
    fi
  done
  printf 'http://127.0.0.1:8188\n'
}

comfy_python() {
  local root="$1"
  if [ -x "$root/.venv/bin/python" ]; then
    printf '%s\n' "$root/.venv/bin/python"
  elif [ -x "$root/venv/bin/python" ]; then
    printf '%s\n' "$root/venv/bin/python"
  else
    printf 'python3\n'
  fi
}
