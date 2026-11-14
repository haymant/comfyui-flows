#!/bin/bash
# Open Comfy Desktop and wait for its local API.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_comfy_root.sh"

SERVER="$(comfy_server)"
if curl -sf -m 2 "$SERVER/system_stats" >/dev/null 2>&1; then
  echo "ComfyUI already running at $SERVER"
  exit 0
fi

echo "opening Comfy Desktop"
open -a "Comfy Desktop" 2>/dev/null || open -a ComfyUI 2>/dev/null || {
  echo "Could not open Comfy Desktop. Open it from /Applications, then retry." >&2
  exit 1
}

printf 'waiting for %s' "$SERVER"
for _ in $(seq 1 120); do
  if curl -sf -m 2 "$SERVER/system_stats" >/dev/null 2>&1; then
    echo
    echo "ComfyUI ready at $SERVER"
    exit 0
  fi
  printf '.'
  sleep 2
done
echo
echo "ComfyUI did not answer on $SERVER. Check the Desktop window or set COMFYUI_SERVER." >&2
exit 1
