#!/bin/bash
# Open ComfyUI Desktop and wait for its local API.
#
# macOS: launches the "Comfy Desktop" app bundle.
# Linux: tries the packaged executable (.deb installs it under /opt), then the
#        desktop-entry launcher, then $COMFY_DESKTOP_CMD. AppImages are not
#        started automatically because they may need --no-sandbox.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/_comfy_root.sh"

SERVER="$(comfy_server)"
if curl -sf -m 2 "$SERVER/system_stats" >/dev/null 2>&1; then
  echo "ComfyUI already running at $SERVER"
  exit 0
fi

linux_launch() {
  if [ -n "${COMFY_DESKTOP_CMD:-}" ]; then
    nohup "$COMFY_DESKTOP_CMD" >/dev/null 2>&1 &
    return 0
  fi
  local cmd
  for cmd in comfyui-desktop comfy-desktop comfyui-desktop-2 "Comfy Desktop"; do
    if command -v "$cmd" >/dev/null 2>&1; then
      nohup "$cmd" >/dev/null 2>&1 &
      return 0
    fi
  done
  local exe
  for exe in \
    "/opt/Comfy Desktop/comfyui-desktop" \
    "/opt/Comfy Desktop/comfy-desktop" \
    "/opt/Comfy Desktop/comfyui-desktop-2" \
    "/opt/Comfy Desktop/Comfy Desktop"; do
    if [ -x "$exe" ]; then
      nohup "$exe" >/dev/null 2>&1 &
      return 0
    fi
  done
  local entry
  for entry in \
    /usr/share/applications/*[Cc]omfy*.desktop \
    "$HOME/.local/share/applications/"*[Cc]omfy*.desktop; do
    if [ -f "$entry" ] && command -v gio >/dev/null 2>&1; then
      gio launch "$entry" >/dev/null 2>&1 &
      return 0
    fi
  done
  return 1
}

case "$(comfy_platform)" in
  macos)
    echo "opening Comfy Desktop"
    open -a "Comfy Desktop" 2>/dev/null || open -a ComfyUI 2>/dev/null || {
      echo "Could not open Comfy Desktop. Open it from /Applications, then retry." >&2
      exit 1
    }
    ;;
  linux)
    echo "opening Comfy Desktop"
    linux_launch || {
      cat >&2 <<'EOF'
Could not find the Comfy Desktop launcher.

Open it from your application menu (or run the AppImage directly; add
--no-sandbox if the Electron sandbox blocks startup). To point this script at
a specific binary:
  COMFY_DESKTOP_CMD="/path/to/comfyui-desktop" bash scripts/open_comfyui.sh
EOF
      exit 1
    }
    ;;
  *)
    echo "Unsupported platform: $(uname -s). Open ComfyUI manually." >&2
    exit 1
    ;;
esac

# The port is written to comfy-procs/ when the instance starts, so re-read it
# while waiting instead of trusting the value from before launch.
printf 'waiting for ComfyUI'
for _ in $(seq 1 180); do
  SERVER="$(comfy_server)"
  if curl -sf -m 2 "$SERVER/system_stats" >/dev/null 2>&1; then
    echo
    echo "ComfyUI ready at $SERVER  ($(comfy_device) backend)"
    exit 0
  fi
  printf '.'
  sleep 2
done
echo
echo "ComfyUI did not answer on $SERVER. Check the Desktop window or set COMFYUI_SERVER." >&2
exit 1
