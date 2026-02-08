#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_mujoco200_wsl.sh --archive /path/to/mujoco200_linux.zip|.tar.gz --mjkey /path/to/mjkey.txt

What it does:
  - Extracts MuJoCo 2.0 (mujoco200) into ~/.mujoco/mujoco200_linux
  - Copies mjkey.txt into ~/.mujoco/mjkey.txt (chmod 600)
  - Appends recommended env vars to ~/.bashrc (idempotent)

Notes:
  - You must download the MuJoCo archive + mjkey.txt yourself due to licensing.
  - After running, open a new shell (or source ~/.bashrc) before installing mujoco-py.
EOF
}

ARCHIVE=""
MJKEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      ARCHIVE="${2:-}"; shift 2 ;;
    --mjkey)
      MJKEY="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$ARCHIVE" || -z "$MJKEY" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 2
fi

if [[ ! -f "$MJKEY" ]]; then
  echo "mjkey.txt not found: $MJKEY" >&2
  exit 2
fi

MUJOCO_HOME="$HOME/.mujoco"
TARGET_LINUX="$MUJOCO_HOME/mujoco200_linux"
TARGET_COMPAT="$MUJOCO_HOME/mujoco200"
TMPDIR="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

mkdir -p "$MUJOCO_HOME"

if [[ -d "$TARGET_LINUX" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup="${TARGET_LINUX}.bak_${ts}"
  echo "Existing $TARGET_LINUX found; moving to $backup" >&2
  mv "$TARGET_LINUX" "$backup"
fi

# Extract
case "$ARCHIVE" in
  *.zip)
    command -v unzip >/dev/null 2>&1 || {
      echo "unzip not found. Install it with: sudo apt-get install -y unzip" >&2
      exit 2
    }
    unzip -q "$ARCHIVE" -d "$TMPDIR"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE" -C "$TMPDIR"
    ;;
  *)
    echo "Unsupported archive type (expected .zip/.tar.gz/.tgz): $ARCHIVE" >&2
    exit 2
    ;;
 esac

# Find extracted mujoco root by locating bin/libmujoco200.so
libpath="$(find "$TMPDIR" -maxdepth 5 -type f -name 'libmujoco200.so' -print -quit || true)"
if [[ -z "$libpath" ]]; then
  echo "Could not find libmujoco200.so in extracted archive." >&2
  echo "Extracted top-level entries:" >&2
  find "$TMPDIR" -maxdepth 2 -mindepth 1 -print >&2
  exit 2
fi

rootdir="$(dirname "$(dirname "$libpath")")"  # .../bin/libmujoco200.so -> root

# Move into place
mv "$rootdir" "$TARGET_LINUX"

# mujoco-py expects ~/.mujoco/mujoco200
ln -sfn "$TARGET_LINUX" "$TARGET_COMPAT"

# Install key
cp "$MJKEY" "$MUJOCO_HOME/mjkey.txt"
chmod 600 "$MUJOCO_HOME/mjkey.txt"

# Basic sanity
if [[ ! -f "$TARGET_LINUX/bin/libmujoco200.so" ]]; then
  echo "Sanity check failed: $TARGET_LINUX/bin/libmujoco200.so missing" >&2
  exit 2
fi

BASHRC="$HOME/.bashrc"
mkdir -p "$(dirname "$BASHRC")"
touch "$BASHRC"

ensure_line() {
  local line="$1"
  grep -Fqx "$line" "$BASHRC" 2>/dev/null || echo "$line" >> "$BASHRC"
}

ensure_line ''
ensure_line '# Added by cee-us/scripts/setup_mujoco200_wsl.sh'
ensure_line 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"'
ensure_line 'export MUJOCO_GL=osmesa'

cat <<EOF
Done.
- MuJoCo installed to: $TARGET_LINUX
- Compatibility symlink: $TARGET_COMPAT -> $TARGET_LINUX
- License key installed to: $MUJOCO_HOME/mjkey.txt

Next:
1) Run: source ~/.bashrc
2) In repo venv, install mujoco-py:
   source /home/zhuzihou/dev/cee-us/.venv/bin/activate
   export PATH=\"$HOME/.local/bin:\$PATH\"
   uv pip install --no-progress mujoco-py==2.0.2.0
EOF
