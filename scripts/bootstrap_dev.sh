#!/usr/bin/env sh
# Create a local virtual environment and install ShadowLink in editable mode.
# Usage: scripts/bootstrap_dev.sh [--with-hamer]
set -eu

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3.10 not found. Set PYTHON=/path/to/python3.10 and retry." >&2
    exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

if [ ! -d third_party/hamer/hamer ]; then
    echo "warning: third_party/hamer is not checked out." >&2
    echo "         run: git submodule update --init" >&2
fi

"$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

if [ "${1:-}" = "--with-hamer" ]; then
    pip install -e ".[dev,hamer]"
    if [ -d third_party/hamer/hamer ]; then
        pip install -e third_party/hamer --no-deps
    else
        echo "Install the submodule first: git submodule update --init" >&2
        exit 1
    fi
else
    pip install -e ".[dev]"
fi

cat <<'EOF'

Installed. Next steps:
  source .venv/bin/activate
  shadowlink setup-models
  # register at https://mano.is.tue.mpg.de and place MANO_RIGHT.pkl
  shadowlink doctor --strict-models
EOF
