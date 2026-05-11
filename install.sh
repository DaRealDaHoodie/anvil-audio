#!/usr/bin/env bash
# install.sh — Anvil Audio self-contained installer for macOS and Linux
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

# ── Platform detection ────────────────────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)

echo "=== Anvil Audio Installer ==="
echo "Platform: $OS / $ARCH"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" < "3.12" ]]; then
    echo "ERROR: Python 3.12 or 3.13 required (found $PY_VER)."
    echo "Install via: brew install python@3.13  (macOS)"
    echo "             sudo apt install python3.13  (Linux)"
    exit 1
fi
echo "Python $PY_VER — OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
echo "Using venv: $VENV_DIR"

$PIP install --upgrade pip --quiet

# ── PyTorch (platform-specific) ───────────────────────────────────────────────
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Installing PyTorch for macOS Apple Silicon..."
    $PIP install torch torchaudio --quiet

elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    echo "Installing PyTorch for Linux CUDA..."
    $PIP install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 --quiet

else
    echo "Installing PyTorch (CPU fallback)..."
    $PIP install torch torchaudio --quiet
fi

# ── Anvil Audio ───────────────────────────────────────────────────────────────
echo "Installing anvil-audio..."
$PIP install -e . --quiet

# ── MLX acceleration (Apple Silicon only) ────────────────────────────────────
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Installing MLX acceleration (mlx-audiogen)..."
    $PIP install mlx-audiogen --quiet
fi

# ── ACE-Step (optional music generation) ─────────────────────────────────────
read -r -p "Install ACE-Step for music generation? [y/N] " INSTALL_AS
if [[ "$INSTALL_AS" =~ ^[Yy]$ ]]; then

    if [[ "$OS" == "Linux" ]]; then
        echo "Installing nano-vllm (required on Linux)..."
        $PIP install \
            "git+https://github.com/ace-step/ACE-Step.git#subdirectory=acestep/third_parts/nano-vllm" \
            --quiet
    fi

    echo "Installing ACE-Step..."
    $PIP install \
        "ace-step @ git+https://github.com/ace-step/ACE-Step.git" \
        --quiet
    echo "ACE-Step installed."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation complete ==="
echo ""
echo "Activate your environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Check your setup:"
echo "  anvil setup"
echo ""
echo "Generate audio:"
echo "  anvil generate --model stable-audio-open-1.0-mlx --prompt \"rain on a tin roof\""
