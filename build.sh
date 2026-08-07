#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
LINKER="${LINKER:-mold}"
JOBS="${JOBS:-$(nproc)}"

log() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }

mkdir -p "$BUILD_DIR"

if [ -n "${MLIR_DIR:-}" ] && [ -d "$MLIR_DIR" ]; then
    log "Detected provisioned toolchain (MLIR_DIR=$MLIR_DIR), skipping system setup"
    GUROBI_DIR="${GUROBI_DIR:-${GUROBI_HOME:-}}"
    if [ -z "$GUROBI_DIR" ] || [ ! -d "$GUROBI_DIR" ]; then
        echo "ERROR: MLIR_DIR is set but GUROBI_DIR/GUROBI_HOME is not a valid directory." >&2
        exit 1
    fi
    if [ ! -d "$VENV_DIR" ]; then
        echo "ERROR: expected a Python venv at $VENV_DIR." >&2
        exit 1
    fi
else
    LLVM_TARBALL="${1:-${LLVM_TARBALL:-}}"
    if [ -z "$LLVM_TARBALL" ] || [ ! -f "$LLVM_TARBALL" ]; then
        echo "Usage: $0 <path-to-llvm-22.tar.zst>" >&2
        echo "Download the prebuilt LLVM/MLIR tarball from this repo's GitHub Releases first." >&2
        echo "(Skip the argument entirely when running inside 'nix develop'.)" >&2
        exit 1
    fi

    GUROBI_VERSION="${GUROBI_VERSION:-12.0.3}"
    GUROBI_MAJOR_MINOR="${GUROBI_VERSION%.*}"
    GUROBI_HOME_NAME="gurobi$(tr -d '.' <<<"$GUROBI_MAJOR_MINOR")0"

    PY_DEPS=(
        'nanobind>=2.9,<3.0'
        'pybind11>=2.10'
        'PyYAML>=5.4.0,<=6.0.1'
        'typing_extensions>=4.12.2'
        'numpy>=2.1.0,<=2.1.2'
        'ml_dtypes>=0.5.0,<=0.6.0'
    )

    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    fi

    log "Installing system packages"
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        cmake ninja-build clang clang++ lld mold xz-utils zstd curl \
        python3-dev python3-venv portaudio19-dev build-essential doxygen

    if ! command -v uv >/dev/null 2>&1; then
        log "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    LLVM_ROOT="$BUILD_DIR/$(tar --use-compress-program='zstd -d' -tf "$LLVM_TARBALL" | head -1 | cut -d/ -f1)"
    if [ -x "$LLVM_ROOT/build/bin/mlir-opt" ]; then
        log "LLVM/MLIR already extracted at $LLVM_ROOT"
    else
        log "Extracting LLVM/MLIR from $LLVM_TARBALL"
        tar --use-compress-program='zstd -d' -xf "$LLVM_TARBALL" -C "$BUILD_DIR"
        if [ ! -x "$LLVM_ROOT/build/bin/mlir-opt" ]; then
            echo "ERROR: extracted $LLVM_TARBALL but $LLVM_ROOT/build/bin/mlir-opt is missing." >&2
            exit 1
        fi
    fi
    MLIR_DIR="$LLVM_ROOT/build/lib/cmake/mlir"
    LLVM_DIR="$LLVM_ROOT/build/lib/cmake/llvm"
    LLVM_EXTERNAL_LIT="$LLVM_ROOT/build/bin/llvm-lit"

    if [ ! -d "$VENV_DIR" ]; then
        log "Setting up Python venv at $VENV_DIR"
        uv venv "$VENV_DIR" -p 3.12
        uv pip install "${PY_DEPS[@]}" -p "$VENV_DIR/bin/python"
    fi

    GUROBI_DIR="${GUROBI_DIR:-}"
    if [ -n "$GUROBI_DIR" ] && [ -d "$GUROBI_DIR" ]; then
        log "Using existing Gurobi install at $GUROBI_DIR"
    else
        GUROBI_DIR="$BUILD_DIR/$GUROBI_HOME_NAME/linux64"
        if [ -d "$GUROBI_DIR" ]; then
            log "Gurobi already installed at $GUROBI_DIR"
        else
            log "Downloading and installing Gurobi $GUROBI_VERSION"
            curl -L "https://packages.gurobi.com/${GUROBI_MAJOR_MINOR}/gurobi${GUROBI_VERSION}_linux64.tar.gz" \
                -o /tmp/gurobi.tar.gz
            tar xzf /tmp/gurobi.tar.gz -C "$BUILD_DIR"
            rm /tmp/gurobi.tar.gz
        fi
    fi
fi

if [ ! -f "$HOME/.gurobi/gurobi.lic" ]; then
    if [ -n "${WLSACCESSID:-}" ] && [ -n "${WLSSECRET:-}" ] && [ -n "${LICENSEID:-}" ]; then
        log "Writing Gurobi WLS license to \$HOME/.gurobi/gurobi.lic"
        mkdir -p "$HOME/.gurobi"
        cat >"$HOME/.gurobi/gurobi.lic" <<EOF
WLSACCESSID=${WLSACCESSID}
WLSSECRET=${WLSSECRET}
LICENSEID=${LICENSEID}
EOF
    else
        echo
        echo "WARNING: no Gurobi license found at \$HOME/.gurobi/gurobi.lic, and" >&2
        echo "WLSACCESSID/WLSSECRET/LICENSEID are not set to generate one." >&2
        echo "The build will still succeed, but anything using Gurobi will fail at runtime." >&2
        echo
    fi
fi

log "Configuring $(basename "$ROOT")"
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_STANDARD=20 \
    -DLLVM_USE_LINKER="$LINKER" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT" \
    -DGUROBI_DIR="$GUROBI_DIR" \
    -DPython_EXECUTABLE="$VENV_DIR/bin/python" \
    -DPython3_EXECUTABLE="$VENV_DIR/bin/python3" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_PACKAGE_PREFIX=mlir_laksa \
    -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX=mlir_laksa \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

log "Building $(basename "$ROOT")"
cmake --build "$BUILD_DIR" -j "$JOBS"

log "Installing the Python package (editable install into $VENV_DIR)"
cmake --build "$BUILD_DIR" --target install-python-package

cat <<EOF

==> Done.

  MLIR_DIR   = $MLIR_DIR
  LLVM_DIR   = $LLVM_DIR
  GUROBI_DIR = $GUROBI_DIR
  VENV_DIR   = $VENV_DIR

Activate the venv with:
  source "$VENV_DIR/bin/activate"

Run the test suite with:
  cmake --build build --target check-laksa-mlir
  cmake --build build --target check-laksa-python

Build the docs with:
  cmake --build build --target build-laksa-doc
EOF
