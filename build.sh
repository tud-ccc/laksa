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
    case "$(uname -m)" in
        x86_64)         LLVM_ASSET=llvm-23.1.2-amd64.tar.zst; GUROBI_PLATFORM=linux64 ;;
        aarch64|arm64)  LLVM_ASSET=llvm-23.1.2-arm64.tar.zst; GUROBI_PLATFORM=armlinux64 ;;
        *)
            echo "ERROR: unsupported architecture $(uname -m)." >&2
            exit 1
            ;;
    esac

    LLVM_TARBALL="${1:-${LLVM_TARBALL:-$BUILD_DIR/$LLVM_ASSET}}"
    LLVM_RELEASE_TAG="${LLVM_RELEASE_TAG:-llvm-mlir-py312-release-shared}"
    LLVM_RELEASE_URL="${LLVM_RELEASE_URL:-https://github.com/jhbi826c/llvm-build/releases/download/$LLVM_RELEASE_TAG}"

    GUROBI_VERSION="${GUROBI_VERSION:-12.0.3}"
    GUROBI_MAJOR_MINOR="${GUROBI_VERSION%.*}"
    GUROBI_HOME_NAME="gurobi${GUROBI_VERSION//./}"

    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    fi

    log "Installing system packages"
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        cmake ninja-build clang lld mold xz-utils zstd curl \
        python3-dev python3-venv portaudio19-dev build-essential doxygen

    if ! command -v uv >/dev/null 2>&1; then
        log "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    LLVM_ROOT="$BUILD_DIR/llvm-23"
    if [ -x "$LLVM_ROOT/build/bin/mlir-opt" ]; then
        log "LLVM/MLIR already extracted at $LLVM_ROOT"
    else
        if [ ! -f "$LLVM_TARBALL" ]; then
            log "Downloading $LLVM_ASSET from $LLVM_RELEASE_TAG"
            curl -fL --retry 3 --progress-bar \
                "$LLVM_RELEASE_URL/$LLVM_ASSET" -o "$LLVM_TARBALL.part"
            mv "$LLVM_TARBALL.part" "$LLVM_TARBALL"
        fi

        set +o pipefail
        LLVM_ROOT="$BUILD_DIR/$(tar --use-compress-program='zstd -d' -tf "$LLVM_TARBALL" | head -1 | cut -d/ -f1)"
        set -o pipefail

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
        uv pip install -p "$VENV_DIR/bin/python" \
            -r "$ROOT/python/requirements.txt" \
            -r "$ROOT/docs/requirements.txt"
    fi

    GUROBI_DIR="${GUROBI_DIR:-}"
    if [ -n "$GUROBI_DIR" ] && [ -d "$GUROBI_DIR" ]; then
        log "Using existing Gurobi install at $GUROBI_DIR"
    else
        GUROBI_DIR="$BUILD_DIR/$GUROBI_HOME_NAME/$GUROBI_PLATFORM"
        if [ -d "$GUROBI_DIR" ]; then
            log "Gurobi already installed at $GUROBI_DIR"
        else
            log "Downloading and installing Gurobi $GUROBI_VERSION ($GUROBI_PLATFORM)"
            curl -L "https://packages.gurobi.com/${GUROBI_MAJOR_MINOR}/gurobi${GUROBI_VERSION}_${GUROBI_PLATFORM}.tar.gz" \
                -o /tmp/gurobi.tar.gz
            tar xzf /tmp/gurobi.tar.gz -C "$BUILD_DIR"
            rm /tmp/gurobi.tar.gz
        fi
        if [ ! -f "$GUROBI_DIR/include/gurobi_c.h" ]; then
            echo "ERROR: expected Gurobi headers at $GUROBI_DIR/include." >&2
            exit 1
        fi
    fi
fi

GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/.gurobi/gurobi.lic}"
if [ ! -f "$GRB_LICENSE_FILE" ]; then
    echo
    echo "WARNING: no Gurobi license found at $GRB_LICENSE_FILE." >&2
    echo "The build will still succeed, but anything using Gurobi will fail at runtime." >&2
    echo
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
