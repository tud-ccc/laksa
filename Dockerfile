# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04

################################################################################
# Stage 1: builder
################################################################################
FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG TARGETARCH
ARG LLVM_RELEASE=https://github.com/tud-ccc/laksa/releases/download/llvm-mlir-22.1.7-py312-release-shared
ARG LLVM_ASSET_AMD64=llvm-22-amd64.tar.zst
ARG LLVM_ASSET_ARM64=llvm-22-arm64.tar.zst
ARG GUROBI_VERSION=12.0.3
ARG BUILD_TYPE=Release

ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_ROOT=/opt/llvm-22 \
    GUROBI_DIR=/opt/gurobi \
    VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        lld \
        mold \
        ninja-build \
        patchelf \
        portaudio19-dev \
        python3-dev \
        python3-venv \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# --- Prebuilt LLVM/MLIR ------------------------------------------------------
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) llvm_asset="${LLVM_ASSET_AMD64}" ;; \
        arm64) llvm_asset="${LLVM_ASSET_ARM64}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fL --retry 3 "${LLVM_RELEASE}/${llvm_asset}" -o /tmp/llvm.tar.zst; \
    tar --use-compress-program='zstd -d' -xf /tmp/llvm.tar.zst -C /opt; \
    rm /tmp/llvm.tar.zst; \
    test -x "${LLVM_ROOT}/build/bin/mlir-opt"

# --- Gurobi ------------------------------------------------------------------
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) grb_plat=linux64 ;; \
        arm64) grb_plat=armlinux64 ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    grb_mm="${GUROBI_VERSION%.*}"; \
    grb_dir="/opt/gurobi$(printf '%s' "${GUROBI_VERSION}" | tr -d '.')"; \
    curl -fL --retry 3 \
        "https://packages.gurobi.com/${grb_mm}/gurobi${GUROBI_VERSION}_${grb_plat}.tar.gz" \
        -o /tmp/gurobi.tar.gz; \
    tar xzf /tmp/gurobi.tar.gz -C /opt; \
    rm /tmp/gurobi.tar.gz; \
    mv "${grb_dir}/${grb_plat}" "${GUROBI_DIR}"; \
    rm -rf "${grb_dir}"; \
    test -f "${GUROBI_DIR}/include/gurobi_c.h"

RUN set -eux; \
    echo "${LLVM_ROOT}/build/lib" > /etc/ld.so.conf.d/llvm-22.conf; \
    echo "${GUROBI_DIR}/lib"      > /etc/ld.so.conf.d/gurobi.conf; \
    ldconfig

# --- Python build environment ------------------------------------------------
# Copied on its own so this layer is only invalidated when the pinned versions
# change, not on every source edit.
COPY python/requirements.txt /src/laksa/python/requirements.txt

RUN set -eux; \
    python3 -m venv "${VIRTUAL_ENV}"; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir -r /src/laksa/python/requirements.txt

# --- Source ------------------------------------------------------------------
# The build context is this repository; see .dockerignore for what is excluded.
COPY . /src/laksa

# --- Configure and build -----------------------------------------------------
RUN set -eux; \
    cd /src/laksa; \
    nanobind_dir="$(python3 -c 'import nanobind; print(nanobind.cmake_dir())')"; \
    cmake -S . -B build -G Ninja \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_RPATH="${LLVM_ROOT}/build/lib:${GUROBI_DIR}/lib" \
        -DLLVM_USE_LINKER=mold \
        -DMLIR_DIR="${LLVM_ROOT}/build/lib/cmake/mlir" \
        -DLLVM_DIR="${LLVM_ROOT}/build/lib/cmake/llvm" \
        -DLLVM_EXTERNAL_LIT="${LLVM_ROOT}/build/bin/llvm-lit" \
        -DGUROBI_DIR="${GUROBI_DIR}" \
        -Dnanobind_DIR="${nanobind_dir}" \
        -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
        -DPython3_EXECUTABLE="${VIRTUAL_ENV}/bin/python3" \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DMLIR_PYTHON_PACKAGE_PREFIX=mlir_laksa \
        -DMLIR_BINDINGS_PYTHON_INSTALL_PREFIX=mlir_laksa \
        -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir_laksa; \
    cmake --build build

# --- Stage the runtime tree --------------------------------------------------
RUN set -eux; \
    py_site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    test "${py_site}" = "${VIRTUAL_ENV}/lib/python3.12/site-packages"; \
    mkdir -p "/rootfs${LLVM_ROOT}/build/lib" \
             "/rootfs${GUROBI_DIR}/lib" \
             /rootfs/opt/laksa; \
    \
    cd /src/laksa; \
    DESTDIR=/rootfs cmake --install build --component LAKSATools; \
    DESTDIR=/rootfs cmake --install build --component LAKSAPythonModules \
        --prefix "${py_site}"; \
    test -f "/rootfs${py_site}/mlir_laksa/ir.py"; \
    \
    cp -a "${LLVM_ROOT}"/build/lib/*.so* "/rootfs${LLVM_ROOT}/build/lib/"; \
    cp -a "${GUROBI_DIR}"/lib/libgurobi*.so* "/rootfs${GUROBI_DIR}/lib/"; \
    \
    cp /src/laksa/LICENSE /rootfs/opt/laksa/LICENSE; \
    \
    find "/rootfs${py_site}/mlir_laksa" -name '*.so' -print0 \
        | xargs -0 -r -n1 patchelf --set-rpath \
            "\$ORIGIN:\$ORIGIN/_mlir_libs:${LLVM_ROOT}/build/lib:${GUROBI_DIR}/lib"; \
    \
    find /rootfs/usr/local/bin -type f -exec strip --strip-unneeded {} + || true; \
    find "/rootfs${LLVM_ROOT}" "/rootfs${GUROBI_DIR}" "/rootfs${py_site}" \
        -name '*.so*' -type f -exec strip --strip-unneeded {} + || true; \
    du -sh /rootfs

################################################################################
# Stage 2: runtime
################################################################################
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LAKSA" \
      org.opencontainers.image.description="LAKSA MLIR compiler (ladle, laksa-opt, laksa-translate, laksa-lsp-server) with the mlir_laksa Python bindings" \
      org.opencontainers.image.source="https://github.com/tud-ccc/laksa" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_ROOT=/opt/llvm-22 \
    GUROBI_HOME=/opt/gurobi \
    GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
    VIRTUAL_ENV=/opt/venv \
    SHELL=/bin/bash
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        graphviz \
        libedit2 \
        libgomp1 \
        libncurses6 \
        libstdc++6 \
        libtinfo6 \
        libxml2 \
        libzstd1 \
        python3 \
        python3-venv \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# --- Python runtime environment ----------------------------------------------
COPY python/requirements.txt /tmp/requirements.txt

RUN set -eux; \
    python3 -m venv "${VIRTUAL_ENV}"; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    rm /tmp/requirements.txt; \
    pip install --no-cache-dir \
        matplotlib \
        pandas \
        scipy \
        sympy \
        networkx \
        graphviz \
        pillow \
        tabulate \
        tqdm \
        rich \
        pytest \
        lit \
        filecheck; \
    python3 -c 'import numpy; assert numpy.__version__ <= "2.1.2", numpy.__version__'

COPY --from=builder /rootfs/ /

RUN set -eux; \
    echo "${LLVM_ROOT}/build/lib" > /etc/ld.so.conf.d/llvm-22.conf; \
    echo "${GUROBI_HOME}/lib"     > /etc/ld.so.conf.d/gurobi.conf; \
    ldconfig

# --- Expose the venv packages to the system interpreter too --------------------
RUN set -eux; \
    sys_site="$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    mkdir -p "${sys_site}"; \
    echo "${VIRTUAL_ENV}/lib/python3.12/site-packages" > "${sys_site}/mlir_laksa.pth"; \
    /usr/bin/python3 -c 'import mlir_laksa.ir'

# --- Smoke test ---------------------------------------------------------------
RUN set -eux; \
    missing=0; \
    for f in /usr/local/bin/ladle /usr/local/bin/laksa-opt \
             /usr/local/bin/laksa-translate /usr/local/bin/laksa-lsp-server \
             $(find "${VIRTUAL_ENV}"/lib/python3.12/site-packages/mlir_laksa -name '*.so'); do \
        if ldd "$f" 2>&1 | grep -q 'not found'; then \
            echo "unresolved shared libraries in $f:" >&2; \
            ldd "$f" | grep 'not found' >&2; \
            missing=1; \
        fi; \
    done; \
    test "${missing}" -eq 0; \
    laksa-opt --version; \
    laksa-translate --help > /dev/null; \
    python3 -c \
        "from mlir_laksa.ir import Context; from mlir_laksa.dialects import dfg, emithls; print('mlir_laksa import OK')"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
