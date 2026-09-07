<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/LAKSA-Light.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/LAKSA-Dark.png">
    <img alt="LAKSA" src="docs/LAKSA-Dark.png" width=96 valign="middle">
  </picture>
  LAKSA
</h1>

## About the project

This repo contains **LAKSA**, the compiler infrastructure component of the [MYRTUS](https://myrtus-project.eu) project, which funds this work.
All analysis, optimization, and backend code generation for LAKSA targets flow through the dialects and passes defined here.

The compiler is built on top of [MLIR](https://mlir.llvm.org/), which allows each stage of the pipeline to be expressed as a dialect lowering, from high-level dataflow graph descriptions down to target-specific output.
Each dialect in this project represents one layer of that pipeline.

> The `DFG` dialect is a reimplementation of [Feliix42/dfg-mlir](https://github.com/Feliix42/dfg-mlir).

The paper describing this project has been accepted at [CASES'26](https://esweek.org/cases/).
The proceedings aren't out yet (a link to the paper will be added here later).

## Related projects

[`laksa-hls-kria-driver`](https://github.com/tud-ccc/laksa-hls-kria-dirver) is a companion Linux kernel driver for deploying LAKSA-generated HLS kernels on Xilinx Kria boards (tested on the KV260, ZynqMP).
It exposes `/dev/laksa`, a character device through which userspace programs manage the DMA buffer lifecycle and drive kernel execution over AXI Lite via `ioctl` calls, with no kernel recompilation needed when switching between HLS designs.

## Development Environment (Nix)

This repo ships a [Nix flake](flake.nix) that provides a complete dev shell with LLVM/MLIR, `cmake`, `ninja`, `mold`, `doxygen`, `gurobi`, and the Python toolchain already on `PATH`, with `LLVM_DIR`, `MLIR_DIR`, `LLVM_EXTERNAL_LIT`, `PYTHONPATH`, and `GUROBI_HOME` set automatically:

```bash
nix develop
```

If you use [direnv](https://direnv.net/), `direnv allow` will load the shell automatically via the included `.envrc`.
With the shell active, [`build.sh`](build.sh) detects the toolchain env vars and configures/builds directly, no arguments needed:

```bash
bash build.sh
```

## Build on Ubuntu 24.04 (no Nix)

Built and tested against [`llvm-22.1.7`](https://github.com/llvm/llvm-project/tree/llvmorg-22.1.7).
Outside the Nix shell, [`build.sh`](build.sh) also installs the system packages, downloads Gurobi, and sets up the Python venv:

1. Download `llvm-22.tar.zst` from this repo's [Releases](https://github.com/tud-ccc/laksa/releases) page based on your own architecture.
2. Run the script, pointing it at the downloaded file:

   ```bash
   bash build.sh /path/to/llvm-22.tar.zst
   ```

A Gurobi license is required either way.
Either export `WLSACCESSID`, `WLSSECRET`, and `LICENSEID` before running the script (it writes `$HOME/.gurobi/gurobi.lic` for you), or place a license file there yourself.

## Run the tests

The `.mlir` FileCheck test suite under [`test/`](test) is run with:

```bash
cmake --build build --target check-laksa-mlir
```

**Using the Python bindings**

See [`python/examples/`](python/examples) — it mirrors [`test/`](test)'s layout, with one `.py` file per `.mlir` test rebuilding the same IR via the Python bindings and running the same pass(es).
Start there for runnable references on constructing ops and invoking passes.

```bash
cmake --build build --target check-laksa-python
```

This target verifies every example still matches its `.mlir` counterpart, depending on `install-python-package` to rebuild and reinstall first.

## Build the documentation

The `Doxygen` and `Sphinx` documentation is found under `build/docs/sphinx/index.html`:
```bash
cmake --build build --target build-laksa-doc
```
