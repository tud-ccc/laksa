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

This repo ships a [Nix flake](flake.nix) that provides a complete dev shell with LLVM/MLIR, `cmake`, `ninja`, `mold`, `doxygen`, `gurobi`, and the Python toolchain already on `PATH`, with `LLVM_DIR`, `MLIR_DIR`, `LLVM_EXTERNAL_LIT`, `PYTHONPATH`, `GUROBI_HOME`, and `GRB_LICENSE_FILE` set automatically:

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
Both `x86_64` and `aarch64` are supported; [`build.sh`](build.sh) reads `uname -m` and picks the matching Gurobi build (`linux64` / `armlinux64`) itself.

Outside the Nix shell, [`build.sh`](build.sh) installs the system packages, fetches LLVM/MLIR and Gurobi, and sets up the Python venv, so it needs no arguments here either:

```bash
bash build.sh
```

The prebuilt LLVM/MLIR tarball is about 700 MB and lands in `build/`, so the first run takes a while.

## Gurobi license

A license is required however you build, and nothing here creates one for you.
`build.sh` only checks that the license file exists and warns if it does not; the build still succeeds without it, but anything using Gurobi fails at runtime.

Put your license at `$HOME/.gurobi/gurobi.lic`, which is where both the Nix shell and `build.sh` look by default.
To keep it somewhere else, point `GRB_LICENSE_FILE` at it before building:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

## Installing

```bash
cmake --install build --prefix /where/you/want
```

This gives you `bin/` with the four tools, `include/` with the headers and the TableGen output merged into one tree, `lib/` with the static libraries, and `lib/cmake/laksa/` so that a downstream CMake project can do:

```cmake
find_package(LAKSA REQUIRED)
target_link_libraries(my-tool PRIVATE LAKSA::DFGIR)
target_include_directories(my-tool PRIVATE ${LAKSA_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
```

Pass `--component` to install one piece on its own: `LAKSATools`, `LAKSAHeaders`, `LAKSALibraries`, `LAKSADevelopment`, or `LAKSAPythonModules`.

### The Python package

The bindings are a `pip` package, built from **this repository**:

```bash
pip install /path/to/laksa
```

While working on the bindings themselves, `cmake --build build --target install-python-package` does an editable install instead, so edits to the `.py` sources under [`python/mlir_laksa/`](python/mlir_laksa) take effect without reinstalling.

`cmake --install` is not a third way to do this, and nothing it writes is a `pip` source directory.
It copies the package to `<prefix>/mlir_laksa`, which no interpreter looks in, as a staging copy for packagers to relocate.
That path is what it is because `pip` sets the install prefix to the wheel's `platlib` directory, so the package has to sit at the root of the prefix.

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
pip install -r docs/requirements.txt
cmake --build build --target build-laksa-doc
```

The Nix shell already provides Sphinx, so the first line is only needed elsewhere.
