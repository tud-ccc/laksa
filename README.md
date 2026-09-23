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

## Run in Docker

Images for `linux/amd64` and `linux/arm64` are published to the GitHub Container Registry.
`main` publishes `latest`, every other branch publishes under its own name:

```bash
docker pull ghcr.io/tud-ccc/laksa:latest
```

The image carries the four tools on `PATH` and the `mlir_laksa` bindings importable from both the venv and the system interpreter.
It starts in `/work`, so mount your working directory there:

```bash
docker run --rm -it \
    -v "/PATH/TO/YOUR/WORK:/work" \
    -v "/PATH/TO/gurobi.lic:/opt/gurobi/gurobi.lic:ro" \
    ghcr.io/tud-ccc/laksa:latest
```

The license mount is not optional for that command: the pragma DSE pass in the HLS pipeline is Gurobi-backed.
Without it the entrypoint warns and only the passes that do not need Gurobi still run.

Because of `--rm`, changes made inside the container are gone once it exits, unless they are in a mounted directory.

To build the image from this repository instead:

```bash
docker build -t laksa .
```

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

## Compiling a design

`ladle` is the driver.
It dispatches to `laksa-opt` and `laksa-translate`, so everything below can also be done by hand with those two.

Run one pipeline and one translation:

```bash
ladle input.mlir -p convert-to-emithls -t emithls-to-cpp -o main.cpp
```

`--hls` does the whole thing instead.
It lowers the input twice and writes every artifact a board deployment needs below one directory:

```bash
ladle input.mlir --hls -o out_dir
```

```text
out_dir/
├── hls.mlir            the design, lowered to the EmitHLS dialect
├── ref.mlir            the same input lowered to emitc instead
├── hw/                 what the build host needs
│   ├── main.cpp        Vitis HLS input
│   ├── run_hls.tcl     C synthesis and IP export
│   ├── run_vivado.tcl  block design, synthesis, implementation, bitstream
│   └── build.sh        runs both, extracts <design>.bit from the XSA
└── app/                what the board needs
    ├── app.h           buffer sizes and AXI-Lite register offsets
    ├── app.c           userspace driver, talks to /dev/laksa
    ├── app.dtsi        device tree overlay
    ├── ref.h           the scalar reference
    ├── ref.c           compares the board's output against ref.h
    └── run.sh          loads the design, runs it, checks it
```

Every `hw/` artifact and most `app/` ones are translated from `hls.mlir`.
`ref.h` and `ref.c` come from `ref.mlir`, which is the same input lowered without any of the HLS-specific restructuring, so the reference computes what the design is *meant* to compute rather than a re-derivation of what it does.

Each step prints what it is producing, so a failure names the artifact that could not be written.

`--num-bram` and `--num-dsp` give the pragma DSE a different resource budget than the default one of `convert-to-emithls` (by default 288 BRAMs and 1248 DSPs):

```bash
ladle input.mlir --hls --num-bram=144 --num-dsp=600 -o out_dir
```

## Running a design on the board

`hw/` and `app/` are independent and can live on different machines.

**On the build host**, with the Xilinx tools sourced:

```bash
# In hw
./build.sh
```

This runs `vitis-run --mode hls`, then `vivado -mode batch`, then extracts `<design>.bit` from the exported XSA.
It takes a while.
`<design>` is the name of the design's top function, `main_top` for the examples under [`examples/`](examples).

`<design>.bit` is the one file the board needs out of `hw/`.
Put it in the `app/` directory, next to `app.c`, and deploy that directory to the board however you like; `run.sh` looks for the bitstream beside itself and nowhere else.

**On the board**, which needs [`laksa-hls-kria-driver`](https://github.com/tud-ccc/laksa-hls-kria-dirver) loaded and its `laksa.h` installed:

```bash
# In app
./run.sh
```

`run.sh` compiles the overlay with `dtc`, loads it and the bitstream with `fpgautil`, builds `app` and `ref`, fills any missing `input<n>.bin` with random bytes, runs the design, and compares what it wrote against the reference:

```text
output0.bin: all <num> elements match the reference
```

Mismatches are reported per element, with the index into the output buffer, and `ref` exits non-zero.

The `.bin` files are raw dumps of the DMA buffers with no header, one `input<n>.bin` per argument the kernel reads and one `output<n>.bin` per argument it writes.
Dropping in your own inputs of the right size is all it takes to run real data; the sizes are in `app.h`.
