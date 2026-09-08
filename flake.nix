{
  description = "LAKSA-Compiler Development Environment";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";

    mlir = {
      url = "github:jhbi826c/mlir-flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, mlir }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        mlirPkg = mlir.packages.${system}.mlir;
        pythonEnv = mlir.packages.${system}.pythonEnv;
        mlirPythonPath = "${mlirPkg}/python_packages/mlir_core";
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.cmake pkgs.ninja pkgs.mold
            mlirPkg
            pkgs.doxygen
            pkgs.gurobi
            pkgs.zlib pkgs.libxml2
            pkgs.docker-client pkgs.docker-buildx
          ];
          shellHook = ''
            export LLVM_DIR="${mlirPkg}/lib/cmake/llvm"
            export MLIR_DIR="${mlirPkg}/lib/cmake/mlir"
            export LLVM_EXTERNAL_LIT="${mlirPkg}/bin/lit"
            export PYTHONPATH="${mlirPythonPath}:$PYTHONPATH"
            export GUROBI_HOME="${pkgs.gurobi}"
            export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
            if [ ! -d .venv ]; then
              ${pythonEnv}/bin/python3 -m venv .venv --system-site-packages
            fi
            source .venv/bin/activate
            python -m pip install --upgrade pip
          '';
        };
      }
    );
}
