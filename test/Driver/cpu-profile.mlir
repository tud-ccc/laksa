// RUN: rm -rf %t
// RUN: ladle %s --cpu-profile -o %t
// RUN: test -f %t/cpu.mlir
// RUN: test ! -e %t/cpu/nodes.inc
// RUN: test -x %t/cpu/run.sh
// RUN: sh -n %t/cpu/run.sh
// RUN: %t/cpu/run.sh --help | FileCheck %s --check-prefix=RUNHELP
// RUN: c++ -O2 -std=c++17 %t/cpu/nodes.cpp %t/cpu/benchmark.cpp -o %t/cpu/benchmark
// RUN: %t/cpu/benchmark --help | FileCheck %s --check-prefix=HELP
// RUN: FileCheck %s --check-prefix=BENCH < %t/cpu/benchmark.cpp
// RUN: FileCheck %s --check-prefix=SCRIPT < %t/cpu/run.sh

// HELP: Usage: benchmark [--node NAME]
// RUNHELP: Usage: ./run.sh [options]
// RUNHELP: --processor-type NAME
// RUNHELP: --cpu ID
// BENCH: PERF_COUNT_HW_CPU_CYCLES
// BENCH: options.node == "main_node_0"
// SCRIPT: profile_file="profiles_${processor_type}.yaml"
// SCRIPT: echo "Running benchmark with:"
// SCRIPT: > "$profile_file"

#identity = affine_map<(d0) -> (d0)>

func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<4xi32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%input: i32, %output: i32):
    %doubled = arith.addi %input, %input : i32
    linalg.yield %doubled : i32
  } -> tensor<4xi32>
  return %result : tensor<4xi32>
}
