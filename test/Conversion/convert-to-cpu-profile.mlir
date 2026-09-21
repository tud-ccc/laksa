// RUN: laksa-opt %s --convert-to-laksa-cpu-profile | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

// CHECK-NOT: emitc.func @main(
// CHECK-LABEL: emitc.func @main_node_0(
// CHECK-SAME: !emitc.array<4xi32>
// CHECK-NOT: memref.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
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
