// RUN: laksa-opt %s --convert-to-laksa-emitc | FileCheck %s

// Verify that the generic EmitC pipeline lowers the subview and copy produced
// by bufferizing tensor.pad.

// CHECK-LABEL: emitc.func @main(
// CHECK-NOT:     memref.
// CHECK-NOT:     tensor.
// CHECK-NOT:     unrealized_conversion_cast
// CHECK:         return
func.func @main(%arg: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %zero = arith.constant 0 : i32
  %padded = tensor.pad %arg low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : i32
  } : tensor<2x2xi32> to tensor<4x4xi32>
  return %padded : tensor<4x4xi32>
}
