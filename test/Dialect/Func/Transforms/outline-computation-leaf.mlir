// RUN: laksa-opt %s --func-outline-computation-leaf | FileCheck %s

// One function exercising all paths in the pass:
//   - linalg.generic fill (stored/absorbed into node 0)
//   - linalg.generic computation using fill output + block arg (node 0)
//   - tensor.expand_shape folded into node 0 return
//   - tensor.pad with body-captured constant (node 1)
//   - tensor.collapse_shape folded into node 1 return

#scalar = affine_map<(d0) -> ()>
#map    = affine_map<(d0) -> (d0)>

// Node 0: fill absorbed, arg0 passed in; expand_shape folded into return.
// CHECK-LABEL: func.func @test_node_0(
// CHECK-SAME:    %[[A0:.*]]: tensor<4xf32>) -> tensor<1x4xf32>
// CHECK-DAG:     arith.constant 0.{{0+}}e+{{0+}} : f32
// CHECK-DAG:     tensor.empty() : tensor<4xf32>
// CHECK:         linalg.generic {{.*}} ins(%[[A0]]
// CHECK:         tensor.expand_shape %{{.*}} {{\[}}[0, 1]{{\]}} output_shape [1, 4]
// CHECK:         return %{{.*}} : tensor<1x4xf32>

// Node 1: pad with captured constant; collapse_shape folded into return.
// CHECK-LABEL: func.func @test_node_1(
// CHECK-SAME:    %[[A1:.*]]: tensor<4x4xf32>) -> tensor<36xf32>
// CHECK:         %[[CST:.*]] = arith.constant 1.{{0+}}e+{{0+}} : f32
// CHECK:         tensor.pad %[[A1]] low[1, 1] high[1, 1]
// CHECK:           tensor.yield %[[CST]]
// CHECK:         tensor.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
// CHECK:         return %{{.*}} : tensor<36xf32>

// Wrapper: two calls forwarding each block arg, returning both results.
// CHECK-LABEL: func.func @test_top(
// CHECK-SAME:    %[[A0:.*]]: tensor<4xf32>, %[[A1:.*]]: tensor<4x4xf32>
// CHECK:         %[[R0:.*]] = call @test_node_0(%[[A0]])
// CHECK:         %[[R1:.*]] = call @test_node_1(%[[A1]])
// CHECK:         return %[[R0]], %[[R1]]

func.func @test(%arg0: tensor<4xf32>, %arg1: tensor<4x4xf32>)
        -> (tensor<1x4xf32>, tensor<36xf32>) {
    // fill/broadcast stored generic absorbed by node 0
    %zero      = arith.constant 0.0 : f32
    %empty_acc = tensor.empty() : tensor<4xf32>
    %filled    = linalg.generic {
        indexing_maps = [#scalar, #map], iterator_types = ["parallel"]
    } ins(%zero : f32) outs(%empty_acc : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<4xf32>

    // node 0: add arg0 and fill result; result folded through expand_shape
    %empty0 = tensor.empty() : tensor<4xf32>
    %sum    = linalg.generic {
        indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]
    } ins(%arg0, %filled : tensor<4xf32>, tensor<4xf32>)
      outs(%empty0 : tensor<4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
        %v = arith.addf %in0, %in1 : f32
        linalg.yield %v : f32
    } -> tensor<4xf32>
    %expanded = tensor.expand_shape %sum [[0, 1]] output_shape [1, 4]
                    : tensor<4xf32> into tensor<1x4xf32>

    // node 1: pad arg1 (constant captured from body region);
    //         result folded through collapse_shape
    %pad_cst = arith.constant 1.0 : f32
    %padded  = tensor.pad %arg1 low[1, 1] high[1, 1] {
    ^bb0(%i: index, %j: index):
        tensor.yield %pad_cst : f32
    } : tensor<4x4xf32> to tensor<6x6xf32>
    %collapsed = tensor.collapse_shape %padded [[0, 1]]
                    : tensor<6x6xf32> into tensor<36xf32>

    return %expanded, %collapsed : tensor<1x4xf32>, tensor<36xf32>
}
