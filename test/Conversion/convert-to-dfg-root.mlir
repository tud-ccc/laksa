// RUN: laksa-opt %s --convert-to-dfg | FileCheck %s

// Verify that the conversion pipeline creates and propagates the root marker;
// the input function itself has no laksa.root attribute.

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: dfg.operator @main_node_0
// CHECK-LABEL: dfg.region @main_top
// CHECK: attributes {laksa.root}

func.func @main(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel"]
    } ins(%arg : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
        %negated = arith.negf %in : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
}
