// RUN: laksa-opt %s --linalg-scalarize-splat-dense | FileCheck %s

// Verify that a splat dense tensor constant is replaced by a scalar 
// constant with the splat value, when it's used in a linalg.generic
// as input.

// CHECK-DAG: #[[MAP_SCALAR:.+]] = affine_map<(d0) -> ()>
// CHECK-DAG: #[[MAP_1D:.+]] = affine_map<(d0) -> (d0)>
#map = affine_map<(d0)->(d0)>

// CHECK-LABEL: func.func @scalarize(
// CHECK: %[[SCALAR:.+]] = arith.constant 2 : i8
// CHECK: %[[DENSE:.+]] = arith.constant
// CHECK: %[[EMPTY0:.+]] = tensor.empty
// CHECK: %[[EMPTY1:.+]] = tensor.empty
// CHECK: %[[RESULTS:.+]]:2 = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP_SCALAR]], #[[MAP_1D]], #[[MAP_1D]], #[[MAP_1D]], #[[MAP_1D]]]
// CHECK-SAME: iterator_types = ["parallel"]
func.func @scalarize(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
{
    %0 = arith.constant dense<2> : tensor<4xi8>
    %1 = arith.constant dense<[1,2,3,4]> : tensor<4xi8>
    %2 = tensor.empty() : tensor<4xi8>
    %3 = tensor.empty() : tensor<4xi8>
    %4, %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1, %arg0 : tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) outs(%2, %3 : tensor<4xi8>, tensor<4xi8>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i8, %out_0: i8):
        %6 = arith.addi %in, %in_0 : i8
        %7 = arith.muli %in, %in_1 : i8
        linalg.yield %6, %7 : i8, i8
    } -> (tensor<4xi8>, tensor<4xi8>)
    func.return %4, %5 : tensor<4xi8>, tensor<4xi8>
}
