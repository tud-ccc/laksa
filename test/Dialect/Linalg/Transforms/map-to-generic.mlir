// RUN: laksa-opt %s --linalg-map-to-generic | FileCheck %s

// CHECK-DAG: #[[MAP_SCALAR:.+]] = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-DAG: #[[MAP_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP_1D:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @map_fill()
// CHECK:       %[[ZERO:.+]] = arith.constant 0 : i8
// CHECK:       %[[EMPTY:.+]] = tensor.empty
// CHECK:       %[[MAP:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP_SCALAR]], #[[MAP_4D]]]
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:    ins(%[[ZERO]] : i8) outs(%[[EMPTY]] : tensor<1x32x32x8xi8>)
// CHECK:       ^bb0(%[[IN:.+]]: i8, %{{.+}}: i8):
// CHECK:         linalg.yield %[[IN]] : i8
func.func @map_fill() -> tensor<1x32x32x8xi8>
{
    %0 = arith.constant 0 : i8
    %empty = tensor.empty() : tensor<1x32x32x8xi8>
    %map = linalg.map outs(%empty: tensor<1x32x32x8xi8>)
    (%out: i8) {
        linalg.yield %0 : i8
    }
    return %map : tensor<1x32x32x8xi8>
}

// An elementwise addition
// CHECK-LABEL: func.func @map_elementwise(
// CHECK-SAME:    %[[LHS:.+]]: tensor<64xi32>, %[[RHS:.+]]: tensor<64xi32>
// CHECK:       %[[EMPTY:.+]] = tensor.empty
// CHECK:       %[[MAP:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP_1D]], #[[MAP_1D]], #[[MAP_1D]]]
// CHECK-SAME:    iterator_types = ["parallel"]
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]] : tensor<64xi32>, tensor<64xi32>) outs(%[[EMPTY]] : tensor<64xi32>)
// CHECK:       ^bb0(%[[IN:.+]]: i32, %[[IN_0:.+]]: i32, %{{.+}}: i32):
// CHECK:         %[[SUM:.+]] = arith.addi %[[IN]], %[[IN_0]] : i32
// CHECK:         linalg.yield %[[SUM]] : i32
func.func @map_elementwise(%lhs: tensor<64xi32>, %rhs: tensor<64xi32>) -> tensor<64xi32>
{
    %empty = tensor.empty() : tensor<64xi32>
    %map = linalg.map ins(%lhs, %rhs : tensor<64xi32>, tensor<64xi32>) outs(%empty : tensor<64xi32>)
    (%in: i32, %in_0: i32, %out: i32) {
        %0 = arith.addi %in, %in_0 : i32
        linalg.yield %0 : i32
    }
    return %map : tensor<64xi32>
}
