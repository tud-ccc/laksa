// RUN: laksa-opt %s --convert-linalg-to-laksa-loops | FileCheck %s

// CHECK-DAG: #[[SET2D:.+]] = affine_set<(d0, d1) : (d0 - 2 >= 0, d1 - 2 >= 0)>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>

memref.global "private" constant @__constant_8xi32 : memref<8xi32>
memref.global "private" constant @__constant_8x3x3x8xi8 : memref<8x3x3x8xi8>

// CHECK-LABEL: dfg.process @kernel_node_0
dfg.process @kernel_node_0 inputs(%in0: !dfg.output<1x224x224x8xi8>) outputs(%out0: !dfg.input<1x222x222x8xi32>) {
  %c-128_i32 = arith.constant -128 : i32
  %c0_i32 = arith.constant 0 : i32
  dfg.loop inputs(%in0: !dfg.output<1x224x224x8xi8>) outputs(%out0: !dfg.input<1x222x222x8xi32>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x224x224x8xi8>
    %0 = memref.get_global @__constant_8x3x3x8xi8 : memref<8x3x3x8xi8>
    // CHECK: %[[ALLOC:.+]] = memref.alloc() {accu_at = [0 : i32, 3 : i32], alignment = 64 : i64, fill_with = 0 : i32} : memref<1x222x222x8xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x222x222x8xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc : memref<1x222x222x8xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    // CHECK: affine.for %[[ARG0:.+]] = 0 to 224 {
    // CHECK-NEXT: affine.for %[[ARG1:.+]] = 0 to 224 {
    // CHECK-NEXT: %[[LINEBUF:.+]] = emithls.helper.linebuf %token_memref0 num_chan [1, 8] num_line 2 keep %[[ARG1]] : memref<1x224x224x8xi8> -> memref<1x8x2x224xi8>
    // CHECK-NEXT: %[[WINDOW:.+]] = emithls.helper.window %[[LINEBUF]] from [%[[ARG0]], %[[ARG1]]] : memref<1x8x2x224xi8> -> memref<1x8x3x3xi8>
    // CHECK-NEXT: affine.if #[[SET2D]](%[[ARG0]], %[[ARG1]]) {
    // CHECK-NEXT: affine.for %[[ARG2:.+]] = 0 to 1 {
    // CHECK-NEXT: affine.for %[[ARG3:.+]] = 0 to 8 {
    // CHECK: affine.for %[[ARG4:.+]] = 0 to 3 {
    // CHECK-NEXT: affine.for %[[ARG5:.+]] = 0 to 3 {
    // CHECK-NEXT: affine.for %[[ARG6:.+]] = 0 to 8 {
    // CHECK: emithls.helper.accumulate %[[ALLOC]] at [%[[ARG2]], %[[ARG3]]] add %{{.+}} : memref<1x222x222x8xi32>
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%token_memref0, %0 : memref<1x224x224x8xi8>, memref<8x3x3x8xi8>) outs(%alloc : memref<1x222x222x8xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %1 = arith.extsi %in : i8 to i32
      %2 = arith.subi %1, %c-128_i32 : i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    }
    dfg.push_memref %alloc to %out0 : !dfg.input<1x222x222x8xi32>
  }
}

// CHECK-LABEL: dfg.process @kernel_node_1
dfg.process @kernel_node_1 inputs(%in0: !dfg.output<1x222x222x8xi32>) outputs(%out0: !dfg.input<1x222x222x8xi8>) {
  %c127_i32 = arith.constant 127 : i32
  %c-128_i32 = arith.constant -128 : i32
  %c40_i64 = arith.constant 40 : i64
  %c-1073741824_i64 = arith.constant -1073741824 : i64
  %c1073741824_i64 = arith.constant 1073741824 : i64
  %c0_i32 = arith.constant 0 : i32
  %c549755813888_i64 = arith.constant 549755813888 : i64
  dfg.loop inputs(%in0: !dfg.output<1x222x222x8xi32>) outputs(%out0: !dfg.input<1x222x222x8xi8>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x222x222x8xi32>
    %0 = memref.get_global @__constant_8xi32 : memref<8xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x222x222x8xi8>
    // CHECK: affine.for %{{.+}} = 0 to 1 {
    // CHECK-NEXT: affine.for %{{.+}} = 0 to 222 {
    // CHECK-NEXT: affine.for %{{.+}} = 0 to 222 {
    // CHECK-NEXT: affine.for %{{.+}} = 0 to 8 {
    // CHECK: affine.store %{{.+}}, %alloc{{.*}} : memref<1x222x222x8xi8>
    linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%token_memref0, %0 : memref<1x222x222x8xi32>, memref<8xi32>) outs(%alloc : memref<1x222x222x8xi8>) {
    ^bb0(%in: i32, %in_0: i32, %out: i8):
      %1 = arith.extsi %in : i32 to i64
      %2 = arith.extsi %in_0 : i32 to i64
      %3 = arith.muli %1, %2 : i64
      %4 = arith.addi %3, %c549755813888_i64 : i64
      %5 = arith.cmpi sge, %in, %c0_i32 : i32
      %6 = arith.select %5, %c1073741824_i64, %c-1073741824_i64 : i64
      %7 = arith.addi %6, %4 : i64
      %8 = arith.shrsi %7, %c40_i64 : i64
      %9 = arith.trunci %8 : i64 to i32
      %10 = arith.addi %9, %c-128_i32 : i32
      %11 = arith.maxsi %10, %c-128_i32 : i32
      %12 = arith.minsi %11, %c127_i32 : i32
      %13 = arith.trunci %12 : i32 to i8
      linalg.yield %13 : i8
    }
    dfg.push_memref %alloc to %out0 : !dfg.input<1x222x222x8xi8>
  }
}

dfg.region @combined
    inputs(%conv_in : !dfg.output<1x224x224x8xi8>)
    outputs(%conv_out : !dfg.input<1x222x222x8xi8>) {
  %conv_mid_in, %conv_mid_out = dfg.channel() : 1x222x222x8xi32
  dfg.instantiate @kernel_node_0 inputs(%conv_in) outputs(%conv_mid_in) : (!dfg.output<1x224x224x8xi8>) -> !dfg.input<1x222x222x8xi32>
  dfg.instantiate @kernel_node_1 inputs(%conv_mid_out) outputs(%conv_out) : (!dfg.output<1x222x222x8xi32>) -> !dfg.input<1x222x222x8xi8>
}
