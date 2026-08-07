// RUN: laksa-opt %s --convert-memref-pad-to-laksa-loops | FileCheck %s

// CHECK-DAG: #[[SET2D:.+]] = affine_set<(d0, d1) : (d0 - 1 >= 0, -d0 + 30 >= 0, d1 - 1 >= 0, -d1 + 30 >= 0)>
// CHECK-DAG: #[[SET1D:.+]] = affine_set<(d0) : (d0 - 4 >= 0, -d0 + 103 >= 0)>

// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<1x8xi8>
// CHECK-NEXT: affine.for %[[I0:.+]] = 0 to 32 {
// CHECK-NEXT: affine.for %[[I1:.+]] = 0 to 32 {
// CHECK-NEXT: affine.if #[[SET2D]](%[[I0]], %[[I1]]) {
// CHECK-NEXT: %[[TOK:.+]] = dfg.pull_as_memref %in0 : !dfg.output<1x8xi8>
// CHECK-NEXT: dfg.push_memref %[[TOK]] to %out0 : !dfg.input<1x8xi8>
// CHECK-NEXT: } else {
// CHECK-NEXT: dfg.push_memref %[[ALLOC]] to %out0 : !dfg.input<1x8xi8>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
dfg.process @pad_i8 inputs(%in0: !dfg.output<1x8xi8>) outputs(%out0: !dfg.input<1x8xi8>) {
  dfg.loop inputs(%in0: !dfg.output<1x8xi8>) outputs(%out0: !dfg.input<1x8xi8>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x8xi8>
    %0 = builtin.unrealized_conversion_cast %token_memref0 : memref<1x8xi8> to memref<1x30x30x8xi8>
    %alloc = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<1x32x32x8xi8>
    %subview = memref.subview %alloc[0, 1, 1, 0] [1, 30, 30, 8] [1, 1, 1, 1] : memref<1x32x32x8xi8> to memref<1x30x30x8xi8, strided<[8192, 256, 8, 1], offset: 264>>
    memref.copy %0, %subview : memref<1x30x30x8xi8> to memref<1x30x30x8xi8, strided<[8192, 256, 8, 1], offset: 264>>
    %1 = builtin.unrealized_conversion_cast %alloc : memref<1x32x32x8xi8> to memref<1x8xi8>
    dfg.push_memref %1 to %out0 : !dfg.input<1x8xi8>
  }
}

// CHECK: %[[CST:.+]] = arith.constant -128 : i8
// CHECK-NEXT: affine.for %[[J0:.+]] = 0 to 104 {
// CHECK-NEXT: affine.if #[[SET1D]](%[[J0]]) {
// CHECK-NEXT: %[[TOK2:.+]] = dfg.pull %in0 : !dfg.output<i8>
// CHECK-NEXT: dfg.push %[[TOK2]] to %out0 : !dfg.input<i8>
// CHECK-NEXT: } else {
// CHECK-NEXT: dfg.push %[[CST]] to %out0 : !dfg.input<i8>
// CHECK-NEXT: }
// CHECK-NEXT: }
dfg.process @pad_scalar_i8 inputs(%in0: !dfg.output<i8>) outputs(%out0: !dfg.input<i8>) {
  dfg.loop inputs(%in0: !dfg.output<i8>) outputs(%out0: !dfg.input<i8>) {
    %token0 = dfg.pull %in0 : !dfg.output<i8>
    %0 = builtin.unrealized_conversion_cast %token0 : i8 to memref<100xi8>
    %alloc = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<104xi8>
    %subview = memref.subview %alloc[4] [100] [1] : memref<104xi8> to memref<100xi8, strided<[1], offset: 4>>
    memref.copy %0, %subview : memref<100xi8> to memref<100xi8, strided<[1], offset: 4>>
    %1 = builtin.unrealized_conversion_cast %alloc : memref<104xi8> to i8
    dfg.push %1 to %out0 : !dfg.input<i8>
  }
}
