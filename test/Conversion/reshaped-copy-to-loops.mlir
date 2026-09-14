// RUN: laksa-opt %s --convert-reshaped-copy-to-loops | FileCheck %s

// CHECK-LABEL: func.func @unit_dims
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         scf.for %[[I:.*]] = %[[ZERO]] to %[[FOUR]]
// CHECK:           scf.for %[[J:.*]] = %[[ZERO]] to %[[EIGHT]]
// CHECK:             %[[V:.*]] = memref.load %{{.*}}[%[[I]], %[[ZERO]], %[[ZERO]], %[[J]]]
// CHECK:             memref.store %[[V]], %{{.*}}[%[[ZERO]], %[[ZERO]], %[[I]], %[[J]]]
func.func @unit_dims(%src: memref<4x1x1x8xi8>, %dst: memref<1x1x4x8xi8>) {
  %c = memref.reinterpret_cast %src to offset: [0], sizes: [1, 1, 4, 8], strides: [32, 32, 8, 1] : memref<4x1x1x8xi8> to memref<1x1x4x8xi8>
  memref.copy %c, %dst : memref<1x1x4x8xi8> to memref<1x1x4x8xi8>
  return
}

// CHECK-LABEL: func.func @regrouped
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[FOUR:.*]] = arith.constant 4 : index
// CHECK:         scf.for %[[I:.*]] = %[[ZERO]] to %[[TWO]]
// CHECK:           scf.for %[[J:.*]] = %[[ZERO]] to %[[THREE]]
// CHECK:             scf.for %[[K:.*]] = %[[ZERO]] to %[[FOUR]]
// CHECK:               %[[JS:.*]] = arith.muli %[[J]], %[[FOUR]]
// CHECK:               %[[SRC:.*]] = arith.addi %[[JS]], %[[K]]
// CHECK:               %[[V:.*]] = memref.load %{{.*}}[%[[I]], %[[SRC]]]
// CHECK:               %[[IS:.*]] = arith.muli %[[I]], %[[THREE]]
// CHECK:               %[[DST:.*]] = arith.addi %[[IS]], %[[J]]
// CHECK:               memref.store %[[V]], %{{.*}}[%[[DST]], %[[K]]]
func.func @regrouped(%src: memref<2x12xi8>, %dst: memref<6x4xi8>) {
  %c = memref.reinterpret_cast %src to offset: [0], sizes: [6, 4], strides: [4, 1] : memref<2x12xi8> to memref<6x4xi8>
  memref.copy %c, %dst : memref<6x4xi8> to memref<6x4xi8>
  return
}

// CHECK-LABEL: func.func @incompatible
// CHECK:         memref.reinterpret_cast
// CHECK:         memref.copy
func.func @incompatible(%src: memref<6x4xi8>, %dst: memref<8x3xi8>) {
  %c = memref.reinterpret_cast %src to offset: [0], sizes: [8, 3], strides: [3, 1] : memref<6x4xi8> to memref<8x3xi8>
  memref.copy %c, %dst : memref<8x3xi8> to memref<8x3xi8>
  return
}

// CHECK-LABEL: func.func @offset
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         scf.for %[[I:.*]] = %[[ZERO]] to %[[FOUR]]
// CHECK:           scf.for %[[J:.*]] = %[[ZERO]] to %[[EIGHT]]
// CHECK:             %[[ROW:.*]] = arith.addi %[[I]], %[[FOUR]]
// CHECK:             %[[V:.*]] = memref.load %{{.*}}[%[[ROW]], %[[J]]]
// CHECK:             memref.store %[[V]], %{{.*}}[%[[I]], %[[J]]]
func.func @offset(%src: memref<8x8xi8>, %dst: memref<4x8xi8>) {
  %c = memref.reinterpret_cast %src to offset: [32], sizes: [4, 8], strides: [8, 1] : memref<8x8xi8> to memref<4x8xi8, strided<[8, 1], offset: 32>>
  memref.copy %c, %dst : memref<4x8xi8, strided<[8, 1], offset: 32>> to memref<4x8xi8>
  return
}

// CHECK-LABEL: func.func @padded
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[FOUR:.*]] = arith.constant 4 : index
// CHECK:         scf.for %[[I:.*]] = %[[ZERO]] to %[[TWO]]
// CHECK:           scf.for %[[J:.*]] = %[[ZERO]] to %[[TWO]]
// CHECK:             scf.for %[[K:.*]] = %[[ZERO]] to %[[FOUR]]
// CHECK:               %[[V:.*]] = memref.load %{{.*}}[%[[ZERO]], %[[I]], %[[J]], %[[K]]]
// CHECK:               %[[ROW:.*]] = arith.addi %[[I]], %[[ONE]]
// CHECK:               %[[COL:.*]] = arith.addi %[[J]], %[[ONE]]
// CHECK:               memref.store %[[V]], %{{.*}}[%[[ZERO]], %[[ROW]], %[[COL]], %[[K]]]
func.func @padded(%src: memref<1x2x2x4xi8>, %dst: memref<1x4x4x4xi8>) {
  %c = memref.reinterpret_cast %dst to offset: [20], sizes: [1, 2, 2, 4], strides: [64, 16, 4, 1] : memref<1x4x4x4xi8> to memref<1x2x2x4xi8, strided<[64, 16, 4, 1], offset: 20>>
  memref.copy %src, %c : memref<1x2x2x4xi8> to memref<1x2x2x4xi8, strided<[64, 16, 4, 1], offset: 20>>
  return
}

// CHECK-LABEL: func.func @stepped
// CHECK:         memref.reinterpret_cast
// CHECK:         memref.copy
func.func @stepped(%src: memref<8x8xi8>, %dst: memref<4x8xi8>) {
  %c = memref.reinterpret_cast %src to offset: [0], sizes: [4, 8], strides: [16, 1] : memref<8x8xi8> to memref<4x8xi8, strided<[16, 1]>>
  memref.copy %c, %dst : memref<4x8xi8, strided<[16, 1]>> to memref<4x8xi8>
  return
}
