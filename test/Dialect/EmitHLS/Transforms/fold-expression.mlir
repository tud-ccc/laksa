// RUN: laksa-opt %s --emithls-fold-expression --canonicalize | FileCheck %s

// CHECK-LABEL: emithls.func @fold_compare
// CHECK-DAG: %[[ONE:.*]] = emithls.variable as const index = 1
// CHECK-DAG: %[[THIRTY:.*]] = emithls.variable as const index = 30
// CHECK: emithls.for %idx0 = 0 to 32 step 1
// CHECK-NEXT: emithls.for %idx1 = 0 to 32 step 1
// CHECK-NEXT: %{{.*}} = emithls.expr : i1 {
// CHECK-NEXT: %[[C1:.*]] = emithls.arith.cmp ge, %idx0, %[[ONE]] : index
// CHECK-NEXT: %[[C2:.*]] = emithls.arith.cmp le, %idx0, %[[THIRTY]] : index
// CHECK-NEXT: %[[C3:.*]] = emithls.arith.cmp ge, %idx1, %[[ONE]] : index
// CHECK-NEXT: %[[C4:.*]] = emithls.arith.cmp le, %idx1, %[[THIRTY]] : index
// CHECK-NEXT: %[[AND:.*]] = emithls.arith.logical_and %[[C1]], %[[C2]], %[[C3]], %[[C4]] : i1
// CHECK-NEXT: emithls.yield %[[AND]] : i1
emithls.func @fold_compare(%arg0: !emithls.array<32x!emithls.stream<i8>>, %arg1: !emithls.array<32x!emithls.stream<i8>>) {
  %const_index_0 = emithls.variable as const index = 30
  %const_index_1 = emithls.variable as const index = -1
  %const_index_2 = emithls.variable as const index = 0
  emithls.for %idx0 = 0 to 32 step 1 {
    emithls.for %idx1 = 0 to 32 step 1 {
      %expr0 = emithls.expr : i1 {
        %0 = emithls.arith.add %idx0, %const_index_1 : index
        %1 = emithls.arith.cmp ge, %0, %const_index_2 : index
        %2 = emithls.arith.mul %idx0, %const_index_1 : index
        %3 = emithls.arith.add %2, %const_index_0 : index
        %4 = emithls.arith.cmp ge, %3, %const_index_2 : index
        %5 = emithls.arith.add %idx1, %const_index_1 : index
        %6 = emithls.arith.cmp ge, %5, %const_index_2 : index
        %7 = emithls.arith.mul %idx1, %const_index_1 : index
        %8 = emithls.arith.add %7, %const_index_0 : index
        %9 = emithls.arith.cmp ge, %8, %const_index_2 : index
        %10 = emithls.arith.logical_and %1, %4, %6, %9 : i1
        emithls.yield %10 : i1
      }
      emithls.if %expr0 {
        %v = emithls.stream.read %arg0[%idx0] : !emithls.array<32x!emithls.stream<i8>> -> i8
        emithls.stream.write %v to %arg1[%idx0] : i8 -> !emithls.array<32x!emithls.stream<i8>>
      }
    }
  }
}
