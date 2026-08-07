// RUN: laksa-opt %s --convert-affine-to-emithls --canonicalize | FileCheck %s

#set = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 2 >= 0)>

// CHECK-LABEL: dfg.process @affine
dfg.process @affine inputs(%in: !dfg.output<2x4xi8>) outputs(%out: !dfg.input<2x4xi8>) {
    // CHECK: %[[CN2:.+]] = arith.constant -2 : index
    // CHECK-NEXT: %[[CN1:.+]] = arith.constant -1 : index
    // CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
    dfg.loop inputs(%in: !dfg.output<2x4xi8>) outputs(%out: !dfg.input<2x4xi8>) {
        %0 = dfg.pull_as_memref %in : !dfg.output<2x4xi8>
        // CHECK: emithls.for %idx0 = 0 to 2 step 1 {
        // CHECK-NEXT: emithls.for %idx1 = 0 to 4 step 1 {
        affine.for %idx0 = 0 to 2 {
            affine.for %idx1 = 0 to 4 {
                // CHECK-NEXT: %[[EXPR:.+]] = emithls.expr : i1 {
                // CHECK-NEXT: %[[E0:.+]] = arith.addi %idx0, %[[CN1]] : index
                // CHECK-NEXT: %[[CMP0:.+]] = emithls.arith.cmp ge, %[[E0]], %[[C0]] : index
                // CHECK-NEXT: %[[E1:.+]] = arith.addi %idx1, %[[CN2]] : index
                // CHECK-NEXT: %[[CMP1:.+]] = emithls.arith.cmp ge, %[[E1]], %[[C0]] : index
                // CHECK-NEXT: %[[AND:.+]] = emithls.arith.logical_and %[[CMP0]], %[[CMP1]] : i1
                // CHECK-NEXT: emithls.yield %[[AND]] : i1
                // CHECK-NEXT: }
                // CHECK-NEXT: emithls.if %[[EXPR]] {
                affine.if #set(%idx0, %idx1) {
                    // CHECK-NEXT: %[[ARR0:.+]] = builtin.unrealized_conversion_cast %token_memref0 : memref<2x4xi8> to !emithls.array<2x4xi8>
                    // CHECK-NEXT: %[[VAL:.+]] = emithls.array.read %[[ARR0]][%idx0, %idx1] : !emithls.array<2x4xi8> -> i8
                    %1 = affine.load %0[%idx0, %idx1] : memref<2x4xi8>
                    // CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[VAL]], %[[VAL]] : i8
                    %2 = arith.addi %1, %1 : i8
                    // CHECK-NEXT: dfg.push %[[SUM]] to %out0[%idx0, %idx1] : !dfg.input<2x4xi8>
                    dfg.push %2 to %out[%idx0, %idx1] : !dfg.input<2x4xi8>
                }
            }
        }
    }
}

// CHECK-NOT: memref.global
// CHECK-NOT: memref.get_global
memref.global "private" constant @__constant_4xi8 : memref<4xi8> = dense<[1, 2, 3, 4]>

// CHECK-LABEL: dfg.process @global_const
// CHECK: %[[VAR:.+]] = emithls.variable as const !emithls.array<4xi8> = dense<[1, 2, 3, 4]>
dfg.process @global_const outputs(%out: !dfg.input<4xi8>) {
    // CHECK: dfg.loop outputs(%out0: !dfg.input<4xi8>) {
    dfg.loop outputs(%out: !dfg.input<4xi8>) {
        %g = memref.get_global @__constant_4xi8 : memref<4xi8>
        // CHECK-NEXT: emithls.for %idx0 = 0 to 4 step 1 {
        affine.for %idx0 = 0 to 4 {
            %0 = affine.load %g[%idx0] : memref<4xi8>
            // CHECK-NEXT: %[[VAL:.+]] = emithls.array.read %[[VAR]][%idx0] : !emithls.array<4xi8> -> i8
            // CHECK-NEXT: dfg.push %[[VAL]] to %out0[%idx0] : !dfg.input<4xi8>
            dfg.push %0 to %out[%idx0] : !dfg.input<4xi8>
        }
    }
}

// CHECK-LABEL: dfg.region @top
dfg.region @top inputs(%in0: !dfg.output<2x4xi8>) outputs(%out0: !dfg.input<2x4xi8>, %out1: !dfg.input<4xi8>) {
    // CHECK: dfg.instantiate @affine inputs(%in0) outputs(%out0) : (!dfg.output<2x4xi8>) -> !dfg.input<2x4xi8>
    dfg.instantiate @affine inputs(%in0) outputs(%out0) : (!dfg.output<2x4xi8>) -> !dfg.input<2x4xi8>
    // CHECK: dfg.instantiate @global_const outputs(%out1) : () -> !dfg.input<4xi8>
    dfg.instantiate @global_const outputs(%out1) : () -> !dfg.input<4xi8>
}
