// RUN: laksa-opt %s --emithls-fuse-operator | FileCheck %s

// CHECK-LABEL: emithls.func @fuse
emithls.func @fuse()
{
    // CHECK: %[[CST:.*]]  = emithls.variable as const i32 = 2
    // CHECK: %[[V0:.*]]   = emithls.variable as i32
    // CHECK: %[[V1:.*]]   = emithls.variable as i32
    // %0 is kept because it is still used by the last arith.add below.
    // CHECK: %[[T:.*]]    = emithls.arith.add %[[V0]], %[[V1]] : i32
    // CHECK:                emithls.arith.fused add, %[[V0]], %[[V1]] : i32
    // CHECK-NOT:            emithls.arith.sub
    // CHECK:                emithls.arith.fused sub, %[[V1]], %[[V0]] : i32
    // CHECK-NOT:            emithls.arith.mul
    // CHECK:                emithls.arith.fused mul, %[[V0]], %[[V1]] : i32
    // %0 is stale for var0 here (var0 was overwritten by the mul), so the
    // last add + update cannot be fused and must remain explicit.
    // CHECK: %[[S:.*]]    = emithls.arith.add %[[T]], %[[CST]] : i32
    // CHECK:                emithls.update %[[V1]] with %[[S]] : i32 <- i32
    %cst = emithls.variable as const i32 = 2
    %var0 = emithls.variable as i32
    %var1 = emithls.variable as i32
    %0 = emithls.arith.add %var0, %var1 : i32
    emithls.update %var0 with %0 : i32 <- i32
    %1 = emithls.arith.sub %var0, %var1 : i32
    emithls.update %var1 with %1 : i32 <- i32
    %2 = emithls.arith.mul %0, %1 : i32
    emithls.update %var0 with %2 : i32 <- i32
    %3 = emithls.arith.add %0, %cst : i32
    emithls.update %var1 with %3 : i32 <- i32
}
