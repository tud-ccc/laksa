// RUN: laksa-opt %s --emithls-merge-cast-chain --canonicalize | FileCheck %s

// CHECK-LABEL: emithls.func @merge_chain
emithls.func @merge_chain(%arg0: !emithls.stream<i8>, %arg1: !emithls.stream<i64>)
{
    // CHECK:      %[[READ:.*]] = emithls.stream.read %arg0 : !emithls.stream<i8> -> i8
    // CHECK-NEXT: %[[CAST:.*]] = emithls.arith.cast %[[READ]] : i8 to i64
    // CHECK-NEXT: emithls.stream.write %[[CAST]] to %arg1 : i64 -> !emithls.stream<i64>
    // CHECK-NOT:  emithls.arith.cast {{.*}} : i8 to i16
    // CHECK-NOT:  emithls.arith.cast {{.*}} : i16 to i32
    // CHECK-NOT:  emithls.arith.cast {{.*}} : i32 to i64
    %0 = emithls.stream.read %arg0 : !emithls.stream<i8> -> i8
    %1 = emithls.arith.cast %0 : i8 to i16
    %2 = emithls.arith.cast %1 : i16 to i32
    %3 = emithls.arith.cast %2 : i32 to i64
    emithls.stream.write %3 to %arg1 : i64 -> !emithls.stream<i64>
}

// CHECK-LABEL: emithls.func @merge_multiout
emithls.func @merge_multiout(%arg0: !emithls.stream<i8>, %arg1: !emithls.stream<i32>, %arg2: !emithls.stream<i64>)
{
    // CHECK:      %[[READ:.*]] = emithls.stream.read %arg0 : !emithls.stream<i8> -> i8
    // CHECK-NEXT: %[[CAST0:.*]] = emithls.arith.cast %[[READ]] : i8 to i32
    // CHECK-NEXT: %[[CAST1:.*]] = emithls.arith.cast %[[READ]] : i8 to i64
    // CHECK-NEXT: emithls.stream.write %[[CAST0]] to %arg1 : i32 -> !emithls.stream<i32>
    // CHECK-NEXT: emithls.stream.write %[[CAST1]] to %arg2 : i64 -> !emithls.stream<i64>
    // CHECK-NOT:  emithls.arith.cast {{.*}} : i8 to i16
    %0 = emithls.stream.read %arg0 : !emithls.stream<i8> -> i8
    %1 = emithls.arith.cast %0 : i8 to i16
    %2 = emithls.arith.cast %1 : i16 to i32
    %3 = emithls.arith.cast %2 : i32 to i64
    emithls.stream.write %2 to %arg1 : i32 -> !emithls.stream<i32>
    emithls.stream.write %3 to %arg2 : i64 -> !emithls.stream<i64>
}
