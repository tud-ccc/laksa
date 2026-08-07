// RUN: laksa-opt %s --emithls-insert-includes | FileCheck %s

// The pass inspects operand/result types of all ops and inserts the required
// headers at the module top.  Insertion order: cstddef, algorithm, ap_int.h,
// hls_stream.h.

// CHECK:      emithls.include "cstddef"
// CHECK:      emithls.include "algorithm"
// CHECK:      emithls.include "ap_int.h"
// CHECK:      emithls.include "hls_stream.h"

// !emithls.stream<i32> → hasStream=true (StreamType) and hasInteger=true
// (i32 element type checked via checkLeafType).
// emithls.arith.min/max → hasAlgo=true.
emithls.func @triggers_stream_integer_algo(%arg0: !emithls.stream<i32>)
{
    %v = emithls.stream.read %arg0 : !emithls.stream<i32> -> i32
    %mn = emithls.arith.min %v, %v : i32
    %mx = emithls.arith.max %v, %v : i32
}

// arith.constant produces an index result → hasSizeT=true.
emithls.func @triggers_index(%arg0: !emithls.array<4xi32>)
{
    %idx = arith.constant 0 : index
    %v = emithls.array.read %arg0[%idx] : !emithls.array<4xi32> -> i32
}
