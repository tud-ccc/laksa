// RUN: laksa-opt %s --canonicalize | FileCheck %s

// CHECK:      emithls.include "ap_int"
// CHECK-NEXT: emithls.include "cstdin"
// CHECK-NEXT: emithls.include "hls_stream"
// CHECK-NOT:  emithls.include
emithls.include "hls_stream"
emithls.include "cstdin"
emithls.include "ap_int"
emithls.include "ap_int"

// CHECK-LABEL: func.func @pipeline_dedup
// CHECK:       emithls.for {{.*}} = 0 to 4 step 1
// CHECK-NEXT:  emithls.pragma.pipeline II=1
// CHECK-NOT:   emithls.pragma.pipeline
func.func @pipeline_dedup(%arg0: memref<4xi32>, %arg1: memref<4xi8>)
{
    %lo = arith.constant 0 : index
    %hi = arith.constant 7 : index
    emithls.for %i = 0 to 4 step 1 {
        emithls.pragma.pipeline II=1
        emithls.for %j = 0 to 4 step 1 {
            emithls.pragma.pipeline II=1
            %v = memref.load %arg0[%i] : memref<4xi32>
            %w = emithls.arith.data_range %v(%lo, %hi) : i32 -> i8
            memref.store %w, %arg1[%i] : memref<4xi8>
        }
    }
    func.return
}

// CHECK-LABEL: emithls.func @stream_pipeline_dedup
// CHECK:       emithls.for {{.*}} = 0 to 8 step 1
// CHECK-NEXT:  emithls.pragma.pipeline II=1
// CHECK-NOT:   emithls.pragma.pipeline
emithls.func @stream_pipeline_dedup(%arg0: !emithls.stream<i8>, %arg1: !emithls.array<8x!emithls.stream<i8>>)
{
    %v = emithls.stream.read %arg0 : !emithls.stream<i8> -> i8
    emithls.for %i = 0 to 8 step 1 {
        emithls.pragma.pipeline II=1
        emithls.for %j = 0 to 8 step 1 {
            emithls.pragma.pipeline II=1
            emithls.stream.write %v to %arg1[%i] : i8 -> !emithls.array<8x!emithls.stream<i8>>
        }
    }
}

// CHECK-LABEL: emithls.func @collapse_outer_loop
// CHECK:       emithls.for {{.*}} = 0 to 8 step 1
// CHECK-NEXT:  emithls.for {{.*}} = 0 to 8 step 1
// CHECK-NOT:   emithls.for
emithls.func @collapse_outer_loop(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>)
{
    emithls.for %idx0 = 0 to 2 step 1 {
        emithls.for %idx1 = 0 to 4 step 1 {
            emithls.for %idx2 = 0 to 8 step 1 {
                %v = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
                %squre = emithls.arith.mul %v, %v : i8
                emithls.stream.write %squre to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
            }
        }
    }
}
