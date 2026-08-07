// RUN: laksa-opt %s --emithls-resolve-helpers --canonicalize | FileCheck %s


// CHECK-LABEL: emithls.func @conv_2d
// CHECK-DAG: %[[BUF:.*]] = emithls.variable as !emithls.array<8x2x32xi8>
// CHECK-DAG: %[[WIN:.*]] = emithls.variable as !emithls.array<8x3x3xi8>
emithls.func @conv_2d(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
  %const_index_0 = emithls.variable as const index = -2
  %const_index_1 = emithls.variable as const index = 0
  %const_int32_0 = emithls.variable as const i32 = 2
  %const_int32_1 = emithls.variable as const i32 = -128
  emithls.for %idx0 = 0 to 32 step 1 {
    emithls.for %idx1 = 0 to 32 step 1 {
      %alloc = memref.alloc() : memref<8xi8>
      emithls.for %idx2 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
        memref.store %0, %alloc[%idx2] : memref<8xi8>
      }
      // CHECK: emithls.for %idx2 = 0 to 8 step 1 {
      // CHECK:   %[[TOK:.*]] = memref.load %alloc[%idx2] : memref<8xi8>
      // CHECK:   %[[FIRSTROW:.*]] = emithls.expr : i1 {
      // CHECK:     emithls.arith.cmp eq, %idx0, %{{.*}} : index
      // CHECK:   }
      // CHECK:   emithls.for %idx3 = 0 to 3 step 1 {
      // CHECK:     emithls.for %idx4 = 0 to 2 step 1 {
      // CHECK:       emithls.if %[[FIRSTROW]] {
      // CHECK:         emithls.update %[[WIN]][%idx2, %idx3, %idx4] with %{{.*}} : !emithls.array<8x3x3xi8> <- i8
      // CHECK:       } else {
      // CHECK:         %[[SHIFTCOL:.*]] = emithls.expr : index {
      // CHECK:           emithls.arith.add %idx4, %{{.*}} : index
      // CHECK:         }
      // CHECK:         emithls.update %[[WIN]][%idx2, %idx3, %idx4] with %[[WIN]][%idx2, %idx3, %[[SHIFTCOL]]] : !emithls.array<8x3x3xi8> <- !emithls.array<8x3x3xi8>
      // CHECK:       }
      // CHECK:     }
      // CHECK:   }
      // CHECK:   emithls.if %{{.*}} {
      // CHECK:     %[[SLOT0:.*]] = emithls.expr : index {
      // CHECK:       %{{.*}} = emithls.arith.sub %idx0, %{{.*}} : index
      // CHECK:       %{{.*}} = emithls.arith.rem %{{.*}}, %{{.*}} : index
      // CHECK:     }
      // CHECK:     emithls.update %[[WIN]][%idx2, %{{.*}}, %{{.*}}] with %[[BUF]][%idx2, %[[SLOT0]], %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
      // CHECK:   } else {
      // CHECK:     emithls.update %[[WIN]][%idx2, %{{.*}}, %{{.*}}] with %{{.*}} : !emithls.array<8x3x3xi8> <- i8
      // CHECK:   }
      // CHECK:   emithls.if %{{.*}} {
      // CHECK:     %[[SLOT1:.*]] = emithls.expr : index {
      // CHECK:       %{{.*}} = emithls.arith.sub %idx0, %{{.*}} : index
      // CHECK:       %{{.*}} = emithls.arith.rem %{{.*}}, %{{.*}} : index
      // CHECK:     }
      // CHECK:     emithls.update %[[WIN]][%idx2, %{{.*}}, %{{.*}}] with %[[BUF]][%idx2, %[[SLOT1]], %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
      // CHECK:   } else {
      // CHECK:     emithls.update %[[WIN]][%idx2, %{{.*}}, %{{.*}}] with %{{.*}} : !emithls.array<8x3x3xi8> <- i8
      // CHECK:   }
      // CHECK:   %[[CURSLOT:.*]] = emithls.expr : index {
      // CHECK:     emithls.arith.rem %idx0, %{{.*}} : index
      // CHECK:   }
      // CHECK:   emithls.update %[[WIN]][%idx2, %{{.*}}, %{{.*}}] with %[[TOK]] : !emithls.array<8x3x3xi8> <- i8
      // CHECK:   emithls.update %[[BUF]][%idx2, %[[CURSLOT]], %idx1] with %[[TOK]] : !emithls.array<8x2x32xi8> <- i8
      // CHECK: }
      %linebuf_0 = emithls.helper.linebuf %alloc num_chan [8] num_line 2 keep %idx1 : memref<8xi8> -> !emithls.array<8x2x32xi8>
      // CHECK-NOT: emithls.helper.linebuf
      %window_0 = emithls.helper.window %linebuf_0 from [%idx0, %idx1] : !emithls.array<8x2x32xi8> -> !emithls.array<8x3x3xi8>
      // CHECK-NOT: emithls.helper.window
      %expr0 = emithls.expr : i1 {
        %0 = emithls.arith.add %idx0, %const_index_0 : index
        %1 = emithls.arith.cmp ge, %0, %const_index_1 : index
        %2 = emithls.arith.add %idx1, %const_index_0 : index
        %3 = emithls.arith.cmp ge, %2, %const_index_1 : index
        %4 = emithls.arith.logical_and %1, %3 : i1
        emithls.yield %4 : i1
      }
      // CHECK: emithls.if %{{.*}} {
      // CHECK:   emithls.for %idx2 = 0 to 8 step 1 {
      // CHECK:     %[[ACC:.*]] = emithls.variable as i32 = 0
      // CHECK:     emithls.for %idx3 = 0 to 3 step 1 {
      // CHECK:       emithls.for %idx4 = 0 to 3 step 1 {
      // CHECK:         emithls.for %idx5 = 0 to 8 step 1 {
      // CHECK:           %[[ELEM:.*]] = emithls.array.read %[[WIN]][%idx5, %idx3, %idx4] : !emithls.array<8x3x3xi8> -> i8
      // CHECK:           %[[CASTED:.*]] = emithls.arith.cast %[[ELEM]] : i8 to i32
      // CHECK:           %[[SUB:.*]] = emithls.arith.sub %[[CASTED]], %const_int32_1 : i32
      // CHECK:           %[[MUL:.*]] = emithls.arith.mul %[[SUB]], %const_int32_0 : i32
      // CHECK:           emithls.arith.fused add, %[[ACC]], %[[MUL]] : i32
      // CHECK-NOT:        emithls.helper.accumulate
      // CHECK:         }
      // CHECK:       }
      // CHECK:     }
      // CHECK:     emithls.stream.write %[[ACC]] to %arg1[%idx2] : i32 -> !emithls.array<8x!emithls.stream<i32>>
      // CHECK:   }
      // CHECK: }
      emithls.if %expr0 {
        emithls.for %idx2 = 0 to 8 step 1 {
          %alloc_0 = memref.alloc() {alignment = 64 : i64, fill_with = 0 : i32} : memref<8xi32>
          emithls.for %idx3 = 0 to 3 step 1 {
            emithls.for %idx4 = 0 to 3 step 1 {
              emithls.for %idx5 = 0 to 8 step 1 {
                %0 = emithls.array.read %window_0[%idx5, %idx3, %idx4] : !emithls.array<8x3x3xi8> -> i8
                %1 = emithls.arith.cast %0 : i8 to i32
                %2 = emithls.arith.sub %1, %const_int32_1 : i32
                %3 = emithls.arith.mul %2, %const_int32_0 : i32
                emithls.helper.accumulate %alloc_0 at [%idx2] add %3 : memref<8xi32>
              }
            }
          }
          emithls.for %idx3 = 0 to 8 step 1 {
            %0 = memref.load %alloc_0[%idx3] : memref<8xi32>
            emithls.stream.write %0 to %arg1[%idx3] : i32 -> !emithls.array<8x!emithls.stream<i32>>
          }
        }
      }
    }
  }
}

// CHECK-LABEL: emithls.func @matmul
// CHECK: %[[PUREBUF:.*]] = emithls.variable as !emithls.array<128xi8>
emithls.func @matmul(%arg0: !emithls.array<128x!emithls.stream<i8>>, %arg1: !emithls.array<256x!emithls.stream<i32>>) {
  %const_int32_0 = emithls.variable as const i32 = -128
  emithls.for %idx0 = 0 to 512 step 1 {
    %alloc = memref.alloc() : memref<128xi8>
    emithls.for %idx1 = 0 to 128 step 1 {
      %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<128x!emithls.stream<i8>> -> i8
      memref.store %0, %alloc[%idx1] : memref<128xi8>
    }
    // CHECK: emithls.for %idx1 = 0 to 128 step 1 {
    // CHECK:   %[[TOK:.*]] = memref.load %alloc[%idx1] : memref<128xi8>
    // CHECK:   emithls.update %[[PUREBUF]][%idx1] with %[[TOK]] : !emithls.array<128xi8> <- i8
    // CHECK: }
    %linebuf_0 = emithls.helper.linebuf %alloc num_chan [] num_line 0 keep %idx0 : memref<128xi8> -> !emithls.array<128xi8>
    // CHECK-NOT: emithls.helper.linebuf
    // CHECK: emithls.for %idx1 = 0 to 256 step 1 {
    // CHECK:   %[[ACC:.*]] = emithls.variable as i32 = 0
    // CHECK:   emithls.for %idx2 = 0 to 128 step 1 {
    // CHECK:     %[[ELEM:.*]] = emithls.array.read %[[PUREBUF]][%idx2] : !emithls.array<128xi8> -> i8
    // CHECK:     %[[CASTED:.*]] = emithls.arith.cast %[[ELEM]] : i8 to i32
    // CHECK:     %[[SUB:.*]] = emithls.arith.sub %[[CASTED]], %const_int32_0 : i32
    // CHECK:     %[[MUL:.*]] = emithls.arith.mul %[[SUB]], %const_int32_0 : i32
    // CHECK:     emithls.arith.fused add, %[[ACC]], %[[MUL]] : i32
    // CHECK-NOT:  emithls.helper.accumulate
    // CHECK:   }
    // CHECK:   emithls.stream.write %[[ACC]] to %arg1[%idx1] : i32 -> !emithls.array<256x!emithls.stream<i32>>
    // CHECK: }
    emithls.for %idx1 = 0 to 256 step 1 {
      %alloc_0 = memref.alloc() {alignment = 64 : i64, fill_with = 0 : i32} : memref<256xi32>
      emithls.for %idx2 = 0 to 128 step 1 {
        %0 = emithls.array.read %linebuf_0[%idx2] : !emithls.array<128xi8> -> i8
        %1 = emithls.arith.cast %0 : i8 to i32
        %2 = emithls.arith.sub %1, %const_int32_0 : i32
        %3 = emithls.arith.mul %2, %const_int32_0 : i32
        emithls.helper.accumulate %alloc_0 at [%idx1] add %3 : memref<256xi32>
      }
      emithls.for %idx2 = 0 to 256 step 1 {
        %0 = memref.load %alloc_0[%idx2] : memref<256xi32>
        emithls.stream.write %0 to %arg1[%idx2] : i32 -> !emithls.array<256x!emithls.stream<i32>>
      }
    }
  }
}

