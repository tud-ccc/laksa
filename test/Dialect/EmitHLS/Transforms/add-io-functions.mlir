// RUN: laksa-opt %s --emithls-add-io-functions --canonicalize | FileCheck %s

// CHECK-LABEL: emithls.func @kernel_read_i8_0(%arg0: !emithls.ptr<i64>, %arg1: !emithls.array<8x!emithls.stream<i8>>)
// CHECK: emithls.for %idx0 = 0 to 900 step 1
// CHECK-NEXT: %[[WORD:.*]] = emithls.array.ptr_read %arg0[%idx0] : !emithls.ptr<i64> -> i64
// CHECK-NEXT: emithls.for %idx1 = 0 to 8 step 1
// CHECK-NEXT: %[[SLICE:.*]] = emithls.expr : i8 {
// CHECK: %{{.*}} = emithls.arith.data_range %[[WORD]](%{{.*}}, %{{.*}}) : i64 -> i8
// CHECK: }
// CHECK-NEXT: emithls.stream.write %[[SLICE]] to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
// CHECK-LABEL: emithls.func @kernel_write_i8_0(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.ptr<i64>)
// CHECK: emithls.for %idx0 = 0 to 900 step 1
// CHECK-NEXT: %[[WORDVAR:.*]] = emithls.variable as i64 = 0
// CHECK-NEXT: emithls.for %idx1 = 0 to 8 step 1
// CHECK-NEXT: %[[CHAN:.*]] = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i8>> -> i8
// CHECK-NEXT: %[[SLICE:.*]] = emithls.expr : i8 {
// CHECK: %{{.*}} = emithls.arith.data_range %[[WORDVAR]](%{{.*}}, %{{.*}}) : i64 -> i8
// CHECK: }
// CHECK-NEXT: emithls.update %[[SLICE]] with %[[CHAN]] : i8 <- i8
// CHECK: emithls.array.ptr_write %[[WORDVAR]], %arg1[%idx0] : i64 -> !emithls.ptr<i64>
// CHECK-LABEL: emithls.func @pad
// CHECK-LABEL: emithls.func @conv
// CHECK-LABEL: emithls.func @relu
// CHECK-LABEL: emithls.func @kernel(%arg0: !emithls.ptr<i64>, %arg1: !emithls.ptr<i64>)
// CHECK: %[[V0:.*]] = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
// CHECK: %[[V1:.*]] = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
// CHECK: emithls.call @kernel_read_i8_0(%arg0, %[[V0]]) : (!emithls.ptr<i64>, !emithls.array<8x!emithls.stream<i8>>) -> ()
// CHECK-NEXT: emithls.call @pad(%[[V0]], %{{.*}}) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
// CHECK-NEXT: emithls.call @conv(%{{.*}}, %{{.*}}) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
// CHECK-NEXT: emithls.call @relu(%{{.*}}, %[[V1]]) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
// CHECK-NEXT: emithls.call @kernel_write_i8_0(%[[V1]], %arg1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.ptr<i64>) -> ()
emithls.func @pad(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
  %const_index_0 = emithls.variable as const index = 30
  %const_index_1 = emithls.variable as const index = 1
  %alloc = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<8xi8>
  emithls.for %idx0 = 0 to 32 step 1 {
    emithls.for %idx1 = 0 to 32 step 1 {
      %expr0 = emithls.expr : i1 {
        %0 = emithls.arith.cmp ge, %idx0, %const_index_1 : index
        %1 = emithls.arith.cmp le, %idx0, %const_index_0 : index
        %2 = emithls.arith.cmp ge, %idx1, %const_index_1 : index
        %3 = emithls.arith.cmp le, %idx1, %const_index_0 : index
        %4 = emithls.arith.logical_and %0, %1, %2, %3 : i1
        emithls.yield %4 : i1
      }
      emithls.if %expr0 {
        emithls.for %idx2 = 0 to 8 step 1 {
          %0 = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
          emithls.stream.write %0 to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
        }
      } else {
        emithls.for %idx2 = 0 to 8 step 1 {
          %0 = memref.load %alloc[%idx2] : memref<8xi8>
          emithls.stream.write %0 to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
        }
      }
    }
  }
}
emithls.func @conv(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
  %const_index_0 = emithls.variable as const index = 2
  %const_index_1 = emithls.variable as const index = 1
  %const_int8_0 = emithls.variable as const i8 = 0
  %const_index_2 = emithls.variable as const index = 0
  %var_array_0 = emithls.variable as !emithls.array<8x3x3xi8>
  %var_array_1 = emithls.variable as !emithls.array<8x2x32xi8>
  %const_int32_0 = emithls.variable as const i32 = 3
  %const_int32_1 = emithls.variable as const i32 = -128
  emithls.for %idx0 = 0 to 32 step 1 {
    emithls.for %idx1 = 0 to 32 step 1 {
      emithls.for %idx2 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
        %expr0 = emithls.expr : i1 {
          %1 = emithls.arith.cmp eq, %idx0, %const_index_2 : index
          emithls.yield %1 : i1
        }
        emithls.for %idx3 = 0 to 3 step 1 {
          emithls.for %idx4 = 0 to 2 step 1 {
            emithls.if %expr0 {
              emithls.update %var_array_0[%idx2, %idx3, %idx4] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
            } else {
              %expr1 = emithls.expr : index {
                %1 = emithls.arith.add %idx4, %const_index_1 : index
                emithls.yield %1 : index
              }
              emithls.update %var_array_0[%idx2, %idx3, %idx4] with %var_array_0[%idx2, %idx3, %expr1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x3x3xi8>
            }
          }
        }
        %expr2 = emithls.expr : i1 {
          %1 = emithls.arith.cmp ge, %idx0, %const_index_0 : index
          emithls.yield %1 : i1
        }
        emithls.if %expr2 {
          %expr3 = emithls.expr : index {
            %1 = emithls.arith.sub %idx0, %const_index_0 : index
            %2 = emithls.arith.rem %1, %const_index_0 : index
            emithls.yield %2 : index
          }
          emithls.update %var_array_0[%idx2, %const_index_2, %const_index_0] with %var_array_1[%idx2, %expr3, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
        } else {
          emithls.update %var_array_0[%idx2, %const_index_2, %const_index_0] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
        }
        %expr4 = emithls.expr : i1 {
          %1 = emithls.arith.cmp ge, %idx0, %const_index_1 : index
          emithls.yield %1 : i1
        }
        emithls.if %expr4 {
          %expr5 = emithls.expr : index {
            %1 = emithls.arith.sub %idx0, %const_index_1 : index
            %2 = emithls.arith.rem %1, %const_index_0 : index
            emithls.yield %2 : index
          }
          emithls.update %var_array_0[%idx2, %const_index_1, %const_index_0] with %var_array_1[%idx2, %expr5, %idx1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x2x32xi8>
        } else {
          emithls.update %var_array_0[%idx2, %const_index_1, %const_index_0] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
        }
        %expr6 = emithls.expr : index {
          %1 = emithls.arith.rem %idx0, %const_index_0 : index
          emithls.yield %1 : index
        }
        emithls.update %var_array_0[%idx2, %const_index_0, %const_index_0] with %0 : !emithls.array<8x3x3xi8> <- i8
        emithls.update %var_array_1[%idx2, %expr6, %idx1] with %0 : !emithls.array<8x2x32xi8> <- i8
      }
      %expr7 = emithls.expr : i1 {
        %0 = emithls.arith.cmp ge, %idx0, %const_index_0 : index
        %1 = emithls.arith.cmp ge, %idx1, %const_index_0 : index
        %2 = emithls.arith.logical_and %0, %1 : i1
        emithls.yield %2 : i1
      }
      emithls.if %expr7 {
        emithls.for %idx2 = 0 to 8 step 1 {
          %var_int32_0 = emithls.variable as i32 = 0
          emithls.for %idx3 = 0 to 3 step 1 {
            emithls.for %idx4 = 0 to 3 step 1 {
              emithls.for %idx5 = 0 to 8 step 1 {
                %0 = emithls.array.read %var_array_0[%idx5, %idx3, %idx4] : !emithls.array<8x3x3xi8> -> i8
                %1 = emithls.arith.cast %0 : i8 to i32
                %2 = emithls.arith.sub %1, %const_int32_1 : i32
                %3 = emithls.arith.mul %2, %const_int32_0 : i32
                emithls.arith.fused add, %var_int32_0, %3 : i32
              }
            }
          }
          emithls.stream.write %var_int32_0 to %arg1[%idx2] : i32 -> !emithls.array<8x!emithls.stream<i32>>
        }
      }
    }
  }
}
emithls.func @relu(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
  %const_array_0 = emithls.variable as const !emithls.array<8xi32> = dense<[1241604906, 1329922038, 1341576575, 1322904542, 1326948123, 1339807398, 1325877483, 1340943676]>
  %const_int32_0 = emithls.variable as const i32 = 127
  %const_int32_1 = emithls.variable as const i32 = -128
  %const_int64_0 = emithls.variable as const i64 = 40
  %const_int64_1 = emithls.variable as const i64 = -1073741824
  %const_int64_2 = emithls.variable as const i64 = 1073741824
  %const_int32_2 = emithls.variable as const i32 = 0
  %const_int64_3 = emithls.variable as const i64 = 549755813888
  emithls.for %idx0 = 0 to 900 step 1 {
    emithls.for %idx1 = 0 to 8 step 1 {
      %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<8x!emithls.stream<i32>> -> i32
      %1 = emithls.array.read %const_array_0[%idx1] : !emithls.array<8xi32> -> i32
      %2 = emithls.arith.cast %0 : i32 to i64
      %3 = emithls.arith.cast %1 : i32 to i64
      %4 = emithls.arith.mul %2, %3 : i64
      %5 = emithls.arith.add %4, %const_int64_3 : i64
      %6 = emithls.arith.cmp ge, %0, %const_int32_2 : i32
      %7 = emithls.arith.select %6, %const_int64_2, %const_int64_1 : i64
      %8 = emithls.arith.add %7, %5 : i64
      %9 = emithls.arith.shr %8, %const_int64_0 : i64
      %10 = emithls.arith.cast %9 : i64 to i32
      %11 = emithls.arith.add %10, %const_int32_1 : i32
      %12 = emithls.arith.max %11, %const_int32_1 : i32
      %13 = emithls.arith.min %12, %const_int32_0 : i32
      %14 = emithls.arith.cast %13 : i32 to i8
      emithls.stream.write %14 to %arg1[%idx1] : i8 -> !emithls.array<8x!emithls.stream<i8>>
    }
  }
}
emithls.func @kernel(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
  %var_array_0 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
  %var_array_1 = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
  emithls.call @pad(%arg0, %var_array_0) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
  emithls.call @conv(%var_array_0, %var_array_1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i32>>) -> ()
  emithls.call @relu(%var_array_1, %arg1) : (!emithls.array<8x!emithls.stream<i32>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
}