// RUN: laksa-opt %s --convert-dfg-to-emithls --reconcile-unrealized-casts | FileCheck %s

// CHECK-LABEL: emithls.func @pad(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>)
dfg.process @pad inputs(%in0: !dfg.output<8xi8>) outputs(%out0: !dfg.input<8xi8>) {
  %const_index_0 = emithls.variable as const index = 30
  %const_index_1 = emithls.variable as const index = -1
  %const_index_2 = emithls.variable as const index = 0
  dfg.loop inputs(%in0: !dfg.output<8xi8>) outputs(%out0: !dfg.input<8xi8>) {
    %alloc = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<8xi8>
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
          // CHECK: emithls.for %idx2 = 0 to 8 step 1 {
          // CHECK:   %[[TOK:.*]] = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
          // CHECK:   memref.store %[[TOK]], %[[BUF:.*]][%idx2] : memref<8xi8>
          // CHECK: }
          %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<8xi8>
          // CHECK: emithls.for %idx2 = 0 to 8 step 1 {
          // CHECK:   %[[VAL:.*]] = memref.load %[[BUF]][%idx2] : memref<8xi8>
          // CHECK:   emithls.stream.write %[[VAL]] to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
          // CHECK: }
          dfg.push_memref %token_memref0 to %out0 : !dfg.input<8xi8>
        } else {
          dfg.push_memref %alloc to %out0 : !dfg.input<8xi8>
        }
      }
    }
  }
}
// CHECK-LABEL: emithls.func @conv(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>)
dfg.process @conv inputs(%in0: !dfg.output<8xi8>) outputs(%out0: !dfg.input<8xi32>) {
  %const_index_0 = emithls.variable as const index = -2
  %const_index_1 = emithls.variable as const index = 0
  %const_int32_0 = emithls.variable as const i32 = 2
  %const_int32_1 = emithls.variable as const i32 = -128
  dfg.loop inputs(%in0: !dfg.output<8xi8>) outputs(%out0: !dfg.input<8xi32>) {
    emithls.for %idx0 = 0 to 32 step 1 {
      emithls.for %idx1 = 0 to 32 step 1 {
        %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<8xi8>
        %linebuf_0 = emithls.helper.linebuf %token_memref0 num_chan [8] num_line 2 keep %idx1 : memref<8xi8> -> !emithls.array<8x2x32xi8>
        %window_0 = emithls.helper.window %linebuf_0 from [%idx0, %idx1] : !emithls.array<8x2x32xi8> -> !emithls.array<8x3x3xi8>
        %expr0 = emithls.expr : i1 {
          %0 = emithls.arith.add %idx0, %const_index_0 : index
          %1 = emithls.arith.cmp ge, %0, %const_index_1 : index
          %2 = emithls.arith.add %idx1, %const_index_0 : index
          %3 = emithls.arith.cmp ge, %2, %const_index_1 : index
          %4 = emithls.arith.logical_and %1, %3 : i1
          emithls.yield %4 : i1
        }
        emithls.if %expr0 {
          emithls.for %idx2 = 0 to 8 step 1 {
            %alloc = memref.alloc() {alignment = 64 : i64, fill_with = 0 : i32} : memref<8xi32>
            emithls.for %idx3 = 0 to 3 step 1 {
              emithls.for %idx4 = 0 to 3 step 1 {
                emithls.for %idx5 = 0 to 8 step 1 {
                  %0 = emithls.array.read %window_0[%idx5, %idx3, %idx4] : !emithls.array<8x3x3xi8> -> i8
                  %1 = emithls.arith.cast %0 : i8 to i32
                  %2 = emithls.arith.sub %1, %const_int32_1 : i32
                  %3 = emithls.arith.mul %2, %const_int32_0 : i32
                  emithls.helper.accumulate %alloc at [%idx2] add %3 : memref<8xi32>
                }
              }
            }
            dfg.push_memref %alloc to %out0 : !dfg.input<8xi32>
          }
        }
      }
    }
  }
}
// CHECK-LABEL: emithls.func @relu(%arg0: !emithls.array<8x!emithls.stream<i32>>, %arg1: !emithls.array<8x!emithls.stream<i8>>)
dfg.process @relu inputs(%in0: !dfg.output<8xi32>) outputs(%out0: !dfg.input<8xi8>) {
  %const_index_0 = emithls.variable as const index = 0
  %const_array_0 = emithls.variable as const !emithls.array<8xi32> = dense<[1523322938, 1544671534, 1547120170, 1546480699, 1536582660, 1557469766, 1525058969, 1526555847]>
  %const_int32_0 = emithls.variable as const i32 = 127
  %const_int32_1 = emithls.variable as const i32 = -128
  %const_int64_0 = emithls.variable as const i64 = 41
  %const_int64_1 = emithls.variable as const i64 = -1073741824
  %const_int64_2 = emithls.variable as const i64 = 1073741824
  %const_int32_2 = emithls.variable as const i32 = 0
  %const_int64_3 = emithls.variable as const i64 = 1099511627776
  dfg.loop inputs(%in0: !dfg.output<8xi32>) outputs(%out0: !dfg.input<8xi8>) {
    emithls.for %idx0 = 0 to 30 step 1 {
      emithls.for %idx1 = 0 to 30 step 1 {
        emithls.for %idx2 = 0 to 8 step 1 {
          // CHECK: %[[TOKEN0:.*]] = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i32>> -> i32
          %token0 = dfg.pull %in0[%idx2] : !dfg.output<8xi32>
          %0 = emithls.array.read %const_array_0[%idx2] : !emithls.array<8xi32> -> i32
          %1 = emithls.arith.cast %token0 : i32 to i64
          %2 = emithls.arith.cast %0 : i32 to i64
          %3 = emithls.arith.mul %1, %2 : i64
          %4 = emithls.arith.add %3, %const_int64_3 : i64
          %5 = emithls.arith.cmp ge, %token0, %const_int32_2 : i32
          %6 = emithls.arith.select %5, %const_int64_2, %const_int64_1 : i64
          %7 = emithls.arith.add %6, %4 : i64
          %8 = emithls.arith.shr %7, %const_int64_0 : i64
          %9 = emithls.arith.cast %8 : i64 to i32
          %10 = emithls.arith.add %9, %const_int32_1 : i32
          %11 = emithls.arith.max %10, %const_int32_1 : i32
          %12 = emithls.arith.min %11, %const_int32_0 : i32
          %13 = emithls.arith.cast %12 : i32 to i8
          // CHECK: emithls.stream.write %{{.*}} to %arg1[%idx2] : i8 -> !emithls.array<8x!emithls.stream<i8>>
          dfg.push %13 to %out0[%idx2] : !dfg.input<8xi8>
        }
      }
    }
  }
}
// CHECK-LABEL: emithls.func @kernel(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) attributes {laksa.root}
// CHECK: %[[CHAN0:.*]] = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
// CHECK: %[[CHAN1:.*]] = emithls.variable as !emithls.array<8x!emithls.stream<i32>>
// CHECK: emithls.call @pad(%arg0, %[[CHAN0]])
// CHECK: emithls.call @conv(%[[CHAN0]], %[[CHAN1]])
// CHECK: emithls.call @relu(%[[CHAN1]], %arg1)
dfg.region @kernel inputs(%in0: !dfg.output<8xi8>) outputs(%out0: !dfg.input<8xi8>) attributes {laksa.root} {
  %in_port_0, %out_port_0 = dfg.channel() : 8xi8
  %in_port_1, %out_port_1 = dfg.channel() : 8xi32
  dfg.instantiate @pad inputs(%in0) outputs(%in_port_0) : (!dfg.output<8xi8>) -> !dfg.input<8xi8>
  dfg.instantiate @conv inputs(%out_port_0) outputs(%in_port_1) : (!dfg.output<8xi8>) -> !dfg.input<8xi32>
  dfg.instantiate @relu inputs(%out_port_1) outputs(%out0) : (!dfg.output<8xi32>) -> !dfg.input<8xi8>
}
