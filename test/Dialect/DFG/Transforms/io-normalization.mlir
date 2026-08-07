// RUN: laksa-opt %s --dfg-io-normalization | FileCheck %s

#set = affine_set<(d0, d1) : (d0 - 2 >= 0, d1 - 2 >= 0)>
memref.global "private" constant @__constant_8xi32 : memref<8xi32> = dense<[1241604906, 1329922038, 1341576575, 1322904542, 1326948123, 1339807398, 1325877483, 1340943676]> {alignment = 64 : i64}

// CHECK-LABEL: dfg.process @pad inputs(%in0: !dfg.output<1x8xi8>) outputs(%out0: !dfg.input<1x8xi8>)
dfg.process @pad inputs(%in0: !dfg.output<1x30x30x8xi8>) outputs(%out0: !dfg.input<1x32x32x8xi8>) {
  dfg.loop inputs(%in0: !dfg.output<1x30x30x8xi8>) outputs(%out0: !dfg.input<1x32x32x8xi8>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x30x30x8xi8>
    %alloc = memref.alloc() {alignment = 64 : i64, fill_with = -128 : i8} : memref<1x32x32x8xi8>
    %subview = memref.subview %alloc[0, 1, 1, 0] [1, 30, 30, 8] [1, 1, 1, 1] : memref<1x32x32x8xi8> to memref<1x30x30x8xi8, strided<[8192, 256, 8, 1], offset: 264>>
    memref.copy %token_memref0, %subview : memref<1x30x30x8xi8> to memref<1x30x30x8xi8, strided<[8192, 256, 8, 1], offset: 264>>
    dfg.push_memref %alloc to %out0 : !dfg.input<1x32x32x8xi8>
  }
}
// CHECK-LABEL: dfg.process @conv inputs(%in0: !dfg.output<1x8xi8>) outputs(%out0: !dfg.input<1x8xi32>)
// CHECK: memref.alloc() {alignment = 64 : i64, fill_with = 0 : i32} : memref<1x8xi32>
dfg.process @conv inputs(%in0: !dfg.output<1x32x32x8xi8>) outputs(%out0: !dfg.input<1x30x30x8xi32>) {
  %c3_i32 = arith.constant 3 : i32
  %c-128_i32 = arith.constant -128 : i32
  dfg.loop inputs(%in0: !dfg.output<1x32x32x8xi8>) outputs(%out0: !dfg.input<1x30x30x8xi32>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x32x32x8xi8>
    %alloc = memref.alloc() {accu_at = [0 : i32, 3 : i32], alignment = 64 : i64, fill_with = 0 : i32} : memref<1x30x30x8xi32>
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %linebuf_0 = emithls.helper.linebuf %token_memref0 num_chan [1, 8] num_line 2 keep %arg1 : memref<1x32x32x8xi8> -> memref<1x8x2x32xi8>
        %window_0 = emithls.helper.window %linebuf_0 from [%arg0, %arg1] : memref<1x8x2x32xi8> -> memref<1x8x3x3xi8>
        affine.if #set(%arg0, %arg1) {
          affine.for %arg2 = 0 to 1 {
            affine.for %arg3 = 0 to 8 {
              affine.for %arg4 = 0 to 3 {
                affine.for %arg5 = 0 to 3 {
                  affine.for %arg6 = 0 to 8 {
                    %0 = affine.load %window_0[%arg2, %arg6, %arg4, %arg5] : memref<1x8x3x3xi8>
                    %1 = arith.extsi %0 : i8 to i32
                    %2 = arith.subi %1, %c-128_i32 : i32
                    %3 = arith.muli %2, %c3_i32 : i32
                    emithls.helper.accumulate %alloc at [%arg2, %arg3] add %3 : memref<1x30x30x8xi32>
                  }
                }
              }
            }
          }
        }
      }
    } 
    dfg.push_memref %alloc to %out0 : !dfg.input<1x30x30x8xi32>
  }
}
// CHECK-LABEL: dfg.process @relu inputs(%in0: !dfg.output<1x8xi32>) outputs(%out0: !dfg.input<1x8xi8>)
// CHECK-NOT: memref.alloc
dfg.process @relu inputs(%in0: !dfg.output<1x30x30x8xi32>) outputs(%out0: !dfg.input<1x30x30x8xi8>) {
  %c127_i32 = arith.constant 127 : i32
  %c-128_i32 = arith.constant -128 : i32
  %c40_i64 = arith.constant 40 : i64
  %c-1073741824_i64 = arith.constant -1073741824 : i64
  %c1073741824_i64 = arith.constant 1073741824 : i64
  %c0_i32 = arith.constant 0 : i32
  %c549755813888_i64 = arith.constant 549755813888 : i64
  dfg.loop inputs(%in0: !dfg.output<1x30x30x8xi32>) outputs(%out0: !dfg.input<1x30x30x8xi8>) {
    %token_memref0 = dfg.pull_as_memref %in0 : !dfg.output<1x30x30x8xi32>
    %0 = memref.get_global @__constant_8xi32 : memref<8xi32>
    %alloc = memref.alloc() {alignment = 64 : i64, parallel} : memref<1x30x30x8xi8>
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 30 {
        affine.for %arg2 = 0 to 30 {
          affine.for %arg3 = 0 to 8 {
            %1 = affine.load %token_memref0[%arg0, %arg1, %arg2, %arg3] : memref<1x30x30x8xi32>
            %2 = affine.load %0[%arg3] : memref<8xi32>
            %3 = arith.extsi %1 : i32 to i64
            %4 = arith.extsi %2 : i32 to i64
            %5 = arith.muli %3, %4 : i64
            %6 = arith.addi %5, %c549755813888_i64 : i64
            %7 = arith.cmpi sge, %1, %c0_i32 : i32
            %8 = arith.select %7, %c1073741824_i64, %c-1073741824_i64 : i64
            %9 = arith.addi %8, %6 : i64
            %10 = arith.shrsi %9, %c40_i64 : i64
            %11 = arith.trunci %10 : i64 to i32
            %12 = arith.addi %11, %c-128_i32 : i32
            %13 = arith.maxsi %12, %c-128_i32 : i32
            %14 = arith.minsi %13, %c127_i32 : i32
            %15 = arith.trunci %14 : i32 to i8
            affine.store %15, %alloc[%arg0, %arg1, %arg2, %arg3] : memref<1x30x30x8xi8>
          }
        }
      }
    }
    dfg.push_memref %alloc to %out0 : !dfg.input<1x30x30x8xi8>
  }
}

// CHECK-LABEL: dfg.region @kernel inputs(%in0 : !dfg.output<1x8xi8>)  outputs(%out0 : !dfg.input<1x8xi8>)
dfg.region @kernel inputs(%in0 : !dfg.output<1x30x30x8xi8>)  outputs(%out0 : !dfg.input<1x30x30x8xi8>)  {
  %in_port_0, %out_port_0 = dfg.channel() : 1x32x32x8xi8
  %in_port_1, %out_port_1 = dfg.channel() : 1x30x30x8xi32
  dfg.instantiate @pad inputs(%in0) outputs(%in_port_0) : (!dfg.output<1x30x30x8xi8>) -> !dfg.input<1x32x32x8xi8>
  dfg.instantiate @conv inputs(%out_port_0) outputs(%in_port_1) : (!dfg.output<1x32x32x8xi8>) -> !dfg.input<1x30x30x8xi32>
  dfg.instantiate @relu inputs(%out_port_1) outputs(%out0) : (!dfg.output<1x30x30x8xi32>) -> !dfg.input<1x30x30x8xi8>
}
