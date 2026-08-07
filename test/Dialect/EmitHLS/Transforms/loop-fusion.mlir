// RUN: laksa-opt %s --emithls-loop-fusion --canonicalize | FileCheck %s

// CHECK-LABEL: emithls.func @pass_through
// CHECK: emithls.for %idx0 = 0 to 512 step 1 {
// CHECK-NEXT: emithls.for %idx1 = 0 to 128 step 1 {
// CHECK-NEXT: %[[TOK:.*]] = emithls.stream.read %arg0[%idx1] : !emithls.array<128x!emithls.stream<i8>> -> i8
// CHECK-NEXT: emithls.update %var_array_0[%idx1] with %[[TOK]] : !emithls.array<128xi8> <- i8
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
emithls.func @pass_through(%arg0: !emithls.array<128x!emithls.stream<i8>>, %arg1: !emithls.array<256x!emithls.stream<i32>>) {
  %var_array_0 = emithls.variable as !emithls.array<128xi8>
  emithls.for %idx0 = 0 to 512 step 1 {
    %alloc = memref.alloc() : memref<128xi8>
    emithls.for %idx1 = 0 to 128 step 1 {
      %0 = emithls.stream.read %arg0[%idx1] : !emithls.array<128x!emithls.stream<i8>> -> i8
      memref.store %0, %alloc[%idx1] : memref<128xi8>
    }
    emithls.for %idx1 = 0 to 128 step 1 {
      %0 = memref.load %alloc[%idx1] : memref<128xi8>
      emithls.update %var_array_0[%idx1] with %0 : !emithls.array<128xi8> <- i8
    }
  }
}
// CHECK-LABEL: emithls.func @head_toe
// CHECK: emithls.for %idx0 = 0 to 32 step 1 {
// CHECK-NEXT: emithls.for %idx1 = 0 to 32 step 1 {
// CHECK-NEXT: emithls.for %idx2 = 0 to 8 step 1 {
// CHECK-NEXT: %[[TOK:.*]] = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
// CHECK: emithls.update %var_array_0[%idx2, %const_index_2, %const_index_2] with %[[TOK]] : !emithls.array<8x3x3xi8> <- i8
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
emithls.func @head_toe(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i32>>) {
  %var_array_0 = emithls.variable as !emithls.array<8x3x3xi8>
  %const_int8_0 = emithls.variable as const i8 = 0
  %const_index_0 = emithls.variable as const index = 1
  %const_index_1 = emithls.variable as const index = 0
  %const_index_2 = emithls.variable as const index = 2
  emithls.for %idx0 = 0 to 32 step 1 {
    emithls.for %idx1 = 0 to 32 step 1 {
      %alloc = memref.alloc() : memref<8xi8>
      emithls.for %idx2 = 0 to 8 step 1 {
        %0 = emithls.stream.read %arg0[%idx2] : !emithls.array<8x!emithls.stream<i8>> -> i8
        memref.store %0, %alloc[%idx2] : memref<8xi8>
      }
      emithls.for %idx2 = 0 to 8 step 1 {
        %0 = memref.load %alloc[%idx2] : memref<8xi8>
        %expr0 = emithls.expr : i1 {
          %1 = emithls.arith.cmp eq, %idx0, %const_index_1 : index
          emithls.yield %1 : i1
        }
        emithls.for %idx3 = 0 to 3 step 1 {
          emithls.for %idx4 = 0 to 2 step 1 {
            emithls.if %expr0 {
              emithls.update %var_array_0[%idx2, %idx3, %idx4] with %const_int8_0 : !emithls.array<8x3x3xi8> <- i8
            } else {
              %expr1 = emithls.expr : index {
                %1 = emithls.arith.add %idx4, %const_index_0 : index
                emithls.yield %1 : index
              }
              emithls.update %var_array_0[%idx2, %idx3, %idx4] with %var_array_0[%idx2, %idx3, %expr1] : !emithls.array<8x3x3xi8> <- !emithls.array<8x3x3xi8>
            }
          }
        }
        emithls.update %var_array_0[%idx2, %const_index_2, %const_index_2] with %0 : !emithls.array<8x3x3xi8> <- i8
      }
    }
  }
}
