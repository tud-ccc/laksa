// RUN: laksa-opt %s | FileCheck %s

// CHECK-LABEL: emithls.func @basic_vars
emithls.func @basic_vars(%arg0: !emithls.ptr<i32>)
{
    // CHECK: emithls.variable as const i32 = 0
    %0 = emithls.variable as const i32 = 0
    // CHECK: emithls.variable as i32
    %1 = emithls.variable as i32 = %0
    // CHECK: emithls.for {{.*}} = 0 to 4 step 1
    emithls.for %i = 0 to 4 step 1 {
        // CHECK: emithls.pragma.pipeline II=1
        emithls.pragma.pipeline II=1
        // CHECK: emithls.array.ptr_read
        %val = emithls.array.ptr_read %arg0[%i] : !emithls.ptr<i32> -> i32
        // CHECK: emithls.array.ptr_write
        emithls.array.ptr_write %val, %arg0[%i] : i32 -> !emithls.ptr<i32>
    }
}

// CHECK-LABEL: emithls.func @array_ops
emithls.func @array_ops(%arg0: !emithls.array<8xi32>)
{
    emithls.for %i = 0 to 8 step 1 {
        // CHECK: emithls.array.read
        %val = emithls.array.read %arg0[%i] : !emithls.array<8xi32> -> i32
        // CHECK: emithls.array.write
        emithls.array.write %val, %arg0[%i] : i32 -> !emithls.array<8xi32>
    }
}

// CHECK-LABEL: emithls.func @stream_ops
emithls.func @stream_ops(%arg0: !emithls.stream<i32>, %arg1: !emithls.array<4x!emithls.stream<i32>>)
{
    // CHECK: emithls.stream.read
    %val = emithls.stream.read %arg0 : !emithls.stream<i32> -> i32
    emithls.for %i = 0 to 4 step 1 {
        // CHECK: emithls.stream.write
        emithls.stream.write %val to %arg1[%i] : i32 -> !emithls.array<4x!emithls.stream<i32>>
    }
    // CHECK: emithls.variable as !emithls.stream<i32>
    %s = emithls.variable as !emithls.stream<i32>
    // CHECK: emithls.pragma.stream
    emithls.pragma.stream variable=%s(!emithls.stream<i32>) depth=8
}

// CHECK-LABEL: emithls.func @pragma_ops
emithls.func @pragma_ops(%arg0: !emithls.array<16xi32>)
{
    // CHECK: emithls.pragma.inline
    emithls.pragma.inline
    // CHECK: emithls.pragma.array_partition
    emithls.pragma.array_partition variable=%arg0(!emithls.array<16xi32>) type=cyclic factor=2 dim=1
    // CHECK: emithls.pragma.bind_storage
    emithls.pragma.bind_storage variable=%arg0(!emithls.array<16xi32>) type=ram_2p impl=bram
}
