// RUN: laksa-opt %s --dfg-operator-to-process | FileCheck %s

// Scalar operator: dfg.output becomes dfg.push inside a dfg.loop.
// CHECK-NOT: dfg.operator @scalar_op
// CHECK-LABEL: dfg.process @scalar_op
// CHECK:       dfg.loop
// CHECK:         dfg.pull %{{.*}} : !dfg.output<i32>
// CHECK:         dfg.push %{{.*}} to %{{.*}} : !dfg.input<i32>
// CHECK:         dfg.push %{{.*}} to %{{.*}} : !dfg.input<i32>
dfg.operator @scalar_op inputs(%in: i32) outputs(%out0: i32, %out1: i32)
{
    dfg.output %in, %in : i32, i32
}

// Memref operator: pull_as_memref / push_memref are used.
// CHECK-NOT: dfg.operator @memref_op
// CHECK-LABEL: dfg.process @memref_op
// CHECK:       dfg.loop
// CHECK:         dfg.pull_as_memref %{{.*}} : !dfg.output<4xi32>
// CHECK:         dfg.push_memref %{{.*}} to %{{.*}} : !dfg.input<4xi32>
dfg.operator @memref_op inputs(%in: memref<4xi32>) outputs(%out: memref<4xi32>)
{
    dfg.output %in : memref<4xi32>
}

// Tensor operator: pull_as_tensor / push_tensor are used.
// CHECK-NOT: dfg.operator @tensor_op
// CHECK-LABEL: dfg.process @tensor_op
// CHECK:       dfg.loop
// CHECK:         dfg.pull_as_tensor %{{.*}} : !dfg.output<4xi32>
// CHECK:         dfg.push_tensor %{{.*}} to %{{.*}} : !dfg.input<4xi32>
dfg.operator @tensor_op inputs(%in: tensor<4xi32>) outputs(%out: tensor<4xi32>)
{
    dfg.output %in : tensor<4xi32>
}
