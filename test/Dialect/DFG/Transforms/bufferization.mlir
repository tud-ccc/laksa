// RUN: laksa-opt %s --one-shot-bufferize | FileCheck %s

// pull_as_tensor is lowered to pull_as_memref and push_tensor to push_memref.
// CHECK-LABEL: dfg.process @identity
// CHECK:       dfg.pull_as_memref %{{.*}} : !dfg.output<4xi32>
// CHECK-NOT:   dfg.pull_as_tensor
// CHECK:       dfg.push_memref %{{.*}} to %{{.*}} : !dfg.input<4xi32>
// CHECK-NOT:   dfg.push_tensor
dfg.process @identity inputs(%in: !dfg.output<4xi32>) outputs(%out: !dfg.input<4xi32>)
{
    %0 = dfg.pull_as_tensor %in : !dfg.output<4xi32>
    dfg.push_tensor %0 to %out : !dfg.input<4xi32>
}

// A process with only scalar pull/push is unchanged by bufferization.
// CHECK-LABEL: dfg.process @passthrough
// CHECK:       dfg.pull %{{.*}} : !dfg.output<i32>
// CHECK:       dfg.push %{{.*}} to %{{.*}} : !dfg.input<i32>
dfg.process @passthrough inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    dfg.loop inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
    {
        %0 = dfg.pull %in : !dfg.output<i32>
        dfg.push %0 to %out : !dfg.input<i32>
    }
}
