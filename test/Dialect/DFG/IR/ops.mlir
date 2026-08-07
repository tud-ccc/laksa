// RUN: laksa-opt %s | FileCheck %s

// CHECK-LABEL: dfg.process @basic_process
dfg.process @basic_process inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    // CHECK: dfg.loop
    dfg.loop inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
    {
        // CHECK: dfg.pull %{{.*}} : !dfg.output<i32>
        %0 = dfg.pull %in : !dfg.output<i32>
        // CHECK: dfg.push %{{.*}} to %{{.*}} : !dfg.input<i32>
        dfg.push %0 to %out : !dfg.input<i32>
    }
}

// CHECK-LABEL: dfg.process @shaped_process
dfg.process @shaped_process inputs(%in: !dfg.output<2xi32>) outputs(%out: !dfg.input<2xi32>)
{
    %idx = arith.constant 0 : index
    // CHECK: dfg.pull %{{.*}}[%{{.*}}] : !dfg.output<2xi32>
    %elem = dfg.pull %in[%idx] : !dfg.output<2xi32>
    // CHECK: dfg.pull_as_tensor %{{.*}} : !dfg.output<2xi32>
    %tensor = dfg.pull_as_tensor %in : !dfg.output<2xi32>
    // CHECK: dfg.pull_as_memref %{{.*}} : !dfg.output<2xi32>
    %mref = dfg.pull_as_memref %in : !dfg.output<2xi32>
    // CHECK: dfg.push_tensor %{{.*}} to %{{.*}} : !dfg.input<2xi32>
    dfg.push_tensor %tensor to %out : !dfg.input<2xi32>
    // CHECK: dfg.push_memref %{{.*}} to %{{.*}} : !dfg.input<2xi32>
    dfg.push_memref %mref to %out : !dfg.input<2xi32>
    // CHECK: dfg.push %{{.*}} to %{{.*}}[%{{.*}}] : !dfg.input<2xi32>
    dfg.push %elem to %out[%idx] : !dfg.input<2xi32>
}

// CHECK-LABEL: dfg.operator @basic_operator
dfg.operator @basic_operator inputs(%in: i32) outputs(%out: i32)
{
    // CHECK: dfg.output %{{.*}} : i32
    dfg.output %in : i32
}

// CHECK-LABEL: dfg.region @basic_region
dfg.region @basic_region inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    // CHECK: dfg.channel() : i32
    %0:2 = dfg.channel() : i32
    // CHECK: dfg.instantiate @basic_process
    dfg.instantiate @basic_process inputs(%in) outputs(%0#0) : (!dfg.output<i32>) -> !dfg.input<i32>
    dfg.instantiate @basic_process inputs(%0#1) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
}

// CHECK-LABEL: dfg.region @offloaded_region
dfg.region @offloaded_region inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    // CHECK: dfg.instantiate offloaded=mdc @basic_process
    dfg.instantiate offloaded=mdc @basic_process inputs(%in) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
}

// CHECK-LABEL: dfg.region @embed_region
dfg.region @embed_region outputs(%out: !dfg.input<i32>)
{
    // CHECK: dfg.channel(2) : i32
    %0, %1 = dfg.channel (2) : i32
    dfg.instantiate @basic_process inputs(%1) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
    // CHECK: dfg.embed @basic_region
    dfg.embed @basic_region inputs(%1) outputs(%0) : (!dfg.output<i32>) -> !dfg.input<i32>
}
