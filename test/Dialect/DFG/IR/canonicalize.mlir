// RUN: laksa-opt %s --canonicalize | FileCheck %s

// Processes and operators with no instantiation sites are erased.
// CHECK-NOT: dfg.process @dead_process
dfg.process @dead_process
// CHECK-NOT: dfg.operator @dead_operator
dfg.operator @dead_operator

// CHECK-LABEL: dfg.process @relay
dfg.process @relay inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    dfg.loop inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
    {
        %0 = dfg.pull %in : !dfg.output<i32>
        dfg.push %0 to %out : !dfg.input<i32>
    }
}

// Canonicalization reorders region bodies so all channel declarations appear
// before instantiate/embed ops.
// CHECK-LABEL: dfg.region @pipeline
// CHECK:       dfg.channel() : i32
// CHECK-NEXT:  dfg.channel() : i32
// CHECK:       dfg.instantiate @relay
// CHECK:       dfg.instantiate @relay
// CHECK:       dfg.instantiate @relay
dfg.region @pipeline inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    %0:2 = dfg.channel() : i32
    dfg.instantiate @relay inputs(%in) outputs(%0#0) : (!dfg.output<i32>) -> !dfg.input<i32>
    %1:2 = dfg.channel() : i32
    dfg.instantiate @relay inputs(%0#1) outputs(%1#0) : (!dfg.output<i32>) -> !dfg.input<i32>
    dfg.instantiate @relay inputs(%1#1) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
}
