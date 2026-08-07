// RUN: laksa-opt %s --dfg-inline-embed-region | FileCheck %s

dfg.process @source outputs(%out: !dfg.input<i32>)
{
    %0 = arith.constant 0 : i32
    dfg.push %0 to %out : !dfg.input<i32>
}

dfg.process @relay inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    %0 = dfg.pull %in : !dfg.output<i32>
    dfg.push %0 to %out : !dfg.input<i32>
}

// @child is referenced by two embed ops; it must be erased once both are
// inlined.
// CHECK-NOT: dfg.region @child
dfg.region @child outputs(%out: !dfg.input<i32>)
{
    %0:2 = dfg.channel() : i32
    dfg.instantiate @source outputs(%0#0) : () -> !dfg.input<i32>
    dfg.instantiate @relay inputs(%0#1) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
}

// @parent embeds @child: its body should contain the inlined channel and
// instantiations from @child alongside its own instantiate op.
// CHECK-LABEL: dfg.region @parent
// CHECK-NOT:   dfg.embed
// CHECK:       dfg.instantiate @relay
// CHECK:       dfg.channel() : i32
// CHECK:       dfg.instantiate @source
// CHECK:       dfg.instantiate @relay
dfg.region @parent inputs(%in: !dfg.output<i32>) outputs(%out0: !dfg.input<i32>, %out1: !dfg.input<i32>)
{
    dfg.instantiate @relay inputs(%in) outputs(%out1) : (!dfg.output<i32>) -> !dfg.input<i32>
    dfg.embed @child outputs(%out0) : () -> !dfg.input<i32>
}

// @standalone has only an embed; after inlining the region body is taken
// directly from @child.
// CHECK-LABEL: dfg.region @standalone
// CHECK-NOT:   dfg.embed
// CHECK:       dfg.channel() : i32
// CHECK:       dfg.instantiate @source
// CHECK:       dfg.instantiate @relay
dfg.region @standalone outputs(%out: !dfg.input<i32>)
{
    dfg.embed @child outputs(%out) : () -> !dfg.input<i32>
}
