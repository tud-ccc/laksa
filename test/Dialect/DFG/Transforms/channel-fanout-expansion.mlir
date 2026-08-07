// RUN: laksa-opt %s --dfg-channel-fanout-expansion | FileCheck %s

// @producer has two outputs of distinct types: i32 at position 0 (fanned out)
// and i64 at position 1 (single consumer, untouched). After expansion the i32
// output is duplicated and inserted at position 1, pushing i64 to position 2.
// CHECK-LABEL: dfg.operator @producer
// CHECK-SAME:  outputs(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i64)
// CHECK:         dfg.output %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i64

dfg.operator @producer inputs(%in0: i32, %in1: i64) outputs(%out0: i32, %out1: i64) {
    dfg.output %in0, %in1 : i32, i64
}

// @consumer_i32 and @consumer_i64 are not modified by the pass.
// CHECK-LABEL: dfg.operator @consumer_i32
// CHECK-NOT:   outputs(
// CHECK-LABEL: dfg.operator @consumer_i64
// CHECK-NOT:   outputs(

dfg.operator @consumer_i32 inputs(%in: i32) {
    dfg.output
}

dfg.operator @consumer_i64 inputs(%in: i64) {
    dfg.output
}

// %fanout_ch (i32, position 0) output port is consumed by two @consumer_i32
// instances. The pass clones %fanout_ch and inserts it right after the original,
// leaving %single_ch (i64) in place. @producer's instantiate gains a third
// output; the i64 output moves to the last position.
//
// CHECK-LABEL: dfg.region @fanout2
// CHECK:         dfg.channel() : i32
// CHECK:         dfg.channel() : i32
// CHECK:         dfg.channel() : i64
// CHECK:         dfg.instantiate @producer
// CHECK-SAME:    outputs(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:         dfg.instantiate @consumer_i32
// CHECK:         dfg.instantiate @consumer_i32
// CHECK:         dfg.instantiate @consumer_i64

dfg.region @fanout2 inputs(%src0: !dfg.output<i32>, %src1: !dfg.output<i64>) {
    %fanout_ch:2 = dfg.channel() : i32
    %single_ch:2 = dfg.channel() : i64
    dfg.instantiate @producer inputs(%src0, %src1) outputs(%fanout_ch#0, %single_ch#0)
        : (!dfg.output<i32>, !dfg.output<i64>) -> (!dfg.input<i32>, !dfg.input<i64>)
    dfg.instantiate @consumer_i32 inputs(%fanout_ch#1)
        : (!dfg.output<i32>) -> ()
    dfg.instantiate @consumer_i32 inputs(%fanout_ch#1)
        : (!dfg.output<i32>) -> ()
    dfg.instantiate @consumer_i64 inputs(%single_ch#1)
        : (!dfg.output<i64>) -> ()
}
