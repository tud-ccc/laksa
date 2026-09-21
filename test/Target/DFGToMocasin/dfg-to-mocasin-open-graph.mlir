// RUN: laksa-translate %s --dfg-to-mocasin | FileCheck %s

// Top-level region ports become one synthetic source/sink process per port.

// CHECK: graph:
// CHECK-NEXT: processes:
// CHECK: parent-in0-source:
// CHECK: parent-out0-sink:
// CHECK: channels:
// CHECK-NEXT: ch0:
// CHECK-NEXT:   src:{{ *}}{ process: child.relay, port: out0 }
// CHECK-NEXT:   dst:{{ *}}{ process: relay, port: in0 }
// CHECK-NEXT:   token_size:{{ *}}4
// CHECK-NEXT: ch1:
// CHECK-NEXT:   src:{{ *}}{ process: parent-in0-source, port: out0 }
// CHECK-NEXT:   dst:{{ *}}{ process: child.relay, port: in0 }
// CHECK-NEXT:   token_size:{{ *}}4
// CHECK-NEXT: ch2:
// CHECK-NEXT:   src:{{ *}}{ process: relay, port: out0 }
// CHECK-NEXT:   dst:{{ *}}{ process: parent-out0-sink, port: in0 }
// CHECK-NEXT:   token_size:{{ *}}4
// CHECK: execution:
// CHECK-NEXT: processes:
// CHECK-NEXT:   profiles:
// CHECK-NEXT:     boundary:
// CHECK-NEXT:       UNKNOWN:
// CHECK-NEXT:         cycles:{{ *}}0
// CHECK-NEXT:     relay:
// CHECK-NEXT:       UNKNOWN:
// CHECK-NEXT:         cycles:{{ *}}0
// CHECK:   instances:
// CHECK:     parent-in0-source:
// CHECK-NEXT:       model:{{ *}}static
// CHECK-NEXT:       profile:{{ *}}boundary
// CHECK:     parent-out0-sink:
// CHECK-NEXT:       model:{{ *}}static
// CHECK-NEXT:       profile:{{ *}}boundary
// CHECK: channels:
// CHECK: ch1:
// CHECK-NEXT:     initial_tokens:{{ *}}0
// CHECK: ch2:
// CHECK-NEXT:     initial_tokens:{{ *}}0

dfg.process @relay inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)

dfg.region @child inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)
{
    dfg.instantiate @relay inputs(%in) outputs(%out)
        : (!dfg.output<i32>) -> !dfg.input<i32>
}

dfg.region @parent inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>) attributes {laksa.root}
{
    %0:2 = dfg.channel() : i32
    dfg.embed @child inputs(%in) outputs(%0#0)
        : (!dfg.output<i32>) -> !dfg.input<i32>
    dfg.instantiate @relay inputs(%0#1) outputs(%out)
        : (!dfg.output<i32>) -> !dfg.input<i32>
}
