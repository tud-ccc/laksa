// RUN: laksa-translate %s --dfg-to-mocasin | FileCheck %s

// CHECK: name:{{ *}}parent
// CHECK: graph:
// CHECK-NEXT: processes:
// CHECK-NEXT: child.relay:
// CHECK-NEXT:   ports:
// CHECK-NEXT:     in:{{ *}}[ in0 ]
// CHECK-NEXT:     out:{{ *}}[ out0 ]
// CHECK-NEXT: child.source:
// CHECK-NEXT:   ports:
// CHECK-NEXT:     out:{{ *}}[ out0 ]
// CHECK-NEXT: sink:
// CHECK-NEXT:   ports:
// CHECK-NEXT:     in:{{ *}}[ in0 ]
// CHECK-NEXT: channels:
// CHECK-NEXT: ch0:
// CHECK-NEXT:   src:{{ *}}{ process: child.source, port: out0 }
// CHECK-NEXT:   dst:{{ *}}{ process: child.relay, port: in0 }
// CHECK-NEXT:   token_size:{{ *}}4
// CHECK-NEXT: ch1:
// CHECK-NEXT:   src:{{ *}}{ process: child.relay, port: out0 }
// CHECK-NEXT:   dst:{{ *}}{ process: sink, port: in0 }
// CHECK-NEXT:   token_size:{{ *}}4
// CHECK: execution:
// CHECK-NEXT: processes:
// CHECK-NEXT:   profiles:
// CHECK-NEXT:     relay:
// CHECK-NEXT:       UNKNOWN:
// CHECK-NEXT:         cycles:{{ *}}0
// CHECK-NEXT:     sink:
// CHECK-NEXT:       UNKNOWN:
// CHECK-NEXT:         cycles:{{ *}}0
// CHECK-NEXT:     source:
// CHECK-NEXT:       UNKNOWN:
// CHECK-NEXT:         cycles:{{ *}}0
// CHECK-NEXT:   instances:
// CHECK-NEXT:     child.relay:
// CHECK-NEXT:       model:{{ *}}static
// CHECK-NEXT:       profile:{{ *}}relay
// CHECK-NEXT:       rates:{{ *}}{ in0: 1, out0: 1 }
// CHECK-NEXT:     child.source:
// CHECK-NEXT:       model:{{ *}}static
// CHECK-NEXT:       profile:{{ *}}source
// CHECK-NEXT:       rates:{{ *}}{ out0: 1 }
// CHECK-NEXT:     sink:
// CHECK-NEXT:       model:{{ *}}static
// CHECK-NEXT:       profile:{{ *}}sink
// CHECK-NEXT:       rates:{{ *}}{ in0: 1 }
// CHECK-NEXT: channels:
// CHECK-NEXT:   ch0:
// CHECK-NEXT:     initial_tokens:{{ *}}0
// CHECK-NEXT:   ch1:
// CHECK-NEXT:     initial_tokens:{{ *}}0

// Declarations (no body) are sufficient here: the Mocasin export only cares
// about graph topology (processes/channels/token sizes), not actor bodies.
dfg.process @sink inputs(%in: !dfg.output<i32>)
dfg.process @source outputs(%out: !dfg.input<i32>)
dfg.process @relay inputs(%in: !dfg.output<i32>) outputs(%out: !dfg.input<i32>)

// @child is embedded into @parent below, so it must not become its own
// top-level Mocasin graph; its instantiate ops are flattened into @parent's
// process list instead, prefixed with the embed's name.
dfg.region @child outputs(%out: !dfg.input<i32>)
{
    %0:2 = dfg.channel() : i32
    dfg.instantiate @source outputs(%0#0) : () -> !dfg.input<i32>
    dfg.instantiate @relay inputs(%0#1) outputs(%out) : (!dfg.output<i32>) -> !dfg.input<i32>
}

dfg.region @parent attributes {laksa.root}
{
    %1:2 = dfg.channel() : i32
    dfg.embed @child outputs(%1#0) : () -> !dfg.input<i32>
    dfg.instantiate @sink inputs(%1#1) : (!dfg.output<i32>) -> ()
}
