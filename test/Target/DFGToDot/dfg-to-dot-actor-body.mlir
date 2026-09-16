// RUN: laksa-translate %s --dfg-to-dot | FileCheck %s

// Computation dialects in actor bodies must be parsed even though only the
// graph topology is exported.
// CHECK:      digraph G_0 {
// CHECK-NEXT:   rankdir=LR;
// CHECK-NEXT:   subgraph cluster_graph_0 {
// CHECK-NEXT:     label="graph";
// CHECK:          graph_0_source_0 [label="source", shape=box];
// CHECK-NEXT:     graph_0_sink_0 [label="sink", shape=box];
// CHECK-NEXT:     graph_0_source_0 -> graph_0_sink_0;
// CHECK-NEXT:   }
// CHECK-NEXT: }

dfg.operator @source outputs(%out: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %empty = tensor.empty() : tensor<4xi32>
    %filled = linalg.fill ins(%c1 : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %value = tensor.extract %filled[%c0] : tensor<4xi32>
    dfg.output %value : i32
}

dfg.process @sink inputs(%in: !dfg.output<i32>)

dfg.region @graph {
    %channel:2 = dfg.channel() : i32
    dfg.instantiate @source outputs(%channel#0) : () -> !dfg.input<i32>
    dfg.instantiate @sink inputs(%channel#1) : (!dfg.output<i32>) -> ()
}
