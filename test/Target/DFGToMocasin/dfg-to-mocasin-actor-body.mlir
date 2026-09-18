// RUN: laksa-translate %s --dfg-to-mocasin | FileCheck %s
// RUN: laksa-opt %s --convert-to-dfg | laksa-translate --dfg-to-mocasin | FileCheck %s

// Computation dialects in actor bodies must be parsed even though only the
// graph topology is exported.
// CHECK: name:{{ *}}graph
// CHECK: src:{{ *}}{ process: source, port: out0 }
// CHECK-NEXT: dst:{{ *}}{ process: sink, port: in0 }
// CHECK-NEXT: token_size:{{ *}}4
// CHECK: execution:

dfg.operator @source outputs(%out: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %empty = tensor.empty() : tensor<4xi32>
    %filled = linalg.fill ins(%c1 : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %value = tensor.extract %filled[%c0] : tensor<4xi32>
    dfg.output %value : i32
}

dfg.process @sink inputs(%in: !dfg.output<i32>)

dfg.region @graph attributes {laksa.root} {
    %channel:2 = dfg.channel() : i32
    dfg.instantiate @source outputs(%channel#0) : () -> !dfg.input<i32>
    dfg.instantiate @sink inputs(%channel#1) : (!dfg.output<i32>) -> ()
}
