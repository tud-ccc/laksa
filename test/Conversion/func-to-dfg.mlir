// RUN: laksa-opt %s --convert-func-to-dfg | FileCheck %s

// CHECK-LABEL: dfg.operator @stream
// CHECK-SAME:  inputs(%in0: i32, %in1: tensor<2xi32>)
// CHECK-SAME:  outputs(%{{[a-z0-9_]+}}: i32, %{{[a-z0-9_]+}}: tensor<2xi32>)
// CHECK:         dfg.output %in0, %in1 : i32, tensor<2xi32>
func.func @stream(%arg0: i32, %arg1: tensor<2xi32>) -> (i32, tensor<2xi32>)
{
    func.return %arg0, %arg1 : i32, tensor<2xi32>
}

// CHECK-LABEL: dfg.region @call
// CHECK-SAME:  inputs(%in0 : !dfg.output<i32>, %in1 : !dfg.output<2xi32>)
// CHECK-SAME:  outputs(%out0 : !dfg.input<i32>, %out1 : !dfg.input<2xi32>)
// CHECK:         %in_port_0, %out_port_0 = dfg.channel() : i32
// CHECK:         %in_port_1, %out_port_1 = dfg.channel() : 2xi32
// CHECK:         dfg.instantiate @stream inputs(%in0, %in1) outputs(%in_port_0, %in_port_1)
// CHECK:         dfg.instantiate @stream inputs(%out_port_0, %out_port_1) outputs(%out0, %out1)
func.func @call(%arg0: i32, %arg1: tensor<2xi32>) -> (i32, tensor<2xi32>)
{
    %0, %1 = func.call @stream(%arg0, %arg1) : (i32, tensor<2xi32>) -> (i32, tensor<2xi32>)
    %2, %3 = func.call @stream(%0, %1) : (i32, tensor<2xi32>) -> (i32, tensor<2xi32>)
    func.return %2, %3 : i32, tensor<2xi32>
}
