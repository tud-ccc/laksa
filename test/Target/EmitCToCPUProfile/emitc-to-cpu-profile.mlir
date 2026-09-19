// RUN: laksa-translate %s --emitc-to-cpu-profile | FileCheck %s
// RUN: laksa-translate %s --emitc-to-cpu-profile-nodes | FileCheck %s --check-prefix=NODES

// NODES: #include <algorithm>
// NODES: void main_node_0(

// CHECK: void main_node_0(int32_t node0Arg0[4]);
// CHECK: void main_node_1(float node1Arg0[2][3]);
// CHECK: PERF_COUNT_HW_CPU_CYCLES
// CHECK: std::uint64_t measureNode0
// CHECK: main_node_0(node0Arg0);
// CHECK: std::uint64_t measureNode1
// CHECK: main_node_1(node1Arg0);
// CHECK: if (options.node.empty() || options.node == "main_node_0")
// CHECK: std::cout << "execution:\n  processes:\n    profiles:\n";

module {
  emitc.func @main_node_0(%arg0: !emitc.array<4xi32>) {
    return
  }
  emitc.func @main_node_1(%arg0: !emitc.array<2x3xf32>) {
    return
  }
}
