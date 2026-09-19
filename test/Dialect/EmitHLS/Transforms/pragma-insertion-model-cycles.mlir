// RUN: laksa-opt %s --emithls-pragma-insertion | FileCheck %s

// Verify that rewriting a function to materialize the DSE solution does not
// discard the HLS model cycle count produced by the preceding pass.

// CHECK-LABEL: emithls.func @main_node_0()
// CHECK-SAME: attributes {emithls.model_cycles = 42 : i64}
emithls.func @main_node_0() attributes {
  dse.factors = [],
  emithls.model_cycles = 42 : i64
} {
  %unused = emithls.variable as i32 = 0
}
