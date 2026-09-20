// RUN: laksa-translate %s --emithls-to-model-profile | FileCheck %s

// CHECK:      metadata: {source: laksa-model}
// CHECK-NEXT: execution:
// CHECK-NEXT:   processes:
// CHECK-NEXT:     profiles:
// CHECK-NEXT:       "main_node_0":
// CHECK-NEXT:         "K26_PL": {cycles: 42}
// CHECK-NEXT:       "main_node_1":
// CHECK-NEXT:         "K26_PL": {cycles: 137}

module {
  emithls.func @main_node_0() attributes {emithls.model_cycles = 42 : i64} {
    %unused = emithls.variable as i32 = 0
  }
  emithls.func @io_bridge() {
    %unused = emithls.variable as i32 = 0
  }
  emithls.func @main_node_1() attributes {emithls.model_cycles = 137 : i64} {
    %unused = emithls.variable as i32 = 0
  }
}
