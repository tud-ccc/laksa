// RUN: laksa-translate %s --emitc-to-cpu-profile-run-script -o %t
// RUN: sh -n %t
// RUN: FileCheck %s < %t

// CHECK: #!/bin/sh
// CHECK: processor_type=CortexA53
// CHECK: --processor-type)
// CHECK: --cpu)
// CHECK: profile_file="profiles_${processor_type}.yaml"
// CHECK: echo "Running benchmark with:"
// CHECK: > "$profile_file"

module {
  emitc.func @main_node_0(%arg0: !emitc.array<4xi32>) {
    return
  }
}
