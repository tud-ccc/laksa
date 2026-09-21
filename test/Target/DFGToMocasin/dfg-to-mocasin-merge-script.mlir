// RUN: laksa-translate %s --dfg-to-mocasin-merge-script | FileCheck %s

// CHECK:      #!/bin/sh
// CHECK:      set -eu
// CHECK:      script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
// CHECK:      exec laksa-merge-profiles
// CHECK:      "$script_dir/template.yaml"
// CHECK:      --profiles-dir "$script_dir/../profiles"
// CHECK:      --boundary CortexA53
// CHECK:      -o "$script_dir/application.yaml"

module {
}
