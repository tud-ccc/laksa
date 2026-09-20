// RUN: laksa-translate %s --emithls-to-hls-profile-script -o %t
// RUN: sh -n %t
// RUN: FileCheck %s < %t

// CHECK: #!/bin/sh
// CHECK: script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
// CHECK: profiles_dir="$script_dir/../profiles"
// CHECK: mkdir -p "$profiles_dir"
// CHECK: exec laksa-extract-hls-profile
// CHECK: "$script_dir"
// CHECK: --processor-type K26_PL
// CHECK: -o "$profiles_dir/profiles_K26_PL_hls.yaml"

module {
}
