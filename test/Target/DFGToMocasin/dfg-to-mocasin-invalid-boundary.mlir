// RUN: not laksa-translate %s --dfg-to-mocasin 2>&1 | FileCheck %s

// CHECK: error: cannot export to Mocasin: top-level region 'dangling' input port 'in0' must have exactly one consumer inside the region

dfg.region @dangling inputs(%in: !dfg.output<i32>) attributes {laksa.root} {
}
