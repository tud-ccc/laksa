// RUN: laksa-opt %s --func-remove-outlined-function-wrappers | FileCheck %s

// The marker, rather than the function name, identifies an outlined wrapper.
// CHECK-NOT: func.func @graph_top
func.func @graph_top() attributes {laksa.root} {
  return
}

// A user function whose name happens to end in `_top` must be preserved.
// CHECK: func.func @user_top
func.func @user_top() {
  return
}

// Unmarked helpers are preserved as well.
// CHECK: func.func @helper
func.func @helper() {
  return
}
