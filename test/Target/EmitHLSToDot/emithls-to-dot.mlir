// RUN: laksa-translate %s --emithls-to-dot | FileCheck %s

// CHECK:      digraph G_0 {
// CHECK-NEXT:   layout=neato;
// CHECK-NEXT:   splines=line;
// CHECK-NEXT:   label="top";
// CHECK-NEXT:   labelloc=t;
// CHECK-NEXT:   top_in_0 [label="in_0", shape=ellipse, pos="{{.*}}"];
// CHECK-NEXT:   top_out_0 [label="out_0", shape=ellipse, pos="{{.*}}"];
// CHECK-NEXT:   top_relu_0 [label="relu", shape=box, pos="{{.*}}"];
// CHECK-NEXT:   top_in_0 -> top_relu_0;
// CHECK-NEXT:   top_label_0 [shape=plaintext, fontsize=10, width=0, height=0, margin=0, label="stream0\ndepth=4", pos="{{.*}}"];
// CHECK-NEXT:   top_relu_0 -> top_out_0;
// CHECK-NEXT:   top_label_1 [shape=plaintext, fontsize=10, width=0, height=0, margin=0, label="stream1\ndepth=16", pos="{{.*}}"];
// CHECK-NEXT: }

emithls.func @top_read_i8_0(%arg0: !emithls.ptr<i8>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
  emithls.for %idx0 = 0 to 8 step 1 {
    %0 = emithls.array.ptr_read %arg0[%idx0] : !emithls.ptr<i8> -> i8
    emithls.stream.write %0 to %arg1[%idx0] : i8 -> !emithls.array<8x!emithls.stream<i8>>
  }
}

emithls.func @relu(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.array<8x!emithls.stream<i8>>) {
  emithls.for %idx0 = 0 to 8 step 1 {
    %0 = emithls.stream.read %arg0[%idx0] : !emithls.array<8x!emithls.stream<i8>> -> i8
    emithls.stream.write %0 to %arg1[%idx0] : i8 -> !emithls.array<8x!emithls.stream<i8>>
  }
}

emithls.func @top_write_i8_0(%arg0: !emithls.array<8x!emithls.stream<i8>>, %arg1: !emithls.ptr<i8>) {
  emithls.for %idx0 = 0 to 8 step 1 {
    %0 = emithls.stream.read %arg0[%idx0] : !emithls.array<8x!emithls.stream<i8>> -> i8
    emithls.array.ptr_write %0, %arg1[%idx0] : i8 -> !emithls.ptr<i8>
  }
}

emithls.func @top(%arg0: !emithls.ptr<i8>, %arg1: !emithls.ptr<i8>) {
  emithls.top_interface
  %var_array_0 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
  emithls.pragma.stream variable=%var_array_0(!emithls.array<8x!emithls.stream<i8>>) depth=4
  %var_array_1 = emithls.variable as !emithls.array<8x!emithls.stream<i8>>
  emithls.pragma.stream variable=%var_array_1(!emithls.array<8x!emithls.stream<i8>>) depth=16
  emithls.pragma.dataflow {
    emithls.call @top_read_i8_0(%arg0, %var_array_0) : (!emithls.ptr<i8>, !emithls.array<8x!emithls.stream<i8>>) -> ()
    emithls.call @relu(%var_array_0, %var_array_1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.array<8x!emithls.stream<i8>>) -> ()
    emithls.call @top_write_i8_0(%var_array_1, %arg1) : (!emithls.array<8x!emithls.stream<i8>>, !emithls.ptr<i8>) -> ()
  }
}
