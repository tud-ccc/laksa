// RUN: laksa-opt %s --convert-index-to-emithls | FileCheck %s

// CHECK-LABEL: emithls.func @index()
// CHECK:    %var_index_0 = emithls.variable as index
// CHECK:    %const_index_0 = emithls.variable as const index = 2
// CHECK:    %0 = emithls.arith.add %var_index_0, %const_index_0 : index
// CHECK:    %1 = emithls.arith.sub %var_index_0, %const_index_0 : index
// CHECK:    %2 = emithls.arith.mul %var_index_0, %const_index_0 : index
// CHECK:    %3 = emithls.arith.rem %var_index_0, %const_index_0 : index
// CHECK:    %4 = emithls.arith.rem %var_index_0, %const_index_0 : index
// CHECK:    %5 = emithls.arith.shl %var_index_0, %const_index_0 : index
// CHECK:    %6 = emithls.arith.shr %var_index_0, %const_index_0 : index
// CHECK:    %7 = emithls.arith.shr %var_index_0, %const_index_0 : index
// CHECK:    %8 = emithls.arith.max %var_index_0, %const_index_0 : index
// CHECK:    %9 = emithls.arith.max %var_index_0, %const_index_0 : index
// CHECK:    %10 = emithls.arith.min %var_index_0, %const_index_0 : index
// CHECK:    %11 = emithls.arith.min %var_index_0, %const_index_0 : index
// CHECK:    %12 = emithls.arith.and %var_index_0, %const_index_0 : index
// CHECK:    %13 = emithls.arith.or %var_index_0, %const_index_0 : index
// CHECK:    %14 = emithls.arith.cast %var_index_0 : index to i32
// CHECK:    %15 = emithls.arith.cast %var_index_0 : index to i32
// CHECK:    %16 = emithls.arith.cmp eq, %var_index_0, %const_index_0 : index
// CHECK:    %17 = emithls.arith.cmp ne, %var_index_0, %const_index_0 : index
// CHECK:    %18 = emithls.arith.cmp lt, %var_index_0, %const_index_0 : index
// CHECK:    %19 = emithls.arith.cmp le, %var_index_0, %const_index_0 : index
// CHECK:    %20 = emithls.arith.cmp gt, %var_index_0, %const_index_0 : index
// CHECK:    %21 = emithls.arith.cmp ge, %var_index_0, %const_index_0 : index
// CHECK:    %22 = emithls.arith.cmp lt, %var_index_0, %const_index_0 : index
// CHECK:    %23 = emithls.arith.cmp le, %var_index_0, %const_index_0 : index
// CHECK:    %24 = emithls.arith.cmp gt, %var_index_0, %const_index_0 : index
// CHECK:    %25 = emithls.arith.cmp ge, %var_index_0, %const_index_0 : index

emithls.func @index()
{
    %idx = emithls.variable as index
    %0 = index.constant 2
    %1 = index.add %idx, %0
    %2 = index.sub %idx, %0
    %3 = index.mul %idx, %0
    %4 = index.remu %idx, %0
    %5 = index.rems %idx, %0
    %6 = index.shl %idx, %0
    %7 = index.shru %idx, %0
    %8 = index.shrs %idx, %0
    %9 = index.maxu %idx, %0
    %10 = index.maxs %idx, %0
    %11 = index.minu %idx, %0
    %12 = index.mins %idx, %0
    %13 = index.and %idx, %0
    %14 = index.or %idx, %0
    %15 = index.casts %idx : index to i32
    %16 = index.castu %idx : index to i32
    %17 = index.cmp eq(%idx, %0)
    %18 = index.cmp ne(%idx, %0)
    %19 = index.cmp slt(%idx, %0)
    %20 = index.cmp sle(%idx, %0)
    %21 = index.cmp sgt(%idx, %0)
    %22 = index.cmp sge(%idx, %0)
    %23 = index.cmp ult(%idx, %0)
    %24 = index.cmp ule(%idx, %0)
    %25 = index.cmp ugt(%idx, %0)
    %26 = index.cmp uge(%idx, %0)
}
