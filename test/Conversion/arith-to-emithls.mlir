// RUN: laksa-opt %s --convert-arith-to-emithls | FileCheck %s

// CHECK-LABEL: emithls.func @arith()
// CHECK:    %var_int32_0 = emithls.variable as i32
// CHECK:    %var_index_0 = emithls.variable as index
// CHECK:    %const_int32_0 = emithls.variable as const i32 = 2
// CHECK:    %0 = emithls.arith.add %var_int32_0, %const_int32_0 : i32
// CHECK:    %1 = emithls.arith.sub %var_int32_0, %const_int32_0 : i32
// CHECK:    %2 = emithls.arith.mul %var_int32_0, %const_int32_0 : i32
// CHECK:    %3 = emithls.arith.rem %var_int32_0, %const_int32_0 : i32
// CHECK:    %4 = emithls.arith.rem %var_int32_0, %const_int32_0 : i32
// CHECK:    %5 = emithls.arith.shl %var_int32_0, %const_int32_0 : i32
// CHECK:    %6 = emithls.arith.shr %var_int32_0, %const_int32_0 : i32
// CHECK:    %7 = emithls.arith.shr %var_int32_0, %const_int32_0 : i32
// CHECK:    %8 = emithls.arith.max %var_int32_0, %const_int32_0 : i32
// CHECK:    %9 = emithls.arith.max %var_int32_0, %const_int32_0 : i32
// CHECK:    %10 = emithls.arith.min %var_int32_0, %const_int32_0 : i32
// CHECK:    %11 = emithls.arith.min %var_int32_0, %const_int32_0 : i32
// CHECK:    %12 = emithls.arith.and %var_int32_0, %const_int32_0 : i32
// CHECK:    %13 = emithls.arith.or %var_int32_0, %const_int32_0 : i32
// CHECK:    %14 = emithls.arith.cast %var_index_0 : index to i32
// CHECK:    %15 = emithls.arith.cast %var_index_0 : index to i32
// CHECK:    %16 = emithls.arith.cast %var_int32_0 : i32 to i64
// CHECK:    %17 = emithls.arith.cast %var_int32_0 : i32 to i64
// CHECK:    %18 = emithls.arith.cast %var_int32_0 : i32 to i16
// CHECK:    %19 = emithls.arith.cmp eq, %var_int32_0, %const_int32_0 : i32
// CHECK:    %20 = emithls.arith.cmp ne, %var_int32_0, %const_int32_0 : i32
// CHECK:    %21 = emithls.arith.cmp lt, %var_int32_0, %const_int32_0 : i32
// CHECK:    %22 = emithls.arith.cmp le, %var_int32_0, %const_int32_0 : i32
// CHECK:    %23 = emithls.arith.cmp gt, %var_int32_0, %const_int32_0 : i32
// CHECK:    %24 = emithls.arith.cmp ge, %var_int32_0, %const_int32_0 : i32
// CHECK:    %25 = emithls.arith.cmp lt, %var_int32_0, %const_int32_0 : i32
// CHECK:    %26 = emithls.arith.cmp le, %var_int32_0, %const_int32_0 : i32
// CHECK:    %27 = emithls.arith.cmp gt, %var_int32_0, %const_int32_0 : i32
// CHECK:    %28 = emithls.arith.cmp ge, %var_int32_0, %const_int32_0 : i32
// CHECK:    %29 = emithls.arith.select %21, %var_int32_0, %const_int32_0 : i32

emithls.func @arith()
{
    %var = emithls.variable as i32
    %idx = emithls.variable as index
    %0 = arith.constant 2 : i32
    %1 = arith.addi %var, %0 : i32
    %2 = arith.subi %var, %0 : i32
    %3 = arith.muli %var, %0 : i32
    %4 = arith.remui %var, %0 : i32
    %5 = arith.remsi %var, %0 : i32
    %6 = arith.shli %var, %0 : i32
    %7 = arith.shrui %var, %0 : i32
    %8 = arith.shrsi %var, %0 : i32
    %9 = arith.maxui %var, %0 : i32
    %10 = arith.maxsi %var, %0 : i32
    %11 = arith.minui %var, %0 : i32
    %12 = arith.minsi %var, %0 : i32
    %13 = arith.andi %var, %0 : i32
    %14 = arith.ori %var, %0 : i32
    %15 = arith.index_cast %idx : index to i32
    %16 = arith.index_castui %idx : index to i32
    %17 = arith.extui %var : i32 to i64
    %18 = arith.extsi %var : i32 to i64
    %19 = arith.trunci %var : i32 to i16
    %20 = arith.cmpi eq, %var, %0 : i32
    %21 = arith.cmpi ne, %var, %0 : i32
    %22 = arith.cmpi slt, %var, %0 : i32
    %23 = arith.cmpi sle, %var, %0 : i32
    %24 = arith.cmpi sgt, %var, %0 : i32
    %25 = arith.cmpi sge, %var, %0 : i32
    %26 = arith.cmpi ult, %var, %0 : i32
    %27 = arith.cmpi ule, %var, %0 : i32
    %28 = arith.cmpi ugt, %var, %0 : i32
    %29 = arith.cmpi uge, %var, %0 : i32
    %30 = arith.select %22, %var, %0 : i32
}
