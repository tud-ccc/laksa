# Recreates ops from test/Dialect/EmitHLS/Transforms/add-io-functions.mlir
# using the Python bindings and runs the emithls-add-io-functions and
# canonicalize passes on them.

import numpy as np

from mlir_laksa.ir import (
    Context,
    DenseIntElementsAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
    RankedTensorType,
    TypeAttr,
)
from mlir_laksa.dialects import emithls, memref
from mlir_laksa.passmanager import PassManager


def const(ty, value):
    return emithls.VariableOp(ty, init_number=IntegerAttr.get(ty, value), is_const=True)


def build_pad(i8, index_ty, arr8_stream_i8) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "pad",
        TypeAttr.get(FunctionType.get([arr8_stream_i8, arr8_stream_i8], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i8, arr8_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        c30 = const(index_ty, 30)
        c1 = const(index_ty, 1)
        alloc = memref.AllocOp(MemRefType.get([8], i8), [], [], alignment=64)
        alloc.operation.attributes["fill_with"] = IntegerAttr.get(i8, -128)

        i1 = IntegerType.get_signless(1)
        outer = emithls.ForOp(0, 32, 1)
        outer_block = outer.body.blocks.append(index_ty)
        with InsertionPoint(outer_block):
            (idx0,) = outer_block.arguments
            inner = emithls.ForOp(0, 32, 1)
            inner_block = inner.body.blocks.append(index_ty)
            with InsertionPoint(inner_block):
                (idx1,) = inner_block.arguments
                expr0 = emithls.ExpressionOp(i1)
                expr0_block = expr0.body.blocks.append()
                with InsertionPoint(expr0_block):
                    cmp0 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge, idx0, c1.variable, results=[i1]
                    )
                    cmp1 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.le, idx0, c30.variable, results=[i1]
                    )
                    cmp2 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge, idx1, c1.variable, results=[i1]
                    )
                    cmp3 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.le, idx1, c30.variable, results=[i1]
                    )
                    and0 = emithls.ArithLogicalAndOp(
                        [cmp0.result, cmp1.result, cmp2.result, cmp3.result]
                    )
                    emithls.YieldOp(and0.result)

                if_op = emithls.IfOp(expr0.result)
                then_block = if_op.thenRegion.blocks.append()
                with InsertionPoint(then_block):
                    then_for = emithls.ForOp(0, 8, 1)
                    then_for_block = then_for.body.blocks.append(index_ty)
                    with InsertionPoint(then_for_block):
                        (idx2,) = then_for_block.arguments
                        read0 = emithls.StreamReadOp(i8, arg0, [idx2])
                        emithls.StreamWriteOp(read0.result, arg1, [idx2])
                else_block = if_op.elseRegion.blocks.append()
                with InsertionPoint(else_block):
                    else_for = emithls.ForOp(0, 8, 1)
                    else_for_block = else_for.body.blocks.append(index_ty)
                    with InsertionPoint(else_for_block):
                        (idx2,) = else_for_block.arguments
                        load0 = memref.LoadOp(alloc.result, [idx2])
                        emithls.StreamWriteOp(load0.result, arg1, [idx2])
    return func_op


def build_conv(i8, i32, index_ty, arr8_stream_i8, arr8_stream_i32) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "conv",
        TypeAttr.get(FunctionType.get([arr8_stream_i8, arr8_stream_i32], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i8, arr8_stream_i32)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        c2 = const(index_ty, 2)
        c1 = const(index_ty, 1)
        c_i8_0 = const(i8, 0)
        c0 = const(index_ty, 0)
        array_8x3x3_i8 = emithls.ArrayType.get(element_type=i8, shape=[8, 3, 3])
        var_array_0 = emithls.VariableOp(array_8x3x3_i8)
        array_8x2x32_i8 = emithls.ArrayType.get(element_type=i8, shape=[8, 2, 32])
        var_array_1 = emithls.VariableOp(array_8x2x32_i8)
        c_i32_0 = const(i32, 3)
        c_i32_neg128 = const(i32, -128)

        i1 = IntegerType.get_signless(1)
        outer0 = emithls.ForOp(0, 32, 1)
        outer0_block = outer0.body.blocks.append(index_ty)
        with InsertionPoint(outer0_block):
            (idx0,) = outer0_block.arguments
            outer1 = emithls.ForOp(0, 32, 1)
            outer1_block = outer1.body.blocks.append(index_ty)
            with InsertionPoint(outer1_block):
                (idx1,) = outer1_block.arguments
                for2 = emithls.ForOp(0, 8, 1)
                for2_block = for2.body.blocks.append(index_ty)
                with InsertionPoint(for2_block):
                    (idx2,) = for2_block.arguments
                    read0 = emithls.StreamReadOp(i8, arg0, [idx2])

                    expr0 = emithls.ExpressionOp(i1)
                    expr0_block = expr0.body.blocks.append()
                    with InsertionPoint(expr0_block):
                        cmp0 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.eq,
                            idx0,
                            c0.variable,
                            results=[i1],
                        )
                        emithls.YieldOp(cmp0.result)

                    for3 = emithls.ForOp(0, 3, 1)
                    for3_block = for3.body.blocks.append(index_ty)
                    with InsertionPoint(for3_block):
                        (idx3,) = for3_block.arguments
                        for4 = emithls.ForOp(0, 2, 1)
                        for4_block = for4.body.blocks.append(index_ty)
                        with InsertionPoint(for4_block):
                            (idx4,) = for4_block.arguments
                            if_op = emithls.IfOp(expr0.result)
                            then_block = if_op.thenRegion.blocks.append()
                            with InsertionPoint(then_block):
                                emithls.UpdateOp(
                                    var_array_0.variable,
                                    [idx2, idx3, idx4],
                                    c_i8_0.variable,
                                    [],
                                )
                            else_block = if_op.elseRegion.blocks.append()
                            with InsertionPoint(else_block):
                                expr1 = emithls.ExpressionOp(index_ty)
                                expr1_block = expr1.body.blocks.append()
                                with InsertionPoint(expr1_block):
                                    add0 = emithls.ArithAddOp(idx4, c1.variable)
                                    emithls.YieldOp(add0.result)
                                emithls.UpdateOp(
                                    var_array_0.variable,
                                    [idx2, idx3, idx4],
                                    var_array_0.variable,
                                    [idx2, idx3, expr1.result],
                                )

                    expr2 = emithls.ExpressionOp(i1)
                    expr2_block = expr2.body.blocks.append()
                    with InsertionPoint(expr2_block):
                        cmp1 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            idx0,
                            c2.variable,
                            results=[i1],
                        )
                        emithls.YieldOp(cmp1.result)

                    if_op2 = emithls.IfOp(expr2.result)
                    then_block2 = if_op2.thenRegion.blocks.append()
                    with InsertionPoint(then_block2):
                        expr3 = emithls.ExpressionOp(index_ty)
                        expr3_block = expr3.body.blocks.append()
                        with InsertionPoint(expr3_block):
                            sub0 = emithls.ArithSubOp(idx0, c2.variable)
                            rem0 = emithls.ArithRemOp(sub0.result, c2.variable)
                            emithls.YieldOp(rem0.result)
                        emithls.UpdateOp(
                            var_array_0.variable,
                            [idx2, c0.variable, c2.variable],
                            var_array_1.variable,
                            [idx2, expr3.result, idx1],
                        )
                    else_block2 = if_op2.elseRegion.blocks.append()
                    with InsertionPoint(else_block2):
                        emithls.UpdateOp(
                            var_array_0.variable,
                            [idx2, c0.variable, c2.variable],
                            c_i8_0.variable,
                            [],
                        )

                    expr4 = emithls.ExpressionOp(i1)
                    expr4_block = expr4.body.blocks.append()
                    with InsertionPoint(expr4_block):
                        cmp2 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            idx0,
                            c1.variable,
                            results=[i1],
                        )
                        emithls.YieldOp(cmp2.result)

                    if_op3 = emithls.IfOp(expr4.result)
                    then_block3 = if_op3.thenRegion.blocks.append()
                    with InsertionPoint(then_block3):
                        expr5 = emithls.ExpressionOp(index_ty)
                        expr5_block = expr5.body.blocks.append()
                        with InsertionPoint(expr5_block):
                            sub1 = emithls.ArithSubOp(idx0, c1.variable)
                            rem1 = emithls.ArithRemOp(sub1.result, c2.variable)
                            emithls.YieldOp(rem1.result)
                        emithls.UpdateOp(
                            var_array_0.variable,
                            [idx2, c1.variable, c2.variable],
                            var_array_1.variable,
                            [idx2, expr5.result, idx1],
                        )
                    else_block3 = if_op3.elseRegion.blocks.append()
                    with InsertionPoint(else_block3):
                        emithls.UpdateOp(
                            var_array_0.variable,
                            [idx2, c1.variable, c2.variable],
                            c_i8_0.variable,
                            [],
                        )

                    expr6 = emithls.ExpressionOp(index_ty)
                    expr6_block = expr6.body.blocks.append()
                    with InsertionPoint(expr6_block):
                        rem2 = emithls.ArithRemOp(idx0, c2.variable)
                        emithls.YieldOp(rem2.result)

                    emithls.UpdateOp(
                        var_array_0.variable,
                        [idx2, c2.variable, c2.variable],
                        read0.result,
                        [],
                    )
                    emithls.UpdateOp(
                        var_array_1.variable,
                        [idx2, expr6.result, idx1],
                        read0.result,
                        [],
                    )

                expr7 = emithls.ExpressionOp(i1)
                expr7_block = expr7.body.blocks.append()
                with InsertionPoint(expr7_block):
                    cmp3 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge, idx0, c2.variable, results=[i1]
                    )
                    cmp4 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge, idx1, c2.variable, results=[i1]
                    )
                    and1 = emithls.ArithLogicalAndOp([cmp3.result, cmp4.result])
                    emithls.YieldOp(and1.result)

                if_op4 = emithls.IfOp(expr7.result)
                then_block4 = if_op4.thenRegion.blocks.append()
                with InsertionPoint(then_block4):
                    for2b = emithls.ForOp(0, 8, 1)
                    for2b_block = for2b.body.blocks.append(index_ty)
                    with InsertionPoint(for2b_block):
                        (idx2b,) = for2b_block.arguments
                        var_int32_0 = emithls.VariableOp(
                            i32, init_number=IntegerAttr.get(i32, 0)
                        )
                        for3b = emithls.ForOp(0, 3, 1)
                        for3b_block = for3b.body.blocks.append(index_ty)
                        with InsertionPoint(for3b_block):
                            (idx3b,) = for3b_block.arguments
                            for4b = emithls.ForOp(0, 3, 1)
                            for4b_block = for4b.body.blocks.append(index_ty)
                            with InsertionPoint(for4b_block):
                                (idx4b,) = for4b_block.arguments
                                for5b = emithls.ForOp(0, 8, 1)
                                for5b_block = for5b.body.blocks.append(index_ty)
                                with InsertionPoint(for5b_block):
                                    (idx5b,) = for5b_block.arguments
                                    read1 = emithls.ArrayReadOp(
                                        var_array_0.variable,
                                        [idx5b, idx3b, idx4b],
                                    )
                                    cast0 = emithls.ArithCastOp(i32, read1.result)
                                    sub2 = emithls.ArithSubOp(
                                        cast0.result, c_i32_neg128.variable
                                    )
                                    mul0 = emithls.ArithMulOp(
                                        sub2.result, c_i32_0.variable
                                    )
                                    emithls.ArithFusedOp(
                                        emithls.FusedOperator.add,
                                        var_int32_0.variable,
                                        mul0.result,
                                    )
                        emithls.StreamWriteOp(var_int32_0.variable, arg1, [idx2b])
    return func_op


def build_relu(
    i8, i32, i64, index_ty, arr8_stream_i32, arr8_stream_i8
) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "relu",
        TypeAttr.get(FunctionType.get([arr8_stream_i32, arr8_stream_i8], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i32, arr8_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        array_i32_8 = emithls.ArrayType.get(element_type=i32, shape=[8])
        packed_type = RankedTensorType.get([8], i32)
        packed_data = DenseIntElementsAttr.get(
            np.array(
                [
                    1241604906,
                    1329922038,
                    1341576575,
                    1322904542,
                    1326948123,
                    1339807398,
                    1325877483,
                    1340943676,
                ],
                dtype=np.int32,
            ),
            type=packed_type,
        )
        const_array_0 = emithls.VariableOp(
            array_i32_8, init_number=packed_data, is_const=True
        )
        c_i32_127 = const(i32, 127)
        c_i32_neg128 = const(i32, -128)
        c_i64_40 = const(i64, 40)
        c_i64_neg = const(i64, -1073741824)
        c_i64_pos = const(i64, 1073741824)
        c_i32_0 = const(i32, 0)
        c_i64_big = const(i64, 549755813888)

        outer = emithls.ForOp(0, 900, 1)
        outer_block = outer.body.blocks.append(index_ty)
        with InsertionPoint(outer_block):
            (idx0,) = outer_block.arguments
            inner = emithls.ForOp(0, 8, 1)
            inner_block = inner.body.blocks.append(index_ty)
            with InsertionPoint(inner_block):
                (idx1,) = inner_block.arguments
                read0 = emithls.StreamReadOp(i32, arg0, [idx1])
                read1 = emithls.ArrayReadOp(const_array_0.variable, [idx1])
                cast0 = emithls.ArithCastOp(i64, read0.result)
                cast1 = emithls.ArithCastOp(i64, read1.result)
                mul0 = emithls.ArithMulOp(cast0.result, cast1.result)
                add0 = emithls.ArithAddOp(mul0.result, c_i64_big.variable)
                i1 = IntegerType.get_signless(1)
                cmp0 = emithls.ArithCmpOp(
                    emithls.CmpPredicate.ge,
                    read0.result,
                    c_i32_0.variable,
                    results=[i1],
                )
                select0 = emithls.ArithSelectOp(
                    cmp0.result, c_i64_pos.variable, c_i64_neg.variable
                )
                add1 = emithls.ArithAddOp(select0.result, add0.result)
                shr0 = emithls.ArithShrOp(add1.result, c_i64_40.variable)
                cast2 = emithls.ArithCastOp(i32, shr0.result)
                add2 = emithls.ArithAddOp(cast2.result, c_i32_neg128.variable)
                max0 = emithls.ArithMaxOp(add2.result, c_i32_neg128.variable)
                min0 = emithls.ArithMinOp(max0.result, c_i32_127.variable)
                cast3 = emithls.ArithCastOp(i8, min0.result)
                emithls.StreamWriteOp(cast3.result, arg1, [idx1])
    return func_op


def build_kernel(
    arr8_stream_i8, arr8_stream_i32, pad_func, conv_func, relu_func
) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "kernel",
        TypeAttr.get(FunctionType.get([arr8_stream_i8, arr8_stream_i8], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i8, arr8_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        var_array_0 = emithls.VariableOp(arr8_stream_i8)
        var_array_1 = emithls.VariableOp(arr8_stream_i32)
        emithls.CallOp([], pad_func.sym_name.value, [arg0, var_array_0.variable])
        emithls.CallOp(
            [], conv_func.sym_name.value, [var_array_0.variable, var_array_1.variable]
        )
        emithls.CallOp([], relu_func.sym_name.value, [var_array_1.variable, arg1])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            index_ty = IndexType.get()
            stream_i8 = emithls.StreamType.get(element_type=i8)
            stream_i32 = emithls.StreamType.get(element_type=i32)
            arr8_stream_i8 = emithls.ArrayType.get(element_type=stream_i8, shape=[8])
            arr8_stream_i32 = emithls.ArrayType.get(element_type=stream_i32, shape=[8])

            pad_func = build_pad(i8, index_ty, arr8_stream_i8)
            conv_func = build_conv(i8, i32, index_ty, arr8_stream_i8, arr8_stream_i32)
            relu_func = build_relu(
                i8, i32, i64, index_ty, arr8_stream_i32, arr8_stream_i8
            )
            build_kernel(
                arr8_stream_i8, arr8_stream_i32, pad_func, conv_func, relu_func
            )

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_add_io_functions_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
