# Recreates ops from test/Conversion/dfg-to-emithls.mlir using the Python
# bindings and runs the convert-dfg-to-emithls pass on them.

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
    TypeAttr,
    UnitAttr,
)
from mlir_laksa.dialects import dfg, emithls, memref
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def const(ty, value):
    return emithls.VariableOp(ty, init_number=IntegerAttr.get(ty, value), is_const=True)


def build_pad(i8, index_ty) -> dfg.ProcessOp:
    shape = [8]
    in_type = dfg.OutputType.get(element_type=i8, shape=shape)
    out_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "pad", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        c30_index = const(index_ty, 30)
        c_neg1_index = const(index_ty, -1)
        c0_index = const(index_ty, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            alloc = memref.AllocOp(MemRefType.get(shape, i8), [], [], alignment=64)
            alloc.operation.attributes["fill_with"] = IntegerAttr.get(i8, -128)

            for0 = emithls.ForOp(0, 32, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                for1 = emithls.ForOp(0, 32, 1)
                for1_block = for1.body.blocks.append(index_ty)
                with InsertionPoint(for1_block):
                    (idx1,) = for1_block.arguments

                    i1 = IntegerType.get_signless(1)
                    expr_op = emithls.ExpressionOp(i1)
                    expr_block = expr_op.body.blocks.append()
                    with InsertionPoint(expr_block):
                        a0 = emithls.ArithAddOp(idx0, c_neg1_index.result)
                        cmp0 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a0.result,
                            c0_index.result,
                            results=[i1],
                        )
                        m1 = emithls.ArithMulOp(idx0, c_neg1_index.result)
                        a1 = emithls.ArithAddOp(m1.result, c30_index.result)
                        cmp1 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a1.result,
                            c0_index.result,
                            results=[i1],
                        )
                        a2 = emithls.ArithAddOp(idx1, c_neg1_index.result)
                        cmp2 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a2.result,
                            c0_index.result,
                            results=[i1],
                        )
                        m2 = emithls.ArithMulOp(idx1, c_neg1_index.result)
                        a3 = emithls.ArithAddOp(m2.result, c30_index.result)
                        cmp3 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a3.result,
                            c0_index.result,
                            results=[i1],
                        )
                        and_op = emithls.ArithLogicalAndOp(
                            [cmp0.result, cmp1.result, cmp2.result, cmp3.result]
                        )
                        emithls.YieldOp(and_op.result)

                    if_op = emithls.IfOp(expr_op.result)
                    then_block = if_op.thenRegion.blocks.append()
                    with InsertionPoint(then_block):
                        token_memref0 = dfg.PullAsMemRefOp(
                            MemRefType.get(shape, i8), in_port
                        )
                        dfg.PushMemRefOp(token_memref0.result, out_port)
                    else_block = if_op.elseRegion.blocks.append()
                    with InsertionPoint(else_block):
                        dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_conv(i8, i32, index_ty) -> dfg.ProcessOp:
    shape = [8]
    in_type = dfg.OutputType.get(element_type=i8, shape=shape)
    out_type = dfg.InputType.get(element_type=i32, shape=shape)
    process_op = dfg.ProcessOp(
        "conv", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        c_neg2 = const(index_ty, -2)
        c0_index = const(index_ty, 0)
        c2_i32 = const(i32, 2)
        c_neg128_i32 = const(i32, -128)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            for0 = emithls.ForOp(0, 32, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                for1 = emithls.ForOp(0, 32, 1)
                for1_block = for1.body.blocks.append(index_ty)
                with InsertionPoint(for1_block):
                    (idx1,) = for1_block.arguments

                    token = dfg.PullAsMemRefOp(MemRefType.get(shape, i8), in_port)
                    linebuf = emithls.HelperLineBufferOp(
                        emithls.ArrayType.get(element_type=i8, shape=[8, 2, 32]),
                        token.result,
                        [8],
                        2,
                        index=idx1,
                    )
                    window = emithls.HelperWindowOp(
                        emithls.ArrayType.get(element_type=i8, shape=[8, 3, 3]),
                        linebuf.result,
                        [idx0, idx1],
                    )

                    i1 = IntegerType.get_signless(1)
                    expr_op = emithls.ExpressionOp(i1)
                    expr_block = expr_op.body.blocks.append()
                    with InsertionPoint(expr_block):
                        a0 = emithls.ArithAddOp(idx0, c_neg2.result)
                        cmp0 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a0.result,
                            c0_index.result,
                            results=[i1],
                        )
                        a1 = emithls.ArithAddOp(idx1, c_neg2.result)
                        cmp1 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            a1.result,
                            c0_index.result,
                            results=[i1],
                        )
                        and_op = emithls.ArithLogicalAndOp([cmp0.result, cmp1.result])
                        emithls.YieldOp(and_op.result)

                    if_op = emithls.IfOp(expr_op.result)
                    then_block = if_op.thenRegion.blocks.append()
                    with InsertionPoint(then_block):
                        for2 = emithls.ForOp(0, 8, 1)
                        for2_block = for2.body.blocks.append(index_ty)
                        with InsertionPoint(for2_block):
                            (idx2,) = for2_block.arguments

                            alloc = memref.AllocOp(
                                MemRefType.get(shape, i32), [], [], alignment=64
                            )
                            alloc.operation.attributes["fill_with"] = IntegerAttr.get(
                                i32, 0
                            )

                            for3 = emithls.ForOp(0, 3, 1)
                            for3_block = for3.body.blocks.append(index_ty)
                            with InsertionPoint(for3_block):
                                (idx3,) = for3_block.arguments
                                for4 = emithls.ForOp(0, 3, 1)
                                for4_block = for4.body.blocks.append(index_ty)
                                with InsertionPoint(for4_block):
                                    (idx4,) = for4_block.arguments
                                    for5 = emithls.ForOp(0, 8, 1)
                                    for5_block = for5.body.blocks.append(index_ty)
                                    with InsertionPoint(for5_block):
                                        (idx5,) = for5_block.arguments

                                        read = emithls.ArrayReadOp(
                                            window.result, [idx5, idx3, idx4]
                                        )
                                        cast0 = emithls.ArithCastOp(i32, read.result)
                                        sub0 = emithls.ArithSubOp(
                                            cast0.result, c_neg128_i32.result
                                        )
                                        mul0 = emithls.ArithMulOp(
                                            sub0.result, c2_i32.result
                                        )
                                        emithls.HelperAccumulateOp(
                                            alloc.result,
                                            emithls.FusedOperator.add,
                                            mul0.result,
                                            [idx2],
                                        )
                            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_relu(i8, i32, i64, index_ty) -> dfg.ProcessOp:
    shape = [8]
    in_type = dfg.OutputType.get(element_type=i32, shape=shape)
    out_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "relu", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        c0_index = const(index_ty, 0)
        arr_ty = emithls.ArrayType.get(element_type=i32, shape=[8])
        data = np.array(
            [
                1523322938,
                1544671534,
                1547120170,
                1546480699,
                1536582660,
                1557469766,
                1525058969,
                1526555847,
            ],
            dtype=np.int32,
        )
        const_array_0 = emithls.VariableOp(
            arr_ty,
            init_number=DenseIntElementsAttr.get(data, type=arr_ty),
            is_const=True,
        )
        c127_i32 = const(i32, 127)
        c_neg128_i32 = const(i32, -128)
        c41_i64 = const(i64, 41)
        c_neg_half_i64 = const(i64, -1073741824)
        c_half_i64 = const(i64, 1073741824)
        c0_i32 = const(i32, 0)
        c_round_i64 = const(i64, 1099511627776)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            for0 = emithls.ForOp(0, 30, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                for1 = emithls.ForOp(0, 30, 1)
                for1_block = for1.body.blocks.append(index_ty)
                with InsertionPoint(for1_block):
                    (idx1,) = for1_block.arguments
                    for2 = emithls.ForOp(0, 8, 1)
                    for2_block = for2.body.blocks.append(index_ty)
                    with InsertionPoint(for2_block):
                        (idx2,) = for2_block.arguments

                        token0 = dfg.PullOp(i32, in_port, [idx2])
                        read0 = emithls.ArrayReadOp(const_array_0.result, [idx2])
                        cast1 = emithls.ArithCastOp(i64, token0.result)
                        cast2 = emithls.ArithCastOp(i64, read0.result)
                        mul3 = emithls.ArithMulOp(cast1.result, cast2.result)
                        add4 = emithls.ArithAddOp(mul3.result, c_round_i64.result)
                        i1 = IntegerType.get_signless(1)
                        cmp5 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.ge,
                            token0.result,
                            c0_i32.result,
                            results=[i1],
                        )
                        select6 = emithls.ArithSelectOp(
                            cmp5.result,
                            c_half_i64.result,
                            c_neg_half_i64.result,
                            results=[i64],
                        )
                        add7 = emithls.ArithAddOp(select6.result, add4.result)
                        shr8 = emithls.ArithShrOp(add7.result, c41_i64.result)
                        cast9 = emithls.ArithCastOp(i32, shr8.result)
                        add10 = emithls.ArithAddOp(cast9.result, c_neg128_i32.result)
                        max11 = emithls.ArithMaxOp(add10.result, c_neg128_i32.result)
                        min12 = emithls.ArithMinOp(max11.result, c127_i32.result)
                        cast13 = emithls.ArithCastOp(i8, min12.result)

                        dfg.PushOp(cast13.result, out_port, [idx2])
    return process_op


def build_kernel(i8, i32) -> dfg.RegionOp:
    in_type = dfg.OutputType.get(element_type=i8, shape=[8])
    out_type = dfg.InputType.get(element_type=i8, shape=[8])
    region_op = dfg.RegionOp(
        "kernel", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    region_op.operation.attributes["laksa.root"] = UnitAttr.get()
    block = region_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in0, out0 = block.arguments
        in_port_0, out_port_0 = dfg.ChannelOp(
            dfg.InputType.get(element_type=i8, shape=[8]),
            dfg.OutputType.get(element_type=i8, shape=[8]),
            i8,
        ).results
        in_port_1, out_port_1 = dfg.ChannelOp(
            dfg.InputType.get(element_type=i32, shape=[8]),
            dfg.OutputType.get(element_type=i32, shape=[8]),
            i32,
        ).results
        dfg.InstantiateOp("pad", [in0], [in_port_0])
        dfg.InstantiateOp("conv", [out_port_0], [in_port_1])
        dfg.InstantiateOp("relu", [out_port_1], [out0])
    return region_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            index_ty = IndexType.get()

            build_pad(i8, index_ty)
            build_conv(i8, i32, index_ty)
            build_relu(i8, i32, i64, index_ty)
            build_kernel(i8, i32)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_dfg_to_emithls_pass(pm)
        pm.add("reconcile-unrealized-casts")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
