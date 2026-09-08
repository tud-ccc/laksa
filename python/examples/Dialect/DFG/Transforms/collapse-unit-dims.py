# Recreates ops from test/Dialect/DFG/Transforms/collapse-unit-dims.mlir using
# the Python bindings and runs the dfg-collapse-unit-dims pass on them.

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
)
from mlir_laksa.dialects import dfg, emithls, memref
from mlir_laksa.passmanager import PassManager


def const(ty, value):
    return emithls.VariableOp(ty, init_number=IntegerAttr.get(ty, value), is_const=True)


def build_kernel_node_0(i8, i32, index_ty) -> dfg.ProcessOp:
    """pull_as_memref -> linebuf -> window -> conditional accumulate into a
    per-line alloc, pushed per line. Exercises every landmark op the pass
    rewrites."""
    shape = [1, 8]
    in_type = dfg.OutputType.get(element_type=i8, shape=shape)
    out_type = dfg.InputType.get(element_type=i32, shape=shape)
    process_op = dfg.ProcessOp(
        "kernel_node_0", TypeAttr.get(FunctionType.get([in_type], [out_type]))
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
                        emithls.ArrayType.get(element_type=i8, shape=[1, 8, 2, 32]),
                        token.result,
                        [1, 8],
                        2,
                        index=idx1,
                    )
                    window = emithls.HelperWindowOp(
                        emithls.ArrayType.get(element_type=i8, shape=[1, 8, 3, 3]),
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
                                            window.result,
                                            [c0_index.result, idx5, idx3, idx4],
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
                                            [c0_index.result, idx2],
                                        )
                            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_kernel_node_1(i8, i32, i64, index_ty) -> dfg.ProcessOp:
    """indexed dfg.pull/dfg.push with no intervening memref op, fanning out
    to two output ports."""
    shape = [1, 8]
    in_type = dfg.OutputType.get(element_type=i32, shape=shape)
    out_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "kernel_node_1",
        TypeAttr.get(FunctionType.get([in_type], [out_type, out_type])),
    )
    block = process_op.body.blocks.append(in_type, out_type, out_type)
    with InsertionPoint(block):
        in_port, out0_port, out1_port = block.arguments

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

        loop_op = dfg.LoopOp([in_port], [out0_port, out1_port])
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

                        token0 = dfg.PullOp(i32, in_port, [c0_index.result, idx2])
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

                        dfg.PushOp(cast13.result, out0_port, [c0_index.result, idx2])
                        dfg.PushOp(cast13.result, out1_port, [c0_index.result, idx2])
    return process_op


def build_kernel_node_2(i8, index_ty) -> dfg.ProcessOp:
    """a port pulled/pushed as a whole memref by both a passthrough
    (pull_as_memref feeding push_memref directly) and a plain memref.alloc
    fallback - both should still collapse."""
    shape = [1, 8]
    in_type = dfg.OutputType.get(element_type=i8, shape=shape)
    out_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "kernel_node_2", TypeAttr.get(FunctionType.get([in_type], [out_type]))
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


def build_scalarize(i8, i32, index_ty) -> dfg.ProcessOp:
    """a port whose every axis is size 1 collapses all the way to a scalar
    port when it's only ever accessed element-wise."""
    shape = [1, 1]
    in_type = dfg.OutputType.get(element_type=i8, shape=shape)
    out_type = dfg.InputType.get(element_type=i32, shape=shape)
    process_op = dfg.ProcessOp(
        "scalarize", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c0_index = const(index_ty, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token0 = dfg.PullOp(i8, in_port, [c0_index.result, c0_index.result])
            cast0 = emithls.ArithCastOp(i32, token0.result)
            dfg.PushOp(cast0.result, out_port, [c0_index.result, c0_index.result])
    return process_op


def build_plain(i8, i32, index_ty) -> dfg.ProcessOp:
    """a process with no size-1 axis on any port must be left byte-identical."""
    in_type = dfg.OutputType.get(element_type=i8, shape=[8])
    out_type = dfg.InputType.get(element_type=i32, shape=[8])
    process_op = dfg.ProcessOp(
        "plain", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c0_index = const(index_ty, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            for0 = emithls.ForOp(0, 8, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                token0 = dfg.PullOp(i8, in_port, [idx0])
                cast0 = emithls.ArithCastOp(i32, token0.result)
                dfg.PushOp(cast0.result, out_port, [idx0])
    return process_op


def build_producer(i8, i32, index_ty) -> dfg.ProcessOp:
    in_type = dfg.OutputType.get(element_type=i8, shape=[1, 8])
    out_type = dfg.InputType.get(element_type=i32, shape=[1, 8])
    process_op = dfg.ProcessOp(
        "producer", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c0 = const(index_ty, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            for0 = emithls.ForOp(0, 8, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                token0 = dfg.PullOp(i8, in_port, [c0.result, idx0])
                cast0 = emithls.ArithCastOp(i32, token0.result)
                dfg.PushOp(cast0.result, out_port, [c0.result, idx0])
    return process_op


def build_consumer(i32, index_ty) -> dfg.ProcessOp:
    in_type = dfg.OutputType.get(element_type=i32, shape=[1, 8])
    out_type = dfg.InputType.get(element_type=i32, shape=[1, 8])
    process_op = dfg.ProcessOp(
        "consumer", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c0 = const(index_ty, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            for0 = emithls.ForOp(0, 8, 1)
            for0_block = for0.body.blocks.append(index_ty)
            with InsertionPoint(for0_block):
                (idx0,) = for0_block.arguments
                token0 = dfg.PullOp(i32, in_port, [c0.result, idx0])
                dfg.PushOp(token0.result, out_port, [c0.result, idx0])
    return process_op


def build_top_region(i8, i32) -> dfg.RegionOp:
    """channel and region boundary types must shrink in lockstep with the
    processes they connect."""
    in_type = dfg.OutputType.get(element_type=i8, shape=[1, 8])
    out_type = dfg.InputType.get(element_type=i32, shape=[1, 8])
    region_op = dfg.RegionOp(
        "top", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = region_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in0, out0 = block.arguments
        ch_in, ch_out = dfg.ChannelOp(
            dfg.InputType.get(element_type=i32, shape=[1, 8]),
            dfg.OutputType.get(element_type=i32, shape=[1, 8]),
            i32,
        ).results
        dfg.InstantiateOp("producer", [in0], [ch_in])
        dfg.InstantiateOp("consumer", [ch_out], [out0])
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

            build_kernel_node_0(i8, i32, index_ty)
            build_kernel_node_1(i8, i32, i64, index_ty)
            build_kernel_node_2(i8, index_ty)
            build_scalarize(i8, i32, index_ty)
            build_plain(i8, i32, index_ty)
            build_producer(i8, i32, index_ty)
            build_consumer(i32, index_ty)
            build_top_region(i8, i32)

        print(module)

        pm = PassManager(context=ctx)
        dfg.add_collapse_unit_dims_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
