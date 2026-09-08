# Recreates ops from test/Dialect/EmitHLS/Transforms/resolve-helpers.mlir
# using the Python bindings and runs the emithls-resolve-helpers and
# canonicalize passes on them.

from mlir_laksa.ir import (
    Context,
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
from mlir_laksa.dialects import emithls, memref
from mlir_laksa.passmanager import PassManager


def const(ty, value):
    return emithls.VariableOp(ty, init_number=IntegerAttr.get(ty, value), is_const=True)


def build_conv_2d(i8, i32, index_ty, arr8_stream_i8, arr8_stream_i32) -> emithls.FuncOp:
    shape = [8]
    func_op = emithls.FuncOp(
        "conv_2d",
        TypeAttr.get(FunctionType.get([arr8_stream_i8, arr8_stream_i32], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i8, arr8_stream_i32)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments

        c_neg2 = const(index_ty, -2)
        c0_index = const(index_ty, 0)
        c2_i32 = const(i32, 2)
        c_neg128_i32 = const(i32, -128)

        for0 = emithls.ForOp(0, 32, 1)
        for0_block = for0.body.blocks.append(index_ty)
        with InsertionPoint(for0_block):
            (idx0,) = for0_block.arguments
            for1 = emithls.ForOp(0, 32, 1)
            for1_block = for1.body.blocks.append(index_ty)
            with InsertionPoint(for1_block):
                (idx1,) = for1_block.arguments

                alloc = memref.AllocOp(MemRefType.get(shape, i8), [], [])
                for2 = emithls.ForOp(0, 8, 1)
                for2_block = for2.body.blocks.append(index_ty)
                with InsertionPoint(for2_block):
                    (idx2,) = for2_block.arguments
                    read0 = emithls.StreamReadOp(i8, arg0, [idx2])
                    memref.StoreOp(read0.result, alloc.result, [idx2])

                linebuf = emithls.HelperLineBufferOp(
                    emithls.ArrayType.get(element_type=i8, shape=[8, 2, 32]),
                    alloc.result,
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
                    for2b = emithls.ForOp(0, 8, 1)
                    for2b_block = for2b.body.blocks.append(index_ty)
                    with InsertionPoint(for2b_block):
                        (idx2b,) = for2b_block.arguments

                        alloc0 = memref.AllocOp(
                            MemRefType.get(shape, i32), [], [], alignment=64
                        )
                        alloc0.operation.attributes["fill_with"] = IntegerAttr.get(
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

                                    read1 = emithls.ArrayReadOp(
                                        window.result, [idx5, idx3, idx4]
                                    )
                                    cast0 = emithls.ArithCastOp(i32, read1.result)
                                    sub0 = emithls.ArithSubOp(
                                        cast0.result, c_neg128_i32.result
                                    )
                                    mul0 = emithls.ArithMulOp(
                                        sub0.result, c2_i32.result
                                    )
                                    emithls.HelperAccumulateOp(
                                        alloc0.result,
                                        emithls.FusedOperator.add,
                                        mul0.result,
                                        [idx2b],
                                    )
                        for6 = emithls.ForOp(0, 8, 1)
                        for6_block = for6.body.blocks.append(index_ty)
                        with InsertionPoint(for6_block):
                            (idx6,) = for6_block.arguments
                            load0 = memref.LoadOp(alloc0.result, [idx6])
                            emithls.StreamWriteOp(load0.result, arg1, [idx6])
    return func_op


def build_matmul(
    i8, i32, index_ty, arr128_stream_i8, arr256_stream_i32
) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "matmul",
        TypeAttr.get(FunctionType.get([arr128_stream_i8, arr256_stream_i32], [])),
    )
    block = func_op.body.blocks.append(arr128_stream_i8, arr256_stream_i32)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        c_neg128_i32 = const(i32, -128)

        for0 = emithls.ForOp(0, 512, 1)
        for0_block = for0.body.blocks.append(index_ty)
        with InsertionPoint(for0_block):
            (idx0,) = for0_block.arguments

            alloc = memref.AllocOp(MemRefType.get([128], i8), [], [])
            for1 = emithls.ForOp(0, 128, 1)
            for1_block = for1.body.blocks.append(index_ty)
            with InsertionPoint(for1_block):
                (idx1,) = for1_block.arguments
                read0 = emithls.StreamReadOp(i8, arg0, [idx1])
                memref.StoreOp(read0.result, alloc.result, [idx1])

            linebuf = emithls.HelperLineBufferOp(
                emithls.ArrayType.get(element_type=i8, shape=[128]),
                alloc.result,
                [],
                0,
                index=idx0,
            )

            for1b = emithls.ForOp(0, 256, 1)
            for1b_block = for1b.body.blocks.append(index_ty)
            with InsertionPoint(for1b_block):
                (idx1b,) = for1b_block.arguments

                alloc0 = memref.AllocOp(
                    MemRefType.get([256], i32), [], [], alignment=64
                )
                alloc0.operation.attributes["fill_with"] = IntegerAttr.get(i32, 0)

                for2 = emithls.ForOp(0, 128, 1)
                for2_block = for2.body.blocks.append(index_ty)
                with InsertionPoint(for2_block):
                    (idx2,) = for2_block.arguments
                    read1 = emithls.ArrayReadOp(linebuf.result, [idx2])
                    cast0 = emithls.ArithCastOp(i32, read1.result)
                    sub0 = emithls.ArithSubOp(cast0.result, c_neg128_i32.result)
                    mul0 = emithls.ArithMulOp(sub0.result, c_neg128_i32.result)
                    emithls.HelperAccumulateOp(
                        alloc0.result,
                        emithls.FusedOperator.add,
                        mul0.result,
                        [idx1b],
                    )

                for3 = emithls.ForOp(0, 256, 1)
                for3_block = for3.body.blocks.append(index_ty)
                with InsertionPoint(for3_block):
                    (idx3,) = for3_block.arguments
                    load0 = memref.LoadOp(alloc0.result, [idx3])
                    emithls.StreamWriteOp(load0.result, arg1, [idx3])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)
            index_ty = IndexType.get()
            stream_i8 = emithls.StreamType.get(element_type=i8)
            stream_i32 = emithls.StreamType.get(element_type=i32)
            arr8_stream_i8 = emithls.ArrayType.get(element_type=stream_i8, shape=[8])
            arr8_stream_i32 = emithls.ArrayType.get(element_type=stream_i32, shape=[8])
            arr128_stream_i8 = emithls.ArrayType.get(
                element_type=stream_i8, shape=[128]
            )
            arr256_stream_i32 = emithls.ArrayType.get(
                element_type=stream_i32, shape=[256]
            )

            build_conv_2d(i8, i32, index_ty, arr8_stream_i8, arr8_stream_i32)
            build_matmul(i8, i32, index_ty, arr128_stream_i8, arr256_stream_i32)

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_resolve_helpers_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
