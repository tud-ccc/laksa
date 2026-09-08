# Recreates ops from test/Dialect/EmitHLS/Transforms/loop-fusion.mlir using
# the Python bindings and runs the emithls-loop-fusion and canonicalize
# passes on them.

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


def build_pass_through(
    i8, index_ty, arr128_stream_i8, arr256_stream_i32
) -> emithls.FuncOp:
    array_i8 = emithls.ArrayType.get(element_type=i8, shape=[128])
    func_op = emithls.FuncOp(
        "pass_through",
        TypeAttr.get(FunctionType.get([arr128_stream_i8, arr256_stream_i32], [])),
    )
    block = func_op.body.blocks.append(arr128_stream_i8, arr256_stream_i32)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        var_array_0 = emithls.VariableOp(array_i8)

        for0 = emithls.ForOp(0, 512, 1)
        for0_block = for0.body.blocks.append(index_ty)
        with InsertionPoint(for0_block):
            alloc = memref.AllocOp(MemRefType.get([128], i8), [], [])
            for1 = emithls.ForOp(0, 128, 1)
            for1_block = for1.body.blocks.append(index_ty)
            with InsertionPoint(for1_block):
                (idx1,) = for1_block.arguments
                read0 = emithls.StreamReadOp(i8, arg0, [idx1])
                memref.StoreOp(read0.result, alloc.result, [idx1])

            for1b = emithls.ForOp(0, 128, 1)
            for1b_block = for1b.body.blocks.append(index_ty)
            with InsertionPoint(for1b_block):
                (idx1b,) = for1b_block.arguments
                load0 = memref.LoadOp(alloc.result, [idx1b])
                emithls.UpdateOp(var_array_0.variable, [idx1b], load0.result, [])
    return func_op


def build_head_toe(i8, index_ty, arr8_stream_i8, arr8_stream_i32) -> emithls.FuncOp:
    win_type = emithls.ArrayType.get(element_type=i8, shape=[8, 3, 3])
    func_op = emithls.FuncOp(
        "head_toe",
        TypeAttr.get(FunctionType.get([arr8_stream_i8, arr8_stream_i32], [])),
    )
    block = func_op.body.blocks.append(arr8_stream_i8, arr8_stream_i32)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        var_array_0 = emithls.VariableOp(win_type)
        c0_i8 = const(i8, 0)
        c1_index = const(index_ty, 1)
        c0_index = const(index_ty, 0)
        c2_index = const(index_ty, 2)

        for0 = emithls.ForOp(0, 32, 1)
        for0_block = for0.body.blocks.append(index_ty)
        with InsertionPoint(for0_block):
            (idx0,) = for0_block.arguments
            for1 = emithls.ForOp(0, 32, 1)
            for1_block = for1.body.blocks.append(index_ty)
            with InsertionPoint(for1_block):
                alloc = memref.AllocOp(MemRefType.get([8], i8), [], [])
                for2 = emithls.ForOp(0, 8, 1)
                for2_block = for2.body.blocks.append(index_ty)
                with InsertionPoint(for2_block):
                    (idx2,) = for2_block.arguments
                    read0 = emithls.StreamReadOp(i8, arg0, [idx2])
                    memref.StoreOp(read0.result, alloc.result, [idx2])

                for2b = emithls.ForOp(0, 8, 1)
                for2b_block = for2b.body.blocks.append(index_ty)
                with InsertionPoint(for2b_block):
                    (idx2b,) = for2b_block.arguments
                    load0 = memref.LoadOp(alloc.result, [idx2b])

                    i1 = IntegerType.get_signless(1)
                    expr_op = emithls.ExpressionOp(i1)
                    expr_block = expr_op.body.blocks.append()
                    with InsertionPoint(expr_block):
                        cmp0 = emithls.ArithCmpOp(
                            emithls.CmpPredicate.eq,
                            idx0,
                            c0_index.result,
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
                            if_op = emithls.IfOp(expr_op.result)
                            then_block = if_op.thenRegion.blocks.append()
                            with InsertionPoint(then_block):
                                emithls.UpdateOp(
                                    var_array_0.variable,
                                    [idx2b, idx3, idx4],
                                    c0_i8.result,
                                    [],
                                )
                            else_block = if_op.elseRegion.blocks.append()
                            with InsertionPoint(else_block):
                                index_expr = emithls.ExpressionOp(index_ty)
                                index_expr_block = index_expr.body.blocks.append()
                                with InsertionPoint(index_expr_block):
                                    add0 = emithls.ArithAddOp(idx4, c1_index.result)
                                    emithls.YieldOp(add0.result)
                                emithls.UpdateOp(
                                    var_array_0.variable,
                                    [idx2b, idx3, idx4],
                                    var_array_0.variable,
                                    [idx2b, idx3, index_expr.result],
                                )

                    emithls.UpdateOp(
                        var_array_0.variable,
                        [idx2b, c2_index.result, c2_index.result],
                        load0.result,
                        [],
                    )
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
            arr128_stream_i8 = emithls.ArrayType.get(
                element_type=stream_i8, shape=[128]
            )
            arr256_stream_i32 = emithls.ArrayType.get(
                element_type=stream_i32, shape=[256]
            )
            arr8_stream_i8 = emithls.ArrayType.get(element_type=stream_i8, shape=[8])
            arr8_stream_i32 = emithls.ArrayType.get(element_type=stream_i32, shape=[8])

            build_pass_through(i8, index_ty, arr128_stream_i8, arr256_stream_i32)
            build_head_toe(i8, index_ty, arr8_stream_i8, arr8_stream_i32)

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_loop_fusion_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
