# Recreates ops from test/Dialect/EmitHLS/Transforms/fold-expression.mlir
# using the Python bindings and runs the emithls-fold-expression and
# canonicalize passes on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import emithls
from mlir_laksa.passmanager import PassManager

def const(ty, value):
    return emithls.VariableOp(
        ty, init_number=IntegerAttr.get(ty, value), is_const=True
    )

def build_fold_compare(
    i8, i1, index_ty, arr32_stream_i8
) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "fold_compare",
        TypeAttr.get(FunctionType.get([arr32_stream_i8, arr32_stream_i8], [])),
    )
    block = func_op.body.blocks.append(arr32_stream_i8, arr32_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        c30 = const(index_ty, 30)
        cm1 = const(index_ty, -1)
        c0 = const(index_ty, 0)

        outer = emithls.ForOp(0, 32, 1)
        outer_block = outer.body.blocks.append(index_ty)
        with InsertionPoint(outer_block):
            (idx0,) = outer_block.arguments
            inner = emithls.ForOp(0, 32, 1)
            inner_block = inner.body.blocks.append(index_ty)
            with InsertionPoint(inner_block):
                (idx1,) = inner_block.arguments

                expr_op = emithls.ExpressionOp(i1)
                expr_block = expr_op.body.blocks.append()
                with InsertionPoint(expr_block):
                    add0 = emithls.ArithAddOp(idx0, cm1.result)
                    cmp1 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge,
                        add0.result,
                        c0.result,
                        results=[i1],
                    )
                    mul0 = emithls.ArithMulOp(idx0, cm1.result)
                    add1 = emithls.ArithAddOp(mul0.result, c30.result)
                    cmp2 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge,
                        add1.result,
                        c0.result,
                        results=[i1],
                    )
                    add2 = emithls.ArithAddOp(idx1, cm1.result)
                    cmp3 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge,
                        add2.result,
                        c0.result,
                        results=[i1],
                    )
                    mul1 = emithls.ArithMulOp(idx1, cm1.result)
                    add3 = emithls.ArithAddOp(mul1.result, c30.result)
                    cmp4 = emithls.ArithCmpOp(
                        emithls.CmpPredicate.ge,
                        add3.result,
                        c0.result,
                        results=[i1],
                    )
                    and0 = emithls.ArithLogicalAndOp(
                        [cmp1.result, cmp2.result, cmp3.result, cmp4.result]
                    )
                    emithls.YieldOp(and0.result)

                if_op = emithls.IfOp(expr_op.result)
                then_block = if_op.thenRegion.blocks.append()
                with InsertionPoint(then_block):
                    v = emithls.StreamReadOp(i8, arg0, [idx0])
                    emithls.StreamWriteOp(v.result, arg1, [idx0])
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i1 = IntegerType.get_signless(1)
            index_ty = IndexType.get()
            stream_i8 = emithls.StreamType.get(element_type=i8)
            arr32_stream_i8 = emithls.ArrayType.get(
                element_type=stream_i8, shape=[32]
            )

            build_fold_compare(i8, i1, index_ty, arr32_stream_i8)

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_fold_expression_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
