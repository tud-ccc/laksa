# Recreates ops from test/Conversion/index-to-emithls.mlir using the Python
# bindings and runs the convert-index-to-emithls pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import emithls, index
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def build_index(i32) -> emithls.FuncOp:
    func_op = emithls.FuncOp("index", TypeAttr.get(FunctionType.get([], [])))
    block = func_op.body.blocks.append()
    with InsertionPoint(block):
        idx = emithls.VariableOp(IndexType.get()).variable
        zero = index.ConstantOp(2)
        index.AddOp(idx, zero.result)
        index.SubOp(idx, zero.result)
        index.MulOp(idx, zero.result)
        index.RemUOp(idx, zero.result)
        index.RemSOp(idx, zero.result)
        index.ShlOp(idx, zero.result)
        index.ShrUOp(idx, zero.result)
        index.ShrSOp(idx, zero.result)
        index.MaxUOp(idx, zero.result)
        index.MaxSOp(idx, zero.result)
        index.MinUOp(idx, zero.result)
        index.MinSOp(idx, zero.result)
        index.AndOp(idx, zero.result)
        index.OrOp(idx, zero.result)
        index.CastSOp(i32, idx)
        index.CastUOp(i32, idx)
        index.CmpOp(index.IndexCmpPredicate.EQ, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.NE, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.SLT, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.SLE, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.SGT, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.SGE, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.ULT, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.ULE, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.UGT, idx, zero.result)
        index.CmpOp(index.IndexCmpPredicate.UGE, idx, zero.result)
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            build_index(i32)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_index_to_emithls_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
