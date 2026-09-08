# Recreates ops from test/Dialect/EmitHLS/Transforms/fuse-operator.mlir using
# the Python bindings and runs the emithls-fuse-operator pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import emithls
from mlir_laksa.passmanager import PassManager


def build_fuse(i32) -> emithls.FuncOp:
    func_op = emithls.FuncOp("fuse", TypeAttr.get(FunctionType.get([], [])))
    block = func_op.body.blocks.append()
    with InsertionPoint(block):
        cst = emithls.VariableOp(
            i32, init_number=IntegerAttr.get(i32, 2), is_const=True
        )
        var0 = emithls.VariableOp(i32)
        var1 = emithls.VariableOp(i32)
        add0 = emithls.ArithAddOp(var0.variable, var1.variable)
        emithls.UpdateOp(var0.variable, [], add0.result, [])
        sub0 = emithls.ArithSubOp(var0.variable, var1.variable)
        emithls.UpdateOp(var1.variable, [], sub0.result, [])
        mul0 = emithls.ArithMulOp(add0.result, sub0.result)
        emithls.UpdateOp(var0.variable, [], mul0.result, [])
        add1 = emithls.ArithAddOp(add0.result, cst.variable)
        emithls.UpdateOp(var1.variable, [], add1.result, [])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            build_fuse(i32)

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_fuse_operator_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
