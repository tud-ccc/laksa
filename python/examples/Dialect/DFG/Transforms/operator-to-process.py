# Recreates ops from test/Dialect/DFG/Transforms/operator-to-process.mlir
# using the Python bindings and runs the dfg-operator-to-process pass.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    RankedTensorType,
    TypeAttr,
)
from mlir_laksa.dialects import dfg
from mlir_laksa.passmanager import PassManager

def build_scalar_operator(name: str, elem_type) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([elem_type], [elem_type, elem_type]))
    )
    block = operator_op.body.blocks.append(elem_type)
    with InsertionPoint(block):
        (in_arg,) = block.arguments
        dfg.OutputOp([in_arg, in_arg])
    return operator_op

def build_shaped_operator(name: str, shaped_type) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([shaped_type], [shaped_type]))
    )
    block = operator_op.body.blocks.append(shaped_type)
    with InsertionPoint(block):
        (in_arg,) = block.arguments
        dfg.OutputOp([in_arg])
    return operator_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            shape = [4]

            build_scalar_operator("scalar_op", i32)
            build_shaped_operator("memref_op", MemRefType.get(shape, i32))
            build_shaped_operator("tensor_op", RankedTensorType.get(shape, i32))

        print(module)

        pm = PassManager(context=ctx)
        dfg.add_operator_to_process_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
