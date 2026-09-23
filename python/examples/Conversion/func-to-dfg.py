# Recreates ops from test/Conversion/func-to-dfg.mlir using the Python
# bindings and runs the convert-func-to-dfg pass on them.

from mlir_laksa.ir import (
    Context,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
    UnitAttr,
)
from mlir_laksa.dialects import func
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def build_stream(i32, tensor2) -> func.FuncOp:
    func_op = func.FuncOp("stream", ([i32, tensor2], [i32, tensor2]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        func.ReturnOp([arg0, arg1])
    return func_op


def build_call(i32, tensor2) -> func.FuncOp:
    func_op = func.FuncOp("call", ([i32, tensor2], [i32, tensor2]))
    func_op.operation.attributes["laksa.root"] = UnitAttr.get()
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        call0 = func.CallOp([i32, tensor2], "stream", [arg0, arg1])
        call1 = func.CallOp(
            [i32, tensor2], "stream", [call0.results[0], call0.results[1]]
        )
        func.ReturnOp([call1.results[0], call1.results[1]])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            tensor2 = RankedTensorType.get([2], i32)
            build_stream(i32, tensor2)
            build_call(i32, tensor2)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_func_to_dfg_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
