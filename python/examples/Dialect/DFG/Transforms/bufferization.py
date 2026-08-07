# Recreates ops from test/Dialect/DFG/Transforms/bufferization.mlir using the
# Python bindings and runs the one-shot-bufferize pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
    TypeAttr,
)
from mlir_laksa.dialects import dfg
from mlir_laksa.passmanager import PassManager

def build_identity_process(name: str, output_type, input_type, elem_type, shape) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        tensor = dfg.PullAsTensorOp(RankedTensorType.get(shape, elem_type), in_port)
        dfg.PushTensorOp(tensor.result, out_port)
    return process_op

def build_passthrough_process(name: str, output_type, input_type, elem_type) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullOp(elem_type, in_port, [])
            dfg.PushOp(token.result, out_port, [])
    return process_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            shape = [4]
            build_identity_process(
                "identity",
                dfg.OutputType.get(element_type=i32, shape=shape),
                dfg.InputType.get(element_type=i32, shape=shape),
                i32,
                shape,
            )

            scalar_output = dfg.OutputType.get(element_type=i32)
            scalar_input = dfg.InputType.get(element_type=i32)
            build_passthrough_process("passthrough", scalar_output, scalar_input, i32)

        print(module)

        pm = PassManager.parse("builtin.module(one-shot-bufferize)", context=ctx)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
