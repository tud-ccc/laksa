# Recreates ops from test/Dialect/DFG/IR/ops.mlir using the Python bindings

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    RankedTensorType,
    TypeAttr,
)
from mlir_laksa.dialects import arith, dfg
from mlir_laksa.passmanager import PassManager

def build_dead_process(name: str) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(
        name, TypeAttr.get(FunctionType.get([],[]))
    )
    return process_op

def build_dead_operator(name: str) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([],[]))
    )
    return operator_op

def build_relay_process(name: str, output_type, input_type, elem_type) -> dfg.ProcessOp:
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

def build_pipeline_region(
    name: str, output_type, input_type, elem_type, callee: str
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = region_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        channel_input_0, channel_output_0 = dfg.ChannelOp(
            input_type, output_type, elem_type
        ).results
        dfg.InstantiateOp(callee, [in_port], [channel_input_0])
        channel_input_1, channel_output_1 = dfg.ChannelOp(
            input_type, output_type, elem_type
        ).results
        dfg.InstantiateOp(callee, [channel_output_0], [channel_input_1])
        dfg.InstantiateOp(callee, [channel_output_1], [out_port])
    return region_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            build_dead_process("dead_process")
            build_dead_operator("dead_operator")

            i32 = IntegerType.get_signless(32)
            scalar_output = dfg.OutputType.get(element_type=i32)
            scalar_input = dfg.InputType.get(element_type=i32)
            build_relay_process("relay", scalar_output, scalar_input, i32)

            build_pipeline_region("pipeline", scalar_output, scalar_input, i32, "relay")

        print(module)

        pm = PassManager.parse("builtin.module(canonicalize)", context=ctx)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
