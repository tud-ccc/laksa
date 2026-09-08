# Recreates ops from test/Dialect/DFG/Transforms/inline-embed-region.mlir
# using the Python bindings and runs the dfg-inline-embed-region pass.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import arith, dfg
from mlir_laksa.passmanager import PassManager


def build_source_process(name: str, input_type, elem_type) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(name, TypeAttr.get(FunctionType.get([], [input_type])))
    block = process_op.body.blocks.append(input_type)
    with InsertionPoint(block):
        (out_port,) = block.arguments
        zero = arith.ConstantOp(elem_type, 0)
        dfg.PushOp(zero.result, out_port, [])
    return process_op


def build_relay_process(name: str, output_type, input_type, elem_type) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        token = dfg.PullOp(elem_type, in_port, [])
        dfg.PushOp(token.result, out_port, [])
    return process_op


def build_child_region(
    name: str, output_type, input_type, elem_type, source_callee: str, relay_callee: str
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(name, TypeAttr.get(FunctionType.get([], [input_type])))
    block = region_op.body.blocks.append(input_type)
    with InsertionPoint(block):
        (out_port,) = block.arguments
        channel_input, channel_output = dfg.ChannelOp(
            input_type, output_type, elem_type
        ).results
        dfg.InstantiateOp(source_callee, [], [channel_input])
        dfg.InstantiateOp(relay_callee, [channel_output], [out_port])
    return region_op


def build_parent_region(
    name: str, output_type, input_type, relay_callee: str, child_callee: str
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type, input_type]))
    )
    block = region_op.body.blocks.append(output_type, input_type, input_type)
    with InsertionPoint(block):
        in_port, out0, out1 = block.arguments
        dfg.InstantiateOp(relay_callee, [in_port], [out1])
        dfg.EmbedOp(child_callee, [], [out0])
    return region_op


def build_standalone_region(name: str, input_type, child_callee: str) -> dfg.RegionOp:
    region_op = dfg.RegionOp(name, TypeAttr.get(FunctionType.get([], [input_type])))
    block = region_op.body.blocks.append(input_type)
    with InsertionPoint(block):
        (out_port,) = block.arguments
        dfg.EmbedOp(child_callee, [], [out_port])
    return region_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            scalar_output = dfg.OutputType.get(element_type=i32)
            scalar_input = dfg.InputType.get(element_type=i32)

            build_source_process("source", scalar_input, i32)
            build_relay_process("relay", scalar_output, scalar_input, i32)

            build_child_region(
                "child", scalar_output, scalar_input, i32, "source", "relay"
            )
            build_parent_region("parent", scalar_output, scalar_input, "relay", "child")
            build_standalone_region("standalone", scalar_input, "child")

        print(module)

        pm = PassManager(context=ctx)
        dfg.add_inline_embed_region_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
