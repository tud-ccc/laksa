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


def build_basic_process(name: str, output_type, input_type, elem_type) -> dfg.ProcessOp:
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


def build_shaped_process(
    name: str, output_type, input_type, elem_type, shape, index_type
) -> dfg.ProcessOp:
    process_op = dfg.ProcessOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        idx = arith.ConstantOp(index_type, 0)
        elem = dfg.PullOp(elem_type, in_port, [idx.result])
        tensor = dfg.PullAsTensorOp(RankedTensorType.get(shape, elem_type), in_port)
        mref = dfg.PullAsMemRefOp(MemRefType.get(shape, elem_type), in_port)
        dfg.PushTensorOp(tensor.result, out_port)
        dfg.PushMemRefOp(mref.result, out_port)
        dfg.PushOp(elem.result, out_port, [idx.result])
    return process_op


def build_basic_operator(name: str, elem_type) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([elem_type], [elem_type]))
    )
    block = operator_op.body.blocks.append(elem_type)
    with InsertionPoint(block):
        (in_arg,) = block.arguments
        dfg.OutputOp([in_arg])
    return operator_op


def build_basic_region(
    name: str, output_type, input_type, elem_type, callee: str
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = region_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        channel_input, channel_output = dfg.ChannelOp(
            input_type, output_type, elem_type
        ).results
        dfg.InstantiateOp(callee, [in_port], [channel_input])
        dfg.InstantiateOp(callee, [channel_output], [out_port])
    return region_op


def build_offloaded_region(
    name: str, output_type, input_type, callee: str
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        name, TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = region_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        dfg.InstantiateOp(
            callee, [in_port], [out_port], offloaded=dfg.OffloadHardware.MDC
        )
    return region_op


def build_embed_region(
    name: str,
    output_type,
    input_type,
    elem_type,
    process_callee: str,
    region_callee: str,
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(name, TypeAttr.get(FunctionType.get([], [input_type])))
    block = region_op.body.blocks.append(input_type)
    with InsertionPoint(block):
        (out_port,) = block.arguments
        channel_input, channel_output = dfg.ChannelOp(
            input_type, output_type, elem_type, buffer_size=2
        ).results
        dfg.InstantiateOp(process_callee, [channel_output], [out_port])
        dfg.EmbedOp(region_callee, [channel_output], [channel_input])
    return region_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            index_type = IndexType.get()
            scalar_output = dfg.OutputType.get(element_type=i32)
            scalar_input = dfg.InputType.get(element_type=i32)

            build_basic_process("basic_process", scalar_output, scalar_input, i32)

            shape = [2]
            build_shaped_process(
                "shaped_process",
                dfg.OutputType.get(element_type=i32, shape=shape),
                dfg.InputType.get(element_type=i32, shape=shape),
                i32,
                shape,
                index_type,
            )

            build_basic_operator("basic_operator", i32)

            build_basic_region(
                "basic_region", scalar_output, scalar_input, i32, "basic_process"
            )

            build_offloaded_region(
                "offloaded_region", scalar_output, scalar_input, "basic_process"
            )

            build_embed_region(
                "embed_region",
                scalar_output,
                scalar_input,
                i32,
                "basic_process",
                "basic_region",
            )

    print(module)


if __name__ == "__main__":
    main()
