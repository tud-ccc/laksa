# Recreates ops from test/Dialect/DFG/Transforms/channel-fanout-expansion.mlir
# using the Python bindings and runs the dfg-channel-fanout-expansion pass.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import dfg
from mlir_laksa.passmanager import PassManager

def build_producer(name: str, i32, i64) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([i32, i64], [i32, i64]))
    )
    block = operator_op.body.blocks.append(i32, i64)
    with InsertionPoint(block):
        in0, in1 = block.arguments
        dfg.OutputOp([in0, in1])
    return operator_op

def build_consumer(name: str, elem_type) -> dfg.OperatorOp:
    operator_op = dfg.OperatorOp(
        name, TypeAttr.get(FunctionType.get([elem_type], []))
    )
    block = operator_op.body.blocks.append(elem_type)
    with InsertionPoint(block):
        dfg.OutputOp([])
    return operator_op

def build_fanout_region(
    name: str,
    output_i32,
    output_i64,
    i32,
    i64,
    producer_callee: str,
    consumer_i32_callee: str,
    consumer_i64_callee: str,
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        name, TypeAttr.get(FunctionType.get([output_i32, output_i64], []))
    )
    block = region_op.body.blocks.append(output_i32, output_i64)
    with InsertionPoint(block):
        src0, src1 = block.arguments
        fanout_input, fanout_output = dfg.ChannelOp(
            dfg.InputType.get(element_type=i32), output_i32, i32
        ).results
        single_input, single_output = dfg.ChannelOp(
            dfg.InputType.get(element_type=i64), output_i64, i64
        ).results
        dfg.InstantiateOp(
            producer_callee, [src0, src1], [fanout_input, single_input]
        )
        dfg.InstantiateOp(consumer_i32_callee, [fanout_output], [])
        dfg.InstantiateOp(consumer_i32_callee, [fanout_output], [])
        dfg.InstantiateOp(consumer_i64_callee, [single_output], [])
    return region_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)

            build_producer("producer", i32, i64)
            build_consumer("consumer_i32", i32)
            build_consumer("consumer_i64", i64)

            build_fanout_region(
                "fanout2",
                dfg.OutputType.get(element_type=i32),
                dfg.OutputType.get(element_type=i64),
                i32,
                i64,
                "producer",
                "consumer_i32",
                "consumer_i64",
            )

        print(module)

        pm = PassManager(context=ctx)
        dfg.add_channel_fanout_expansion_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
