# Recreates ops from test/Conversion/memref-pad-to-laksa-loops.mlir using the
# Python bindings and runs the convert-memref-pad-to-laksa-loops pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import builtin, dfg, memref
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def build_main_node_2(i8) -> dfg.ProcessOp:
    """memref-token port: the filler is a small memref.alloc carrying the
    `fill_with` attribute, to be pushed via dfg.push_memref in the else
    branch the pass generates."""
    port_shape = [1, 8]
    real_shape = [1, 30, 30, 8]
    padded_shape = [1, 32, 32, 8]

    in_type = dfg.OutputType.get(element_type=i8, shape=port_shape)
    out_type = dfg.InputType.get(element_type=i8, shape=port_shape)
    process_op = dfg.ProcessOp(
        "pad_i8", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(port_shape, i8), in_port)
            widened = builtin.UnrealizedConversionCastOp(
                [MemRefType.get(real_shape, i8)], [token.result]
            )
            alloc = memref.AllocOp(
                MemRefType.get(padded_shape, i8), [], [], alignment=64
            )
            alloc.operation.attributes["fill_with"] = IntegerAttr.get(i8, -128)

            subview = memref.subview(
                alloc.result, [0, 1, 1, 0], [1, 30, 30, 8], [1, 1, 1, 1]
            )
            memref.CopyOp(widened.results[0], subview)

            narrowed = builtin.UnrealizedConversionCastOp(
                [MemRefType.get(port_shape, i8)], [alloc.result]
            )
            dfg.PushMemRefOp(narrowed.results[0], out_port)
    return process_op


def build_pad_scalar_i8(i8) -> dfg.ProcessOp:
    """scalar-token port: there's no buffer to stash `fill_with` on, so the
    pass materializes the filler as an arith.constant instead, pushed via
    dfg.push in the else branch."""
    in_type = dfg.OutputType.get(element_type=i8)
    out_type = dfg.InputType.get(element_type=i8)
    process_op = dfg.ProcessOp(
        "pad_scalar_i8", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullOp(i8, in_port, [])
            widened = builtin.UnrealizedConversionCastOp(
                [MemRefType.get([100], i8)], [token.token]
            )
            alloc = memref.AllocOp(MemRefType.get([104], i8), [], [], alignment=64)
            alloc.operation.attributes["fill_with"] = IntegerAttr.get(i8, -128)

            subview = memref.subview(alloc.result, [4], [100], [1])
            memref.CopyOp(widened.results[0], subview)

            narrowed = builtin.UnrealizedConversionCastOp([i8], [alloc.result])
            dfg.PushOp(narrowed.results[0], out_port, [])
    return process_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)

            build_main_node_2(i8)
            build_pad_scalar_i8(i8)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_memref_pad_to_laksa_loops_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
