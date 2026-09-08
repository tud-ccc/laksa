# Recreates ops from test/Dialect/EmitHLS/IR/canonicalize.mlir using the
# Python bindings and runs the canonicalize pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import arith, emithls, func, memref
from mlir_laksa.passmanager import PassManager


def build_includes() -> None:
    emithls.IncludeOp("hls_stream")
    emithls.IncludeOp("cstdin")
    emithls.IncludeOp("ap_int")
    emithls.IncludeOp("ap_int")


def build_pipeline_dedup(i8, memref_i32, memref_i8) -> func.FuncOp:
    func_op = func.FuncOp("pipeline_dedup", ([memref_i32, memref_i8], []))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        lo = arith.ConstantOp(IndexType.get(), 0)
        hi = arith.ConstantOp(IndexType.get(), 7)
        outer = emithls.ForOp(0, 4, 1)
        outer_block = outer.body.blocks.append(IndexType.get())
        with InsertionPoint(outer_block):
            (i,) = outer_block.arguments
            emithls.PragmaPipelineOp(interval=1, style=1)
            inner = emithls.ForOp(0, 4, 1)
            inner_block = inner.body.blocks.append(IndexType.get())
            with InsertionPoint(inner_block):
                emithls.PragmaPipelineOp(interval=1, style=1)
                v = memref.LoadOp(arg0, [i])
                w = emithls.ArithDataRangeOp(i8, v.result, lo.result, hi.result)
                memref.StoreOp(w, arg1, [i])
        func.ReturnOp([])
    return func_op


def build_stream_pipeline_dedup(i8, stream_i8, array_of_stream_i8) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "stream_pipeline_dedup",
        TypeAttr.get(FunctionType.get([stream_i8, array_of_stream_i8], [])),
    )
    block = func_op.body.blocks.append(stream_i8, array_of_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        v = emithls.StreamReadOp(i8, arg0, [])
        outer = emithls.ForOp(0, 8, 1)
        outer_block = outer.body.blocks.append(IndexType.get())
        with InsertionPoint(outer_block):
            (i,) = outer_block.arguments
            emithls.PragmaPipelineOp(interval=1, style=1)
            inner = emithls.ForOp(0, 8, 1)
            inner_block = inner.body.blocks.append(IndexType.get())
            with InsertionPoint(inner_block):
                emithls.PragmaPipelineOp(interval=1, style=1)
                emithls.StreamWriteOp(v, arg1, [i])
    return func_op


def build_collapse_outer_loop(i8, array_of_stream_i8) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "collapse_outer_loop",
        TypeAttr.get(FunctionType.get([array_of_stream_i8, array_of_stream_i8], [])),
    )
    block = func_op.body.blocks.append(array_of_stream_i8, array_of_stream_i8)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        outer0 = emithls.ForOp(0, 2, 1)
        outer0_block = outer0.body.blocks.append(IndexType.get())
        with InsertionPoint(outer0_block):
            outer1 = emithls.ForOp(0, 4, 1)
            outer1_block = outer1.body.blocks.append(IndexType.get())
            with InsertionPoint(outer1_block):
                inner = emithls.ForOp(0, 8, 1)
                inner_block = inner.body.blocks.append(IndexType.get())
                with InsertionPoint(inner_block):
                    (idx2,) = inner_block.arguments
                    v = emithls.StreamReadOp(i8, arg0, [idx2])
                    square = emithls.ArithMulOp(v.result, v.result)
                    emithls.StreamWriteOp(square.result, arg1, [idx2])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            i8 = IntegerType.get_signless(8)

            build_includes()
            build_pipeline_dedup(i8, MemRefType.get([4], i32), MemRefType.get([4], i8))

            stream_i8 = emithls.StreamType.get(element_type=i8)
            array_of_stream_i8 = emithls.ArrayType.get(
                element_type=stream_i8, shape=[8]
            )
            build_stream_pipeline_dedup(i8, stream_i8, array_of_stream_i8)
            build_collapse_outer_loop(i8, array_of_stream_i8)

        print(module)

        pm = PassManager.parse("builtin.module(canonicalize)", context=ctx)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
