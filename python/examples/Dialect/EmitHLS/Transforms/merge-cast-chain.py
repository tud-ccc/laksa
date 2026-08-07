# Recreates ops from test/Dialect/EmitHLS/Transforms/merge-cast-chain.mlir
# using the Python bindings and runs the emithls-merge-cast-chain and
# canonicalize passes on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import emithls
from mlir_laksa.passmanager import PassManager

def build_merge_chain(i8, i16, i32, i64, stream_i8, stream_i64) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "merge_chain", TypeAttr.get(FunctionType.get([stream_i8, stream_i64], []))
    )
    block = func_op.body.blocks.append(stream_i8, stream_i64)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        v0 = emithls.StreamReadOp(i8, arg0, []).result
        v1 = emithls.ArithCastOp(i16, v0).result
        v2 = emithls.ArithCastOp(i32, v1).result
        v3 = emithls.ArithCastOp(i64, v2).result
        emithls.StreamWriteOp(v3, arg1, [])
    return func_op

def build_merge_multiout(
    i8, i16, i32, i64, stream_i8, stream_i32, stream_i64
) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "merge_multiout",
        TypeAttr.get(FunctionType.get([stream_i8, stream_i32, stream_i64], [])),
    )
    block = func_op.body.blocks.append(stream_i8, stream_i32, stream_i64)
    with InsertionPoint(block):
        arg0, arg1, arg2 = block.arguments
        v0 = emithls.StreamReadOp(i8, arg0, []).result
        v1 = emithls.ArithCastOp(i16, v0).result
        v2 = emithls.ArithCastOp(i32, v1).result
        v3 = emithls.ArithCastOp(i64, v2).result
        emithls.StreamWriteOp(v2, arg1, [])
        emithls.StreamWriteOp(v3, arg2, [])
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i16 = IntegerType.get_signless(16)
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            stream_i8 = emithls.StreamType.get(element_type=i8)
            stream_i32 = emithls.StreamType.get(element_type=i32)
            stream_i64 = emithls.StreamType.get(element_type=i64)

            build_merge_chain(i8, i16, i32, i64, stream_i8, stream_i64)
            build_merge_multiout(
                i8, i16, i32, i64, stream_i8, stream_i32, stream_i64
            )

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_merge_cast_chain_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
