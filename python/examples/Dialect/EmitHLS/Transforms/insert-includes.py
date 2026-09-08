# Recreates ops from test/Dialect/EmitHLS/Transforms/insert-includes.mlir
# using the Python bindings and runs the emithls-insert-includes pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import arith, emithls
from mlir_laksa.passmanager import PassManager


def build_triggers_stream_integer_algo(i32, stream_i32) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "triggers_stream_integer_algo",
        TypeAttr.get(FunctionType.get([stream_i32], [])),
    )
    block = func_op.body.blocks.append(stream_i32)
    with InsertionPoint(block):
        (arg0,) = block.arguments
        v = emithls.StreamReadOp(i32, arg0, []).result
        emithls.ArithMinOp(v, v)
        emithls.ArithMaxOp(v, v)
    return func_op


def build_triggers_index(i32, array4_i32) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "triggers_index", TypeAttr.get(FunctionType.get([array4_i32], []))
    )
    block = func_op.body.blocks.append(array4_i32)
    with InsertionPoint(block):
        (arg0,) = block.arguments
        idx = arith.ConstantOp(IndexType.get(), 0)
        emithls.ArrayReadOp(arg0, [idx.result])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            stream_i32 = emithls.StreamType.get(element_type=i32)
            build_triggers_stream_integer_algo(i32, stream_i32)

            array4_i32 = emithls.ArrayType.get(element_type=i32, shape=[4])
            build_triggers_index(i32, array4_i32)

        print(module)

        pm = PassManager(context=ctx)
        emithls.add_insert_includes_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
