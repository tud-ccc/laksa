# Recreates ops from test/Dialect/EmitHLS/IR/ops.mlir using the Python bindings

from mlir_laksa.ir import (
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import emithls


def build_basic_vars(i32, ptr_type) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "basic_vars", TypeAttr.get(FunctionType.get([ptr_type], []))
    )
    block = func_op.body.blocks.append(ptr_type)
    with InsertionPoint(block):
        (arg0,) = block.arguments
        zero = emithls.VariableOp(
            i32, init_number=IntegerAttr.get(i32, 0), is_const=True
        )

        emithls.VariableOp(i32, init_value=zero.variable)
        loop = emithls.ForOp(0, 4, 1)
        loop_block = loop.body.blocks.append(IndexType.get())
        with InsertionPoint(loop_block):
            emithls.PragmaPipelineOp(interval=1, style=1)
            (i,) = loop_block.arguments
            val = emithls.ArrayPointerReadOp(arg0, [i])
            emithls.ArrayPointerWriteOp(val, arg0, [i])
    return func_op


def build_array_ops(i32, array_type) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "array_ops", TypeAttr.get(FunctionType.get([array_type], []))
    )
    block = func_op.body.blocks.append(array_type)
    with InsertionPoint(block):
        (arg0,) = block.arguments
        loop = emithls.ForOp(0, 8, 1)
        loop_block = loop.body.blocks.append(IndexType.get())
        with InsertionPoint(loop_block):
            (i,) = loop_block.arguments
            val = emithls.ArrayReadOp(arg0, [i])
            emithls.ArrayWriteOp(val, arg0, [i])
    return func_op


def build_stream_ops(i32, stream_type, array_of_stream_type) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "stream_ops",
        TypeAttr.get(FunctionType.get([stream_type, array_of_stream_type], [])),
    )
    block = func_op.body.blocks.append(stream_type, array_of_stream_type)
    with InsertionPoint(block):
        arg0, arg1 = block.arguments
        val = emithls.StreamReadOp(i32, arg0, [])
        loop = emithls.ForOp(0, 4, 1)
        loop_block = loop.body.blocks.append(IndexType.get())
        with InsertionPoint(loop_block):
            (i,) = loop_block.arguments
            emithls.StreamWriteOp(val, arg1, [i])
        s = emithls.VariableOp(stream_type)
        emithls.PragmaStreamOp(s.variable, depth=8)
    return func_op


def build_pragma_ops(array_type) -> emithls.FuncOp:
    func_op = emithls.FuncOp(
        "pragma_ops", TypeAttr.get(FunctionType.get([array_type], []))
    )
    block = func_op.body.blocks.append(array_type)
    with InsertionPoint(block):
        (arg0,) = block.arguments
        emithls.PragmaInlineOp()
        emithls.PragmaArrayPartitionOp(arg0, part_type=0, part_factor=2, part_dim=1)
        emithls.PragmaBindStorageOp(arg0, storage_type=2, storage_impl=0)
    return func_op


def main() -> None:
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            ptr_type = emithls.PointerType.get(element_type=i32)
            build_basic_vars(i32, ptr_type)

            array8_type = emithls.ArrayType.get(element_type=i32, shape=[8])
            build_array_ops(i32, array8_type)

            stream_type = emithls.StreamType.get(element_type=i32)
            array_of_stream_type = emithls.ArrayType.get(
                element_type=stream_type, shape=[4]
            )
            build_stream_ops(i32, stream_type, array_of_stream_type)

            array16_type = emithls.ArrayType.get(element_type=i32, shape=[16])
            build_pragma_ops(array16_type)

        print(module)


if __name__ == "__main__":
    main()
