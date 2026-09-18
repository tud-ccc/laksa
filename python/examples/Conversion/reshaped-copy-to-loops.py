# Recreates ops from test/Conversion/reshaped-copy-to-loops.mlir using the
# Python bindings and runs the convert-reshaped-copy-to-loops pass on them.

from mlir_laksa.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    StridedLayoutAttr,
)
from mlir_laksa.dialects import func, memref
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def build_copy_through_cast(
    name, source_type, cast_type, target_type, offset, sizes, strides
) -> func.FuncOp:
    func_op = func.FuncOp(name, FunctionType.get([source_type, target_type], []))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        source, target = block.arguments
        cast = memref.ReinterpretCastOp(
            cast_type, source, [], [], [], [offset], sizes, strides
        )
        memref.CopyOp(cast.result, target)
        func.ReturnOp([])
    return func_op


def build_copy_into_cast(
    name, source_type, cast_type, target_type, offset, sizes, strides
) -> func.FuncOp:
    func_op = func.FuncOp(name, FunctionType.get([source_type, target_type], []))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        source, target = block.arguments
        cast = memref.ReinterpretCastOp(
            cast_type, target, [], [], [], [offset], sizes, strides
        )
        memref.CopyOp(source, cast.result)
        func.ReturnOp([])
    return func_op


def build_copy_into_subview(
    name, source_type, target_type, offsets, sizes, strides
) -> func.FuncOp:
    func_op = func.FuncOp(name, FunctionType.get([source_type, target_type], []))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        source, target = block.arguments
        view = memref.subview(target, offsets, sizes, strides)
        memref.CopyOp(source, view)
        func.ReturnOp([])
    return func_op


def build_copy_from_subview(
    name, source_type, target_type, offsets, sizes, strides
) -> func.FuncOp:
    func_op = func.FuncOp(name, FunctionType.get([source_type, target_type], []))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        source, target = block.arguments
        view = memref.subview(source, offsets, sizes, strides)
        memref.CopyOp(view, target)
        func.ReturnOp([])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)

            build_copy_through_cast(
                "unit_dims",
                MemRefType.get([4, 1, 1, 8], i8),
                MemRefType.get([1, 1, 4, 8], i8),
                MemRefType.get([1, 1, 4, 8], i8),
                0,
                [1, 1, 4, 8],
                [32, 32, 8, 1],
            )
            build_copy_through_cast(
                "regrouped",
                MemRefType.get([2, 12], i8),
                MemRefType.get([6, 4], i8),
                MemRefType.get([6, 4], i8),
                0,
                [6, 4],
                [4, 1],
            )
            build_copy_through_cast(
                "incompatible",
                MemRefType.get([6, 4], i8),
                MemRefType.get([8, 3], i8),
                MemRefType.get([8, 3], i8),
                0,
                [8, 3],
                [3, 1],
            )
            build_copy_through_cast(
                "offset",
                MemRefType.get([8, 8], i8),
                MemRefType.get([4, 8], i8, StridedLayoutAttr.get(32, [8, 1])),
                MemRefType.get([4, 8], i8),
                32,
                [4, 8],
                [8, 1],
            )
            build_copy_into_cast(
                "padded",
                MemRefType.get([1, 2, 2, 4], i8),
                MemRefType.get(
                    [1, 2, 2, 4], i8, StridedLayoutAttr.get(20, [64, 16, 4, 1])
                ),
                MemRefType.get([1, 4, 4, 4], i8),
                20,
                [1, 2, 2, 4],
                [64, 16, 4, 1],
            )
            build_copy_through_cast(
                "stepped",
                MemRefType.get([8, 8], i8),
                MemRefType.get([4, 8], i8, StridedLayoutAttr.get(0, [16, 1])),
                MemRefType.get([4, 8], i8),
                0,
                [4, 8],
                [16, 1],
            )
            build_copy_into_subview(
                "subview_target",
                MemRefType.get([2, 3], i8),
                MemRefType.get([5, 7], i8),
                [1, 2],
                [2, 3],
                [1, 1],
            )
            build_copy_from_subview(
                "strided_subview_source",
                MemRefType.get([6, 8], i8),
                MemRefType.get([2, 3], i8),
                [1, 0],
                [2, 3],
                [2, 2],
            )

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_reshaped_copy_to_loops_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
