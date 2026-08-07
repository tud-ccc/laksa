# Recreates ops from
# test/Dialect/Func/Transforms/outline-computation-leaf.mlir using the Python
# bindings and runs the func-outline-computation-leaf pass on them.

from mlir_laksa.ir import (
    AffineExpr,
    AffineMap,
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    Location,
    Module,
    RankedTensorType,
)
from mlir_laksa.dialects import arith, func, func_ext, linalg, tensor
from mlir_laksa.passmanager import PassManager

def build_test(f32) -> func.FuncOp:
    tensor4 = RankedTensorType.get([4], f32)
    tensor4x4 = RankedTensorType.get([4, 4], f32)
    tensor1x4 = RankedTensorType.get([1, 4], f32)
    tensor6x6 = RankedTensorType.get([6, 6], f32)
    tensor36 = RankedTensorType.get([36], f32)

    scalar_map = AffineMap.get(1, 0, [])
    id_map = AffineMap.get(1, 0, [AffineExpr.get_dim(0)])

    func_op = func.FuncOp("test", ([tensor4, tensor4x4], [tensor1x4, tensor36]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        arg0, arg1 = block.arguments

        zero = arith.ConstantOp(f32, 0.0)
        empty_acc = tensor.EmptyOp([4], f32)
        filled = linalg.GenericOp(
            [tensor4],
            [zero.result],
            [empty_acc.result],
            [scalar_map, id_map],
            ["parallel"],
        )
        filled_block = filled.regions[0].blocks.append(f32, f32)
        with InsertionPoint(filled_block):
            (in_, _out) = filled_block.arguments
            linalg.YieldOp([in_])

        empty0 = tensor.EmptyOp([4], f32)
        sum_op = linalg.GenericOp(
            [tensor4],
            [arg0, filled.results[0]],
            [empty0.result],
            [id_map, id_map, id_map],
            ["parallel"],
        )
        sum_block = sum_op.regions[0].blocks.append(f32, f32, f32)
        with InsertionPoint(sum_block):
            in0, in1, _out = sum_block.arguments
            v = arith.AddFOp(in0, in1)
            linalg.YieldOp([v.result])

        expanded = tensor.ExpandShapeOp(
            tensor1x4, sum_op.results[0], [[0, 1]], [], [1, 4]
        )

        pad_cst = arith.ConstantOp(f32, 1.0)
        padded = tensor.PadOp(tensor6x6, arg1, [], [], [1, 1], [1, 1])
        pad_block = padded.region.blocks.append(IndexType.get(), IndexType.get())
        with InsertionPoint(pad_block):
            tensor.YieldOp(pad_cst.result)

        collapsed = tensor.CollapseShapeOp(tensor36, padded.result, [[0, 1]])

        func.ReturnOp([expanded.result, collapsed.result])
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f32 = F32Type.get()
            build_test(f32)

        print(module)

        pm = PassManager(context=ctx)
        func_ext.add_outline_computation_leaf_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
