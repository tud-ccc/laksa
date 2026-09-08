# Recreates ops from test/Dialect/Linalg/Transforms/scalarize-splat-dense.mlir
# using the Python bindings and runs the linalg-scalarize-splat-dense pass on
# them.

import numpy as np

from mlir_laksa.ir import (
    AffineExpr,
    AffineMap,
    Context,
    DenseElementsAttr,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
)
from mlir_laksa.dialects import arith, func, linalg, linalg_ext, tensor
from mlir_laksa.passmanager import PassManager


def build_scalarize(i8) -> func.FuncOp:
    tensor4 = RankedTensorType.get([4], i8)
    id_map = AffineMap.get(1, 0, [AffineExpr.get_dim(0)])

    func_op = func.FuncOp("scalarize", ([tensor4], [tensor4, tensor4]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        (arg0,) = block.arguments
        splat = arith.ConstantOp(
            DenseElementsAttr.get(np.array([2], dtype=np.int8), type=tensor4), None
        )
        dense = arith.ConstantOp(
            DenseElementsAttr.get(np.array([1, 2, 3, 4], dtype=np.int8), type=tensor4),
            None,
        )
        empty0 = tensor.EmptyOp([4], i8)
        empty1 = tensor.EmptyOp([4], i8)
        generic = linalg.GenericOp(
            [tensor4, tensor4],
            [splat.result, dense.result, arg0],
            [empty0.result, empty1.result],
            [id_map] * 5,
            ["parallel"],
        )
        generic_block = generic.regions[0].blocks.append(i8, i8, i8, i8, i8)
        with InsertionPoint(generic_block):
            in_, in_0, in_1, _out, _out0 = generic_block.arguments
            sum_ = arith.AddIOp(in_, in_0)
            prod = arith.MulIOp(in_, in_1)
            linalg.YieldOp([sum_.result, prod.result])
        func.ReturnOp([generic.result_tensors[0], generic.result_tensors[1]])
    return func_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            build_scalarize(i8)

        print(module)

        pm = PassManager(context=ctx)
        linalg_ext.add_scalarize_splat_dense_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
