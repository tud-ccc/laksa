# Recreates ops from test/Dialect/Linalg/Transforms/soft-transpose.mlir using
# the Python bindings and runs the linalg-soft-transpose pass on them.

import numpy as np

from mlir_laksa.ir import (
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

def build_transpose(i32) -> func.FuncOp:
    tensor2x4 = RankedTensorType.get([2, 4], i32)
    tensor4x2 = RankedTensorType.get([4, 2], i32)

    func_op = func.FuncOp("transpose", ([], [tensor4x2]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        cst = arith.ConstantOp(
            DenseElementsAttr.get(
                np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
                type=tensor2x4,
            ),
            None,
        )
        empty = tensor.EmptyOp([4, 2], i32)
        op = linalg.transpose(cst.result, outs=[empty.result], permutation=[1, 0])
        func.ReturnOp([op.result])
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32 = IntegerType.get_signless(32)
            build_transpose(i32)

        print(module)

        pm = PassManager(context=ctx)
        linalg_ext.add_soft_transpose_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
