# Recreates ops from test/Dialect/Linalg/Transforms/map-to-generic.mlir using
# the Python bindings and runs the linalg-map-to-generic pass on them.

from mlir_laksa.ir import (
    Context,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
)
from mlir_laksa.dialects import arith, func, linalg, linalg_ext, tensor
from mlir_laksa.passmanager import PassManager

def build_map_fill(i8) -> func.FuncOp:
    result_type = RankedTensorType.get([1, 32, 32, 8], i8)
    func_op = func.FuncOp("map_fill", ([], [result_type]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        zero = arith.ConstantOp(i8, 0)
        empty = tensor.EmptyOp([1, 32, 32, 8], i8)
        map_op = linalg.MapOp([result_type], [], empty.result)
        map_block = map_op.regions[0].blocks.append(i8)
        with InsertionPoint(map_block):
            linalg.YieldOp([zero.result])
        func.ReturnOp([map_op.result[0]])
    return func_op

def build_map_elementwise(i32) -> func.FuncOp:
    tensor64 = RankedTensorType.get([64], i32)
    func_op = func.FuncOp("map_elementwise", ([tensor64, tensor64], [tensor64]))
    block = func_op.add_entry_block()
    with InsertionPoint(block):
        lhs, rhs = block.arguments
        empty = tensor.EmptyOp([64], i32)
        map_op = linalg.MapOp([tensor64], [lhs, rhs], empty.result)
        map_block = map_op.regions[0].blocks.append(i32, i32, i32)
        with InsertionPoint(map_block):
            in0, in1, _out = map_block.arguments
            v = arith.AddIOp(in0, in1)
            linalg.YieldOp([v.result])
        func.ReturnOp([map_op.result[0]])
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)
            build_map_fill(i8)
            build_map_elementwise(i32)

        print(module)

        pm = PassManager(context=ctx)
        linalg_ext.add_map_to_generic_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
