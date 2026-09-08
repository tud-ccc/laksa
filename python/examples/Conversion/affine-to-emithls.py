# Recreates ops from test/Conversion/affine-to-emithls.mlir using the Python
# bindings and runs the convert-affine-to-emithls pass on them.

import numpy as np

from mlir_laksa.ir import (
    AffineExpr,
    AffineMap,
    Context,
    DenseIntElementsAttr,
    FunctionType,
    InsertionPoint,
    IntegerSet,
    IntegerType,
    Location,
    MemRefType,
    Module,
    RankedTensorType,
    TypeAttr,
)
from mlir_laksa.dialects import affine, arith, dfg, memref
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager


def build_affine(i8, port_type_in, port_type_out) -> dfg.ProcessOp:
    shape = [2, 4]

    d0, d1 = (AffineExpr.get_dim(i) for i in range(2))
    identity_map = AffineMap.get(2, 0, [d0, d1])
    cond_set = IntegerSet.get(
        2,
        0,
        [
            AffineExpr.get_add(d0, AffineExpr.get_constant(-1)),
            AffineExpr.get_add(d1, AffineExpr.get_constant(-2)),
        ],
        [False, False],
    )

    process_op = dfg.ProcessOp(
        "affine", TypeAttr.get(FunctionType.get([port_type_in], [port_type_out]))
    )
    block = process_op.body.blocks.append(port_type_in, port_type_out)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(shape, i8), in_port)

            for_op0 = affine.AffineForOp(0, shape[0])
            with InsertionPoint(for_op0.body):
                idx0 = for_op0.induction_variable
                for_op1 = affine.AffineForOp(0, shape[1])
                with InsertionPoint(for_op1.body):
                    idx1 = for_op1.induction_variable
                    if_op = affine.AffineIfOp(cond_set, cond_operands=[idx0, idx1])
                    with InsertionPoint(if_op.then_block):
                        val = affine.AffineLoadOp(
                            i8, token.result, [idx0, idx1], identity_map
                        )
                        sum_val = arith.AddIOp(val.result, val.result)
                        dfg.PushOp(sum_val.result, out_port, [idx0, idx1])
                        affine.AffineYieldOp([])
                    affine.AffineYieldOp([])
                affine.AffineYieldOp([])
    return process_op


def build_global_const(i8, port_type_out) -> dfg.ProcessOp:
    """Mirrors @global_const: a memref.get_global-backed load should
    materialize as an emithls.variable instead of an unrealized cast."""
    shape = [4]
    d0 = AffineExpr.get_dim(0)
    identity_map = AffineMap.get(1, 0, [d0])

    process_op = dfg.ProcessOp(
        "global_const", TypeAttr.get(FunctionType.get([], [port_type_out]))
    )
    block = process_op.body.blocks.append(port_type_out)
    with InsertionPoint(block):
        (out_port,) = block.arguments

        loop_op = dfg.LoopOp([], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            const_buf = memref.GetGlobalOp(MemRefType.get(shape, i8), "__constant_4xi8")

            for_op = affine.AffineForOp(0, shape[0])
            with InsertionPoint(for_op.body):
                idx0 = for_op.induction_variable
                val = affine.AffineLoadOp(i8, const_buf.result, [idx0], identity_map)
                dfg.PushOp(val.result, out_port, [idx0])
                affine.AffineYieldOp([])
    return process_op


def build_top(port_type_in, port_type_out, global_port_type_out) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        "top",
        TypeAttr.get(
            FunctionType.get([port_type_in], [port_type_out, global_port_type_out])
        ),
    )
    block = region_op.body.blocks.append(
        port_type_in, port_type_out, global_port_type_out
    )
    with InsertionPoint(block):
        in_port, out_port, global_out_port = block.arguments
        dfg.InstantiateOp("affine", [in_port], [out_port])
        dfg.InstantiateOp("global_const", [], [global_out_port])
    return region_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            port_type_in = dfg.OutputType.get(element_type=i8, shape=[2, 4])
            port_type_out = dfg.InputType.get(element_type=i8, shape=[2, 4])
            global_port_type_out = dfg.InputType.get(element_type=i8, shape=[4])

            const_data = DenseIntElementsAttr.get(
                np.array([1, 2, 3, 4], dtype=np.int8),
                type=RankedTensorType.get([4], i8),
            )
            memref.GlobalOp(
                "__constant_4xi8",
                MemRefType.get([4], i8),
                sym_visibility="private",
                constant=True,
                initial_value=const_data,
            )

            build_affine(i8, port_type_in, port_type_out)
            build_global_const(i8, global_port_type_out)
            build_top(port_type_in, port_type_out, global_port_type_out)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_affine_to_emithls_pass(pm)
        pm.add("canonicalize")
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
