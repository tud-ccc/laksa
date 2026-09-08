# Recreates ops from test/Dialect/DFG/Transforms/io-normalization.mlir using
# the Python bindings and runs the dfg-io-normalization pass on them.

import numpy as np

from mlir_laksa.ir import (
    AffineExpr,
    AffineMap,
    ArrayAttr,
    Context,
    DenseIntElementsAttr,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerSet,
    IntegerType,
    Location,
    MemRefType,
    Module,
    RankedTensorType,
    TypeAttr,
    UnitAttr,
)
from mlir_laksa.dialects import affine, arith, dfg, emithls, memref
from mlir_laksa.passmanager import PassManager


def build_pad(i8) -> dfg.ProcessOp:
    """pad style: materializes a padded alloc via subview+copy, then pushes
    it whole. Untouched by the pass apart from its port shapes."""
    real_shape = [1, 30, 30, 8]
    padded_shape = [1, 32, 32, 8]

    in_type = dfg.OutputType.get(element_type=i8, shape=real_shape)
    out_type = dfg.InputType.get(element_type=i8, shape=padded_shape)
    process_op = dfg.ProcessOp(
        "pad", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(real_shape, i8), in_port)
            alloc = memref.AllocOp(
                MemRefType.get(padded_shape, i8), [], [], alignment=64
            )
            alloc.operation.attributes["fill_with"] = IntegerAttr.get(i8, -128)

            subview = memref.subview(
                alloc.result, [0, 1, 1, 0], [1, 30, 30, 8], [1, 1, 1, 1]
            )
            memref.CopyOp(token.result, subview)
            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_conv(i8, i32) -> dfg.ProcessOp:
    """accumulate style: a sliding-window reduction into an `accu_at` alloc,
    guarded by an affine.if boundary check. The pass shrinks the alloc down
    to just its accu_at axes but keeps it (and the linebuf/window/accumulate
    ops) around."""
    padded_shape = [1, 32, 32, 8]
    out_shape = [1, 30, 30, 8]

    in_type = dfg.OutputType.get(element_type=i8, shape=padded_shape)
    out_type = dfg.InputType.get(element_type=i32, shape=out_shape)
    process_op = dfg.ProcessOp(
        "conv", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c_mul = arith.ConstantOp(i32, 3)
        c_zp = arith.ConstantOp(i32, -128)

        d0, d1 = AffineExpr.get_dim(0), AffineExpr.get_dim(1)
        cond = IntegerSet.get(
            2,
            0,
            [
                AffineExpr.get_add(d0, AffineExpr.get_constant(-2)),
                AffineExpr.get_add(d1, AffineExpr.get_constant(-2)),
            ],
            [False, False],
        )
        identity4 = AffineMap.get_identity(4)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(padded_shape, i8), in_port)
            alloc = memref.AllocOp(MemRefType.get(out_shape, i32), [], [], alignment=64)
            alloc.operation.attributes["accu_at"] = ArrayAttr.get(
                [IntegerAttr.get(i32, 0), IntegerAttr.get(i32, 3)]
            )
            alloc.operation.attributes["fill_with"] = IntegerAttr.get(i32, 0)

            h_loop = affine.AffineForOp(0, 32)
            with InsertionPoint(h_loop.body):
                w_loop = affine.AffineForOp(0, 32)
                with InsertionPoint(w_loop.body):
                    h, w = h_loop.induction_variable, w_loop.induction_variable
                    linebuf = emithls.HelperLineBufferOp(
                        MemRefType.get([1, 8, 2, 32], i8),
                        token.result,
                        [1, 8],
                        2,
                        index=w,
                    )
                    window = emithls.HelperWindowOp(
                        MemRefType.get([1, 8, 3, 3], i8), linebuf.result, [h, w]
                    )
                    if_op = affine.AffineIfOp(cond, cond_operands=[h, w])
                    with InsertionPoint(if_op.then_block):
                        n_loop = affine.AffineForOp(0, 1)
                        with InsertionPoint(n_loop.body):
                            c_loop = affine.AffineForOp(0, 8)
                            with InsertionPoint(c_loop.body):
                                kh_loop = affine.AffineForOp(0, 3)
                                with InsertionPoint(kh_loop.body):
                                    kw_loop = affine.AffineForOp(0, 3)
                                    with InsertionPoint(kw_loop.body):
                                        ic_loop = affine.AffineForOp(0, 8)
                                        with InsertionPoint(ic_loop.body):
                                            n = n_loop.induction_variable
                                            c = c_loop.induction_variable
                                            kh = kh_loop.induction_variable
                                            kw = kw_loop.induction_variable
                                            ic = ic_loop.induction_variable
                                            v = affine.AffineLoadOp(
                                                i8,
                                                window.result,
                                                [n, ic, kh, kw],
                                                identity4,
                                            )
                                            ext = arith.ExtSIOp(i32, v.result)
                                            sub = arith.SubIOp(ext.result, c_zp.result)
                                            mul = arith.MulIOp(sub.result, c_mul.result)
                                            emithls.HelperAccumulateOp(
                                                alloc.result,
                                                emithls.FusedOperator.add,
                                                mul.result,
                                                [n, c],
                                            )
                                            affine.AffineYieldOp([])
                                        affine.AffineYieldOp([])
                                    affine.AffineYieldOp([])
                                affine.AffineYieldOp([])
                            affine.AffineYieldOp([])
                        affine.AffineYieldOp([])
                    affine.AffineYieldOp([])
                affine.AffineYieldOp([])
            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_relu(i32, i8) -> dfg.ProcessOp:
    """parallel style: elementwise affine.for/load/store into a plain
    `parallel` alloc. The pass eliminates the alloc entirely, replacing it
    with per-element dfg.pull/dfg.push."""
    shape = [1, 30, 30, 8]
    bias_shape = [8]

    in_type = dfg.OutputType.get(element_type=i32, shape=shape)
    out_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "relu", TypeAttr.get(FunctionType.get([in_type], [out_type]))
    )
    block = process_op.body.blocks.append(in_type, out_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        i64 = IntegerType.get_signless(64)
        c127 = arith.ConstantOp(i32, 127)
        c_neg128 = arith.ConstantOp(i32, -128)
        c40 = arith.ConstantOp(i64, 40)
        c_neg_half = arith.ConstantOp(i64, -1073741824)
        c_half = arith.ConstantOp(i64, 1073741824)
        c0 = arith.ConstantOp(i32, 0)
        c_round = arith.ConstantOp(i64, 549755813888)

        identity4 = AffineMap.get_identity(4)
        identity1 = AffineMap.get_identity(1)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(shape, i32), in_port)
            bias = memref.GetGlobalOp(
                MemRefType.get(bias_shape, i32), "__constant_8xi32"
            )
            alloc = memref.AllocOp(MemRefType.get(shape, i8), [], [], alignment=64)
            alloc.operation.attributes["parallel"] = UnitAttr.get()

            n_loop = affine.AffineForOp(0, 1)
            with InsertionPoint(n_loop.body):
                h_loop = affine.AffineForOp(0, 30)
                with InsertionPoint(h_loop.body):
                    w_loop = affine.AffineForOp(0, 30)
                    with InsertionPoint(w_loop.body):
                        c_loop = affine.AffineForOp(0, 8)
                        with InsertionPoint(c_loop.body):
                            n = n_loop.induction_variable
                            h = h_loop.induction_variable
                            w = w_loop.induction_variable
                            c = c_loop.induction_variable
                            v = affine.AffineLoadOp(
                                i32, token.result, [n, h, w, c], identity4
                            )
                            b = affine.AffineLoadOp(i32, bias.result, [c], identity1)
                            ext_v = arith.ExtSIOp(i64, v.result)
                            ext_b = arith.ExtSIOp(i64, b.result)
                            prod = arith.MulIOp(ext_v.result, ext_b.result)
                            rounded = arith.AddIOp(prod.result, c_round.result)
                            sign = arith.CmpIOp(
                                arith.CmpIPredicate.sge, v.result, c0.result
                            )
                            half = arith.SelectOp(
                                sign.result, c_half.result, c_neg_half.result
                            )
                            summed = arith.AddIOp(half.result, rounded.result)
                            shifted = arith.ShRSIOp(summed.result, c40.result)
                            trunc32 = arith.TruncIOp(i32, shifted.result)
                            zp = arith.AddIOp(trunc32.result, c_neg128.result)
                            clamped_lo = arith.MaxSIOp(zp.result, c_neg128.result)
                            clamped = arith.MinSIOp(clamped_lo.result, c127.result)
                            narrow = arith.TruncIOp(i8, clamped.result)
                            affine.AffineStoreOp(
                                narrow.result, alloc.result, [n, h, w, c], identity4
                            )
            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op


def build_kernel_region(i8, i32) -> dfg.RegionOp:
    shape = [1, 30, 30, 8]
    output_type = dfg.OutputType.get(element_type=i8, shape=shape)
    input_type = dfg.InputType.get(element_type=i8, shape=shape)
    region_op = dfg.RegionOp(
        "kernel", TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = region_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in0, out0 = block.arguments
        padded_in, padded_out = dfg.ChannelOp(
            dfg.InputType.get(element_type=i8, shape=[1, 32, 32, 8]),
            dfg.OutputType.get(element_type=i8, shape=[1, 32, 32, 8]),
            i8,
        ).results
        conv_in, conv_out = dfg.ChannelOp(
            dfg.InputType.get(element_type=i32, shape=shape),
            dfg.OutputType.get(element_type=i32, shape=shape),
            i32,
        ).results

        dfg.InstantiateOp("pad", [in0], [padded_in])
        dfg.InstantiateOp("conv", [padded_out], [conv_in])
        dfg.InstantiateOp("relu", [conv_out], [out0])
    return region_op


def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)

            bias_data = DenseIntElementsAttr.get(
                np.array(
                    [
                        1241604906,
                        1329922038,
                        1341576575,
                        1322904542,
                        1326948123,
                        1339807398,
                        1325877483,
                        1340943676,
                    ],
                    dtype=np.int32,
                ),
                type=RankedTensorType.get([8], i32),
            )
            memref.GlobalOp(
                "__constant_8xi32",
                MemRefType.get([8], i32),
                sym_visibility="private",
                constant=True,
                initial_value=bias_data,
                alignment=64,
            )

            build_pad(i8)
            build_conv(i8, i32)
            build_relu(i32, i8)
            build_kernel_region(i8, i32)

        print(module)

        pm = PassManager(context=ctx)
        dfg.add_io_normalization_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)


if __name__ == "__main__":
    main()
