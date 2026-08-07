# Recreates ops from test/Conversion/linalg-to-laksa-loops.mlir using the
# Python bindings and runs the convert-linalg-to-laksa-loops pass on them.

from mlir_laksa.ir import (
    AffineExpr,
    AffineMap,
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    TypeAttr,
)
from mlir_laksa.dialects import arith, dfg, linalg, memref
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager

def build_kernel_node_0(i8, i32) -> dfg.ProcessOp:
    input_shape = [1, 224, 224, 8]
    output_shape = [1, 222, 222, 8]
    weight_shape = [8, 3, 3, 8]

    d0, d1, d2, d3, d4, d5, d6 = (AffineExpr.get_dim(i) for i in range(7))
    map_fill = AffineMap.get(4, 0, [d0, d1, d2, d3])
    map_input = AffineMap.get(
        7, 0, [d0, AffineExpr.get_add(d1, d4), AffineExpr.get_add(d2, d5), d6]
    )
    map_weight = AffineMap.get(7, 0, [d3, d4, d5, d6])
    map_output = AffineMap.get(7, 0, [d0, d1, d2, d3])

    output_type = dfg.OutputType.get(element_type=i8, shape=input_shape)
    input_type = dfg.InputType.get(element_type=i32, shape=output_shape)
    process_op = dfg.ProcessOp(
        "kernel_node_0", TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c_neg128 = arith.ConstantOp(i32, -128)
        c0 = arith.ConstantOp(i32, 0)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(
                MemRefType.get(input_shape, i8), in_port
            )
            weight = memref.GetGlobalOp(
                MemRefType.get(weight_shape, i8), "__constant_8x3x3x8xi8"
            )
            alloc = memref.AllocOp(
                MemRefType.get(output_shape, i32), [], [], alignment=64
            )

            fill = linalg.GenericOp(
                [],
                [],
                [alloc.result],
                [map_fill],
                ["parallel"] * 4,
            )
            fill_block = fill.regions[0].blocks.append(i32)
            with InsertionPoint(fill_block):
                (out_arg,) = fill_block.arguments
                linalg.YieldOp([c0.result])

            conv = linalg.GenericOp(
                [],
                [token.result, weight.result],
                [alloc.result],
                [map_input, map_weight, map_output],
                ["parallel"] * 4 + ["reduction"] * 3,
            )
            conv_block = conv.regions[0].blocks.append(i8, i8, i32)
            with InsertionPoint(conv_block):
                in_, in_0, out = conv_block.arguments
                ext_in = arith.ExtSIOp(i32, in_)
                zp = arith.SubIOp(ext_in.result, c_neg128.result)
                ext_w = arith.ExtSIOp(i32, in_0)
                mul = arith.MulIOp(zp.result, ext_w.result)
                acc = arith.AddIOp(out, mul.result)
                linalg.YieldOp([acc.result])

            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op

def build_kernel_node_1(i8, i32) -> dfg.ProcessOp:
    shape = [1, 222, 222, 8]
    bias_shape = [8]

    d0, d1, d2, d3 = (AffineExpr.get_dim(i) for i in range(4))
    map_full = AffineMap.get(4, 0, [d0, d1, d2, d3])
    map_bias = AffineMap.get(4, 0, [d3])

    i64 = IntegerType.get_signless(64)
    output_type = dfg.OutputType.get(element_type=i32, shape=shape)
    input_type = dfg.InputType.get(element_type=i8, shape=shape)
    process_op = dfg.ProcessOp(
        "kernel_node_1", TypeAttr.get(FunctionType.get([output_type], [input_type]))
    )
    block = process_op.body.blocks.append(output_type, input_type)
    with InsertionPoint(block):
        in_port, out_port = block.arguments
        c127 = arith.ConstantOp(i32, 127)
        c_neg128 = arith.ConstantOp(i32, -128)
        c40 = arith.ConstantOp(i64, 40)
        c_neg_half = arith.ConstantOp(i64, -1073741824)
        c_half = arith.ConstantOp(i64, 1073741824)
        c0 = arith.ConstantOp(i32, 0)
        c_round = arith.ConstantOp(i64, 549755813888)

        loop_op = dfg.LoopOp([in_port], [out_port])
        loop_block = loop_op.body.blocks.append()
        with InsertionPoint(loop_block):
            token = dfg.PullAsMemRefOp(MemRefType.get(shape, i32), in_port)
            bias = memref.GetGlobalOp(
                MemRefType.get(bias_shape, i32), "__constant_8xi32"
            )
            alloc = memref.AllocOp(MemRefType.get(shape, i8), [], [], alignment=64)

            requant = linalg.GenericOp(
                [],
                [token.result, bias.result],
                [alloc.result],
                [map_full, map_bias, map_full],
                ["parallel"] * 4,
            )
            requant_block = requant.regions[0].blocks.append(i32, i32, i8)
            with InsertionPoint(requant_block):
                in_, in_0, out = requant_block.arguments
                ext_in = arith.ExtSIOp(i64, in_)
                ext_bias = arith.ExtSIOp(i64, in_0)
                prod = arith.MulIOp(ext_in.result, ext_bias.result)
                rounded = arith.AddIOp(prod.result, c_round.result)
                sign = arith.CmpIOp(arith.CmpIPredicate.sge, in_, c0.result)
                half = arith.SelectOp(sign.result, c_half.result, c_neg_half.result)
                sum_ = arith.AddIOp(half.result, rounded.result)
                shifted = arith.ShRSIOp(sum_.result, c40.result)
                trunc32 = arith.TruncIOp(i32, shifted.result)
                zp = arith.AddIOp(trunc32.result, c_neg128.result)
                clamped_lo = arith.MaxSIOp(zp.result, c_neg128.result)
                clamped = arith.MinSIOp(clamped_lo.result, c127.result)
                narrow = arith.TruncIOp(i8, clamped.result)
                linalg.YieldOp([narrow.result])

            dfg.PushMemRefOp(alloc.result, out_port)
    return process_op

def build_combined_region(
    conv_output_type,
    conv_input_type,
    conv_mid_type,
    i32,
) -> dfg.RegionOp:
    region_op = dfg.RegionOp(
        "combined",
        TypeAttr.get(
            FunctionType.get([conv_output_type], [conv_input_type])
        ),
    )
    block = region_op.body.blocks.append(conv_output_type, conv_input_type)
    with InsertionPoint(block):
        conv_in, conv_out = block.arguments
        conv_mid_in_type = dfg.InputType.get(element_type=i32, shape=[1, 222, 222, 8])
        conv_mid_out_type = dfg.OutputType.get(
            element_type=i32, shape=[1, 222, 222, 8]
        )
        conv_mid_in, conv_mid_out = dfg.ChannelOp(
            conv_mid_in_type, conv_mid_out_type, i32
        ).results

        dfg.InstantiateOp("kernel_node_0", [conv_in], [conv_mid_in])
        dfg.InstantiateOp("kernel_node_1", [conv_mid_out], [conv_out])
    return region_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i8 = IntegerType.get_signless(8)
            i32 = IntegerType.get_signless(32)

            memref.GlobalOp(
                "__constant_8xi32", MemRefType.get([8], i32), sym_visibility="private", constant=True
            )
            memref.GlobalOp(
                "__constant_8x3x3x8xi8",
                MemRefType.get([8, 3, 3, 8], i8),
                sym_visibility="private",
                constant=True,
            )

            build_kernel_node_0(i8, i32)
            build_kernel_node_1(i8, i32)

            build_combined_region(
                dfg.OutputType.get(element_type=i8, shape=[1, 224, 224, 8]),
                dfg.InputType.get(element_type=i8, shape=[1, 222, 222, 8]),
                None,
                i32,
            )

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_linalg_to_laksa_loops_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
