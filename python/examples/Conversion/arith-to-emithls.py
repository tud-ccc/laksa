# Recreates ops from test/Conversion/arith-to-emithls.mlir using the Python
# bindings and runs the convert-arith-to-emithls pass on them.

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
import mlir_laksa.conversion as conversion
from mlir_laksa.passmanager import PassManager

def build_arith(i16, i32, i64) -> emithls.FuncOp:
    func_op = emithls.FuncOp("arith", TypeAttr.get(FunctionType.get([], [])))
    block = func_op.body.blocks.append()
    with InsertionPoint(block):
        var = emithls.VariableOp(i32).variable
        idx = emithls.VariableOp(IndexType.get()).variable
        zero = arith.ConstantOp(i32, 2)
        arith.AddIOp(var, zero.result)
        arith.SubIOp(var, zero.result)
        arith.MulIOp(var, zero.result)
        arith.RemUIOp(var, zero.result)
        arith.RemSIOp(var, zero.result)
        arith.ShLIOp(var, zero.result)
        arith.ShRUIOp(var, zero.result)
        arith.ShRSIOp(var, zero.result)
        arith.MaxUIOp(var, zero.result)
        arith.MaxSIOp(var, zero.result)
        arith.MinUIOp(var, zero.result)
        arith.MinSIOp(var, zero.result)
        arith.AndIOp(var, zero.result)
        arith.OrIOp(var, zero.result)
        arith.IndexCastOp(i32, idx)
        arith.IndexCastUIOp(i32, idx)
        arith.ExtUIOp(i64, var)
        arith.ExtSIOp(i64, var)
        arith.TruncIOp(i16, var)
        arith.CmpIOp(0, var, zero.result)  # eq
        arith.CmpIOp(1, var, zero.result)  # ne
        slt = arith.CmpIOp(2, var, zero.result)  # slt
        arith.CmpIOp(3, var, zero.result)  # sle
        arith.CmpIOp(4, var, zero.result)  # sgt
        arith.CmpIOp(5, var, zero.result)  # sge
        arith.CmpIOp(6, var, zero.result)  # ult
        arith.CmpIOp(7, var, zero.result)  # ule
        arith.CmpIOp(8, var, zero.result)  # ugt
        arith.CmpIOp(9, var, zero.result)  # uge
        arith.SelectOp(slt.result, var, zero.result)
    return func_op

def main() -> None:
    ctx = Context()

    with ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i16 = IntegerType.get_signless(16)
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            build_arith(i16, i32, i64)

        print(module)

        pm = PassManager(context=ctx)
        conversion.add_arith_to_emithls_pass(pm)
        pm.enable_ir_printing()  # same as --mlir-print-ir-after-all
        pm.run(module.operation)

        print(module)

if __name__ == "__main__":
    main()
