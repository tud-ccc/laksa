//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the laksa conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSES
#define CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAffineToEmitHLS : Pass<"convert-affine-to-emithls"> {
    let summary = "Convert Affine to EmitHLS dialect operations.";

    let constructor = "mlir::createConvertAffineToEmitHLSPass()";

    let dependentDialects = [
        "affine::AffineDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertArithToEmitHLS : Pass<"convert-arith-to-emithls"> {
    let summary = "Convert Arith to EmitHLS dialect arith operations.";

    let constructor = "mlir::createConvertArithToEmitHLSPass()";

    let dependentDialects = [
        "arith::ArithDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertDFGToEmitHLS : Pass<"convert-dfg-to-emithls"> {
    let summary = "Convert DFG to EmitHLS dialect operations.";

    let constructor = "mlir::createConvertDFGToEmitHLSPass()";

    let dependentDialects = [
        "dfg::DFGDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertFuncToDFG : Pass<"convert-func-to-dfg"> {
    let summary = "Convert Func to DFG dialect operations.";

    let constructor = "mlir::createConvertFuncToDFGPass()";

    let dependentDialects = [
        "func::FuncDialect",
        "dfg::DFGDialect"
    ];
}

def ConvertIndexToEmitHLS : Pass<"convert-index-to-emithls"> {
    let summary = "Convert Index to EmitHLS dialect index operations.";

    let constructor = "mlir::createConvertIndexToEmitHLSPass()";

    let dependentDialects = [
        "index::IndexDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertLinalgToLAKSALoops : Pass<"convert-linalg-to-laksa-loops"> {
    let summary = "Convert linalg generic operations to affine loops with LAKSA optimizations.";

    let constructor = "mlir::createConvertLinalgToLAKSALoopsPass()";

    let dependentDialects = [
        "affine::AffineDialect",
        "arith::ArithDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertMemRefPadToLAKSALoops : Pass<"convert-memref-pad-to-laksa-loops"> {
    let summary = "Convert MemRef to EmitHLS dialect operations.";

    let constructor = "mlir::createConvertMemRefPadToLAKSALoopsPass()";

    let dependentDialects = [
        "memref::MemRefDialect",
        "emithls::EmitHLSDialect"
    ];
}

def ConvertReshapedCopyToLoops : Pass<"convert-reshaped-copy-to-loops"> {
    let summary = "Convert a memref copy through a pure reshape into an element loop.";

    let constructor = "mlir::createConvertReshapedCopyToLoopsPass()";

    let dependentDialects = [
        "arith::ArithDialect",
        "memref::MemRefDialect",
        "scf::SCFDialect"
    ];
}

#endif
