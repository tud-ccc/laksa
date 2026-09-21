//===- Passes.td - Func transform passes ------------------*- tablegen -*-===//
//
// This is the definitions file for the func dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef FUNC_EXT_PASSES
#define FUNC_EXT_PASSES

include "mlir/Pass/PassBase.td"

def FuncOutlineComputationLeaf : Pass<"func-outline-computation-leaf", "func::FuncOp">
{
    let summary = "Outline computation ops in graph functions into separate leafs.";
    let constructor = "mlir::func::createFuncOutlineComputationLeafPass()";
}

def FuncRemoveOutlinedFunctionWrappers
    : Pass<"func-remove-outlined-function-wrappers", "ModuleOp"> {
    let summary = "Remove root wrappers left by computation-node outlining";
    let constructor =
        "mlir::func::createFuncRemoveOutlinedFunctionWrappersPass()";

    let dependentDialects = [
        "func::FuncDialect"
    ];
}

#endif // FUNC_EXT_PASSES
