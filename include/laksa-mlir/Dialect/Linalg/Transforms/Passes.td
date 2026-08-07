//===- Passes.td - Linalg transform passes ------------------*- tablegen -*-===//
//
// This is the definitions file for the linalg dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_EXT_PASSES
#define LINALG_EXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgMapToGeneric : Pass<"linalg-map-to-generic">
{
    let summary = "Convert map operation to equivalent generic.";
    let constructor = "mlir::linalg::createLinalgMapToGenericPass()";
}

def LinalgSoftTranspose : Pass<"linalg-soft-transpose">
{
    let summary = "Transpose constant tensors during compilation.";
    let constructor = "mlir::linalg::createLinalgSoftTransposePass()";
}

def LinalgScalarizeSplatDense : Pass<"linalg-scalarize-splat-dense">
{
    let summary = "Use scalar value instead of splat tensors in generic operation.";
    let constructor = "mlir::linalg::createLinalgScalarizeSplatDensePass()";
}

#endif // LINALG_EXT_PASSES
