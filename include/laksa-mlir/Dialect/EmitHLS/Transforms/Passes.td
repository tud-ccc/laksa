//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the emithls dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_PASSES
#define EMITHLS_PASSES

include "mlir/Pass/PassBase.td"

def EmitHLSAddIOFunctions : Pass<"emithls-add-io-functions">
{
    let summary = "Add IO functions for top function.";
    let constructor = "mlir::emithls::createEmitHLSAddIOFunctionsPass()";
}

def EmitHLSFoldExpression : Pass<"emithls-fold-expression">
{
    let summary = "Fold expression's content, such as `-a > 2`, to `a < 2`";
    let constructor = "mlir::emithls::createEmitHLSFoldExpressionPass()";
}

def EmitHLSFuseOperator : Pass<"emithls-fuse-operator">
{
    let summary = "Fuse operators i.e. `+=`, `-=`, and `*=`.";
    let constructor = "mlir::emithls::createEmitHLSFuseOperatorPass()";
}

def EmitHLSInsertIncludes : Pass<"emithls-insert-includes">
{
    let summary = "Insert include headers based on the content.";
    let constructor = "mlir::emithls::createEmitHLSInsertIncludesPass()";
}

def EmitHLSLoopFusion : Pass<"emithls-loop-fusion">
{
    let summary = "Merge two emithls.for loops if possible and resolve related ops.";
    let constructor = "mlir::emithls::createEmitHLSLoopFusionPass()";
}

def EmitHLSMergeCastChain : Pass<"emithls-merge-cast-chain">
{
    let summary = "Merge a chain of pure cast operations into one.";
    let constructor = "mlir::emithls::createEmitHLSMergeCastChainPass()";
}

def EmitHLSPragmaDSE : Pass<"emithls-pragma-dse">
{
    let summary = "Perform a design space exploration on where to put pragmas.";
    let constructor = "mlir::emithls::createEmitHLSPragmaDSEPass()";
    let options = [
        Option<"availableBRAM", "available-bram", "int64_t", /*default=*/"288",
            "Total Number of BRAM on the hardware">,
        Option<"availableDSP", "available-dsp", "int64_t", /*default=*/"1248",
            "Total Number of DSP on the hardware">
    ];
}

def EmitHLSPragmaInsertion : Pass<"emithls-pragma-insertion">
{
    let summary = "Rewrite emithls operations based on DSE attributes.";
    let constructor = "mlir::emithls::createEmitHLSPragmaInsertionPass()";
}

def EmitHLSResolveHelpers : Pass<"emithls-resolve-helpers">
{
    let summary = "Resolve helper operations into normal emithls operations.";
    let constructor = "mlir::emithls::createEmitHLSResolveHelpersPass()";
}

#endif
