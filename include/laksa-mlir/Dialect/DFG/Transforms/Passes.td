//===- Passes.td - Conversion passes ------------------------*- tablegen -*-===//
//
// This is the definitions file for the dfg dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_PASSES
#define DFG_PASSES

include "mlir/Pass/PassBase.td"

def DFGOperatorToProcess : Pass<"dfg-operator-to-process">
{
    let summary = "Convert each operator to process";
    let constructor = "mlir::dfg::createDFGOperatorToProcessPass()";
}

def DFGInlineEmbedRegion : Pass<"dfg-inline-embed-region">
{
    let summary = "Inline embedded region at place.";
    let constructor = "mlir::dfg::createDFGInlineEmbedRegionPass()";
}

def DFGIONormalization : Pass<"dfg-io-normalization">
{
    let summary = "Normalize IO shape and pull/push operations.";
    let constructor = "mlir::dfg::createDFGIONormalizationPass()";
}

def DFGChannelFanoutExpansion : Pass<"dfg-channel-fanout-expansion">
{
    let summary = "";
    let constructor = "mlir::dfg::createDFGChannelFanoutExpansionPass()";
}

def DFGCollapseUnitDims : Pass<"dfg-collapse-unit-dims">
{
    let summary = "Collapse statically size-1 axes out of dfg.process channel port shapes.";
    let constructor = "mlir::dfg::createDFGCollapseUnitDimsPass()";
}

#endif
