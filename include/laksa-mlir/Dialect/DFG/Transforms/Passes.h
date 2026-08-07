/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::dfg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Defines constructors of transformation passes
std::unique_ptr<Pass> createDFGOperatorToProcessPass();
std::unique_ptr<Pass> createDFGInlineEmbedRegionPass();
std::unique_ptr<Pass> createDFGIONormalizationPass();
std::unique_ptr<Pass> createDFGChannelFanoutExpansionPass();
std::unique_ptr<Pass> createDFGCollapseUnitDimsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"

} // namespace mlir::dfg
