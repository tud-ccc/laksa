/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::emithls {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

/// Defines constructors of transformation passes
std::unique_ptr<Pass> createEmitHLSAddIOFunctionsPass();
std::unique_ptr<Pass> createEmitHLSFoldExpressionPass();
std::unique_ptr<Pass> createEmitHLSFuseOperatorPass();
std::unique_ptr<Pass> createEmitHLSInsertIncludesPass();
std::unique_ptr<Pass> createEmitHLSLoopFusionPass();
std::unique_ptr<Pass> createEmitHLSMergeCastChainPass();
std::unique_ptr<Pass> createEmitHLSPragmaDSEPass();
std::unique_ptr<Pass>
createEmitHLSPragmaDSEPass(int64_t availableBRAM, int64_t availableDSP);
std::unique_ptr<Pass> createEmitHLSPragmaInsertionPass();
std::unique_ptr<Pass> createEmitHLSResolveHelpersPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"

} // namespace mlir::emithls
