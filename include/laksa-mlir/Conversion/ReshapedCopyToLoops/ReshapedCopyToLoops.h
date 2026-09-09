/// Declaration of the ReshapedCopyToLoops conversion.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTRESHAPEDCOPYTOLOOPS
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateReshapedCopyToLoopsPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertReshapedCopyToLoopsPass();

} // namespace mlir
