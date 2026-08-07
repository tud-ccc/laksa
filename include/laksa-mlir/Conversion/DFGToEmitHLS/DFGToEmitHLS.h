/// Declaration of the DFGToEmitHLS conversions.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTDFGTOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateDFGToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertDFGToEmitHLSPass();

} // namespace mlir
