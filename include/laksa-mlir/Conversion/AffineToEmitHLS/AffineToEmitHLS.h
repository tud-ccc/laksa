/// Declaration of the AffineToEmitHLS conversions.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTAFFINETOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateAffineToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertAffineToEmitHLSPass();

} // namespace mlir
