/// Declaration of the ArithToEmitHLS conversions.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTARITHTOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateArithToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertArithToEmitHLSPass();

} // namespace mlir
