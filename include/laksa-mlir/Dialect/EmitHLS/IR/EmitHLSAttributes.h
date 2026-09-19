/// Declaration of the EmitHLS dialect attributes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "mlir/IR/Attributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

//===- Generated includes -------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::emithls {

/// Number of cycles predicted for a function by the EmitHLS model.
inline constexpr llvm::StringLiteral kModelCyclesAttrName =
    "emithls.model_cycles";

} // namespace mlir::emithls
