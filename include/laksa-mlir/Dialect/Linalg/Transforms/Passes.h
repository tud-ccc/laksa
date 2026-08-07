/// Declares the linalg dialect transform passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::linalg {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createLinalgMapToGenericPass();
std::unique_ptr<Pass> createLinalgSoftTransposePass();
std::unique_ptr<Pass> createLinalgScalarizeSplatDensePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"

} // namespace mlir::linalg
