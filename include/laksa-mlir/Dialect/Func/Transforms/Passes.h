/// Declares the func dialect transform passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::func {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createFuncOutlineComputationLeafPass();
std::unique_ptr<Pass> createFuncRemoveOutlinedFunctionWrappersPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h.inc"

} // namespace mlir::func
