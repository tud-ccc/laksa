//===- Func.cpp - C Interface for Func dialect passes -----------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir-c/Dialect/Func.h"

using namespace mlir;
using namespace mlir::func;

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Dialect/Func/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
