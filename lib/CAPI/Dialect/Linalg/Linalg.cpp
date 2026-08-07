//===- Linalg.cpp - C Interface for extended Linalg passes ------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir-c/Dialect/Linalg.h"

using namespace mlir;
using namespace mlir::linalg;

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
