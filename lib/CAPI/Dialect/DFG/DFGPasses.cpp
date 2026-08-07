//===- DFGPasses.cpp - C API for DFG dialect passes --------------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::dfg;

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Dialect/DFG/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
