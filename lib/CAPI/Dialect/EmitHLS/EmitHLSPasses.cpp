//===- EmitHLSPasses.cpp - C API for EmitHLS dialect passes ------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::emithls;

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.capi.cpp.inc"

MlirPass mlirCreateEmitHLSEmitHLSPragmaDSEWithOptions(
    int64_t availableBRAM,
    int64_t availableDSP)
{
    return wrap(
        createEmitHLSPragmaDSEPass(availableBRAM, availableDSP).release());
}

#ifdef __cplusplus
}
#endif
