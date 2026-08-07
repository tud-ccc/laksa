//===- Conversion.cpp - C Interface for conversion passes -----------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Conversion.h"

#include "laksa-mlir/Conversion/ConvertToEmitHLS/ConvertToEmitHLS.h"
#include "laksa-mlir/Conversion/Passes.h"
#include "mlir/CAPI/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir/Conversion/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::laksa;

void mlirRegisterLAKSAConvertToEmitHLSPipelines()
{ registerConvertToEmitHLSPipelines(); }

void mlirConversionAddConvertToEmitHLSPasses(
    MlirOpPassManager passManager,
    int64_t availableBRAM,
    int64_t availableDSP)
{
    addConvertToEmitHLSPasses(
        *unwrap(passManager),
        availableBRAM,
        availableDSP);
}

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Conversion/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
