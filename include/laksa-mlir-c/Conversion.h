//===--- Conversion.h - C API for conversion passes ---------------*- C -*-===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef LAKSA_MLIR_C_CONVERSION_H
#define LAKSA_MLIR_C_CONVERSION_H

#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the laksa-convert-to-emitc pass pipeline.
MLIR_CAPI_EXPORTED void mlirRegisterLAKSAConvertToEmitCPipelines();
MLIR_CAPI_EXPORTED void mlirConversionAddConvertToEmitCPasses(
    MlirOpPassManager passManager,
    uint32_t maxAllocSizeInBytes);

/// Registers the convert-to-emithls pass pipeline.
MLIR_CAPI_EXPORTED void mlirRegisterLAKSAConvertToEmitHLSPipelines();
MLIR_CAPI_EXPORTED void mlirConversionAddConvertToEmitHLSPasses(
    MlirOpPassManager passManager,
    int64_t availableBRAM,
    int64_t availableDSP);

#ifdef __cplusplus
}
#endif

/// Generated per-pass `mlirCreateConversion*`/`mlirRegisterConversion*`
/// functions, plus the group-level `mlirRegisterConversionPasses`.
#include "laksa-mlir/Conversion/Passes.capi.h.inc"

#endif
