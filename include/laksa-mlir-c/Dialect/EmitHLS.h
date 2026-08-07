//===-- laksa-mlir-c/Dialect/EmitHLS.h - C API for EmitHLS dialect ---*- C
//-*-===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef LAKSA_MLIR_C_DIALECT_EMITHLS_H
#define LAKSA_MLIR_C_DIALECT_EMITHLS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(EmitHLS, emithls);

//===---------------------------------------------------------------------===//
// StreamType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is an emithls::StreamType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitHLSStreamType(MlirType type);

/// Creates an emithls.stream type with the given element type.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSStreamTypeGet(MlirType elementType);

/// Returns the element type of the emithls::StreamType.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSStreamTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is an emithls::PointerType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitHLSPointerType(MlirType type);

/// Creates an emithls.ptr type with the given element type.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSPointerTypeGet(MlirType elementType);

/// Returns the element type of the emithls::PointerType.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSPointerTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is an emithls::ArrayType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitHLSArrayType(MlirType type);

/// Creates an emithls.array type with the given shape and element type.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSArrayTypeGet(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType);

/// Returns the rank of the emithls::ArrayType.
MLIR_CAPI_EXPORTED intptr_t mlirEmitHLSArrayTypeGetRank(MlirType type);

/// Returns the size of the given dimension of the emithls::ArrayType.
MLIR_CAPI_EXPORTED int64_t
mlirEmitHLSArrayTypeGetDimSize(MlirType type, intptr_t dim);

/// Returns the element type of the emithls::ArrayType.
MLIR_CAPI_EXPORTED MlirType mlirEmitHLSArrayTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// Translation
//===---------------------------------------------------------------------===//

/// Translates the EmitHLS IR rooted at `op` to HLS C++ and streams the
/// result through `callback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTranslateEmitHLSToCpp(
    MlirOperation op,
    MlirStringCallback callback,
    void* userData);

#ifdef __cplusplus
}
#endif

//===---------------------------------------------------------------------===//
// Pass Registration
//===---------------------------------------------------------------------===//

#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an "emithls-pragma-dse" pass with explicit BRAM/DSP budgets
MLIR_CAPI_EXPORTED MlirPass mlirCreateEmitHLSEmitHLSPragmaDSEWithOptions(
    int64_t availableBRAM,
    int64_t availableDSP);

#ifdef __cplusplus
}
#endif

#endif
