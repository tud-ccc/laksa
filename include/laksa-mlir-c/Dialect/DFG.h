//===-- laksa-mlir-c/Dialect/DFG.h - C API for DFG dialect -------*- C-*-===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef LAKSA_MLIR_C_DIALECT_DFG_H
#define LAKSA_MLIR_C_DIALECT_DFG_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DFG, dfg);

//===---------------------------------------------------------------------===//
// InputType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a dfg::InputType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAInputType(MlirType type);

/// Creates a scalar dfg.InputType type (no shape).
MLIR_CAPI_EXPORTED MlirType mlirInputTypeGet(MlirType elementType);

/// Creates a shaped dfg.InputType type with the given rank and dimensions.
MLIR_CAPI_EXPORTED MlirType mlirInputTypeGetWithShape(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType);

/// Returns the rank (number of dimensions) of the dfg.InputType.
MLIR_CAPI_EXPORTED intptr_t mlirInputTypeGetRank(MlirType type);

/// Returns the size of the given dimension of the dfg.InputType.
MLIR_CAPI_EXPORTED int64_t mlirInputTypeGetDimSize(MlirType type, intptr_t dim);

/// Returns the element type of dfg::InputType.
MLIR_CAPI_EXPORTED MlirType mlirInputTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// OutputType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a dfg::OutputType dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAOutputType(MlirType type);

/// Creates a scalar dfg.OutputType type (no shape).
MLIR_CAPI_EXPORTED MlirType mlirOutputTypeGet(MlirType elementType);

/// Creates a shaped dfg.OutputType type with the given rank and dimensions.
MLIR_CAPI_EXPORTED MlirType mlirOutputTypeGetWithShape(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType);

/// Returns the rank (number of dimensions) of the dfg.OutputType.
MLIR_CAPI_EXPORTED intptr_t mlirOutputTypeGetRank(MlirType type);

/// Returns the size of the given dimension of the dfg.OutputType.
MLIR_CAPI_EXPORTED int64_t
mlirOutputTypeGetDimSize(MlirType type, intptr_t dim);

/// Returns the element type of dfg::OutputType.
MLIR_CAPI_EXPORTED MlirType mlirOutputTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// BufferizableOpInterface registration
//===---------------------------------------------------------------------===//

/// Attaches the DFG dialect's BufferizableOpInterface external models (for
/// e.g. dfg.pull_as_tensor/dfg.push_tensor) to the given context, so that
/// upstream bufferization passes such as one-shot-bufferize can handle them.
MLIR_CAPI_EXPORTED void
mlirDFGRegisterBufferizableOpInterfaceExternalModels(MlirContext context);

//===---------------------------------------------------------------------===//
// Translation
//===---------------------------------------------------------------------===//

/// Translates the DFG graph rooted at `op` to Dot and streams the result
/// through `callback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTranslateDFGToDot(
    MlirOperation op,
    MlirStringCallback callback,
    void* userData);

#ifdef __cplusplus
}
#endif

//===---------------------------------------------------------------------===//
// Pass Registration
//===---------------------------------------------------------------------===//

/// Generated per-pass `mlirCreateDFG*`/`mlirRegisterDFG*` functions, plus the
/// group-level `mlirRegisterDFGPasses`. See mlir-c/Dialect/Bufferization.h
/// upstream for the same pattern.
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.capi.h.inc"

#endif
