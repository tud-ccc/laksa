//===-- laksa-mlir-c/Dialect/Linalg.h - C API for Linalg passes ------*-
//C-*-===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef LAKSA_MLIR_C_DIALECT_LINALG_H
#define LAKSA_MLIR_C_DIALECT_LINALG_H

#include "mlir-c/Support.h"

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generated per-pass `mlirCreateLinalgExt*`/`mlirRegisterLinalgExt*`
/// functions, plus the group-level `mlirRegisterLinalgExtPasses`.
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.capi.h.inc"

#endif
