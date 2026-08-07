//===-- laksa-mlir-c/Dialect/Func.h - C API for Func passes ----------*- C-*-===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#ifndef LAKSA_MLIR_C_DIALECT_FUNC_H
#define LAKSA_MLIR_C_DIALECT_FUNC_H

#include "mlir-c/Support.h"

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generated per-pass `mlirCreateFuncExt*`/`mlirRegisterFuncExt*` functions,
/// plus the group-level `mlirRegisterFuncExtPasses`.
#include "laksa-mlir/Dialect/Func/Transforms/Passes.capi.h.inc"

#endif
