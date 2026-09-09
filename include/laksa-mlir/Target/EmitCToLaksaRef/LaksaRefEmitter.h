/// Convenience include for the EmitCToLaksaRef translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::laksa {

/// Translates the emitc IR of the scalar reference to a C program checking the
/// board's `output<n>.bin` against it.
LogicalResult translateEmitCToLaksaRef(ModuleOp op, raw_ostream &os);

} // namespace mlir::laksa
