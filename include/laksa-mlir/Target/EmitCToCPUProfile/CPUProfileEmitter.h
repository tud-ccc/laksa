/// EmitC-to-CPU-profile benchmark translation.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::laksa {

LogicalResult translateEmitCToCPUProfile(ModuleOp op, llvm::raw_ostream &os);
LogicalResult
translateEmitCToCPUProfileNodes(ModuleOp op, llvm::raw_ostream &os);

} // namespace mlir::laksa
