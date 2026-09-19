/// Convenience include for the EmitCToCPUProfileRunScript translation.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::laksa {

/// Translates EmitC CPU-profile IR to a script building and running the
/// generated benchmark on the target CPU.
LogicalResult
translateEmitCToCPUProfileRunScript(ModuleOp op, raw_ostream &os);

} // namespace mlir::laksa
