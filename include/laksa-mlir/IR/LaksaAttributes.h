/// Common LAKSA attribute names.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::laksa {

/// Marks the root operation of the compiled application.
inline constexpr llvm::StringLiteral kRootAttrName = "laksa.root";

} // namespace mlir::laksa
