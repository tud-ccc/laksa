/// Declares the conversion passes.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)

#pragma once

#include "laksa-mlir/Conversion/AffineToEmitHLS/AffineToEmitHLS.h"
#include "laksa-mlir/Conversion/ArithToEmitHLS/ArithToEmitHLS.h"
#include "laksa-mlir/Conversion/DFGToEmitHLS/DFGToEmitHLS.h"
#include "laksa-mlir/Conversion/FuncToDFG/FuncToDFG.h"
#include "laksa-mlir/Conversion/IndexToEmitHLS/IndexToEmitHLS.h"
#include "laksa-mlir/Conversion/LinalgToLAKSALoops/LinalgToLAKSALoops.h"
#include "laksa-mlir/Conversion/MemRefPadToLAKSALoops/MemRefPadToLAKSALoops.h"
#include "laksa-mlir/Conversion/ReshapedCopyToLoops/ReshapedCopyToLoops.h"

namespace mlir::laksa {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::laksa
