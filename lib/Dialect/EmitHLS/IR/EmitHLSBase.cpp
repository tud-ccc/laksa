/// Implements the EmitHLS dialect base.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"

using namespace mlir;
using namespace mlir::emithls;

//===- Generated implementation -------------------------------------------===//

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EmitHLSDialect
//===----------------------------------------------------------------------===//

void EmitHLSDialect::initialize()
{
    registerOps();
    registerTypes();
    registerAttributes();
}
