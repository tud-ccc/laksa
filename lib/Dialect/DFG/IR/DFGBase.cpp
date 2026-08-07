/// Implements the DFG dialect base.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFGBase.h"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#include "laksa-mlir/Dialect/DFG/IR/DFGBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DFGDialect
//===----------------------------------------------------------------------===//

void DFGDialect::initialize()
{
    registerOps();
    registerTypes();
    registerAttributes();
}
