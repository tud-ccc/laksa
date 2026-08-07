/// Implements the DFG dialect attributes.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFGAttributes.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "laksa-mlir/Dialect/DFG/IR/DFGAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DFGDialect
//===----------------------------------------------------------------------===//

void DFGDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "laksa-mlir/Dialect/DFG/IR/DFGAttributes.cpp.inc"
        >();
}
