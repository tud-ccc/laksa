/// Implements the EmitHLS dialect attributes.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.h"

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::emithls;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EmitHLSDialect
//===----------------------------------------------------------------------===//

void EmitHLSDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.cpp.inc"
        >();
}
