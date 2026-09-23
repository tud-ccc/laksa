/// Implements the EmitHLS dialect types.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <mlir/IR/Diagnostics.h>

using namespace mlir;
using namespace mlir::emithls;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

/// @brief Verifies the invariants of an StreamType.
///
/// Checks that @p elementType is an integer type.
LogicalResult StreamType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type elementType)
{
    if (!llvm::isa<IntegerType>(elementType))
        return emitError() << "Only integers are currently supported in stream";
    return success();
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

/// @brief Verifies the invariants of an PointerType.
///
/// Checks that @p elementType is an integer.
LogicalResult PointerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type elementType)
{
    if (!llvm::isa<IntegerType>(elementType))
        return emitError()
               << "Only integers are currently supported with pointer";
    return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// @brief Creates an ArrayType with explicit shape and element type
///
/// The @p shape defines the number of elements along each dimension. All
/// dimensions must be static (non-negative).
/// The @p elementType must be a scalar type.
ArrayType ArrayType::get(ArrayRef<int64_t> shape, Type elementType)
{ return Base::get(elementType.getContext(), shape, elementType); }

/// @brief Parses an ArrayType from its assembly format `<[shape x] type>`
///
/// Expects a `<` delimiter, an optional dimension list (e.g. `2x3x`), a scalar
/// element type, and a closing `>`. Returns an empty Type on any parse failure.
Type ArrayType::parse(AsmParser &parser)
{
    if (parser.parseLess()) return {};

    SmallVector<int64_t> shape;
    Type elementType;

    if (parser.parseDimensionList(shape, /*allowDynamic=*/false)) return {};
    if (parser.parseType(elementType)) return {};
    if (parser.parseGreater()) return {};

    return ArrayType::get(shape, elementType);
}

/// @brief Prints an ArrayType in its assembly format `<[shape x] type>`.
///
/// Emits `<`, the dimension list, the element type, and `>`.
void ArrayType::print(AsmPrinter &p) const
{
    p << "<";
    p.printDimensionList(getShape());
    p << "x";
    p << getElementType();
    p << ">";
}

/// @brief Verifies the invariants of an ArrayType.
///
/// Checks that @p elementType is a scalar (not a ShapedType) and that @p shape
/// is non-empty and every dimension in it is non-negative (i.e. static).
LogicalResult ArrayType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape,
    Type elementType)
{
    if (isa<ShapedType>(elementType))
        return emitError()
               << "element type of ArrayType must be a scalar, not a shaped "
                  "type like '"
               << elementType << "'";
    if (shape.empty())
        return emitError() << "shape cannot be empty for an array";
    for (int64_t dim : shape)
        if (dim < 0)
            return emitError() << "dimensions of InputType must be static";
    return success();
}

//===----------------------------------------------------------------------===//
// EmitHLSDialect
//===----------------------------------------------------------------------===//

void EmitHLSDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.cpp.inc"
        >();
}
