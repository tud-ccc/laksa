/// Implements the DFG dialect types.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InputType
//===----------------------------------------------------------------------===//

/// @brief Creates an InputType for a scalar element type with no shape
/// dimensions.
///
/// This is the simplest form of InputType where the channel carries a single
/// scalar value of the given @p elementType. Equivalent to calling
/// InputType::get({}, elementType).
InputType InputType::get(Type elementType)
{
    return Base::get(
        elementType.getContext(),
        ArrayRef<int64_t>{},
        elementType);
}

/// @brief Creates an InputType with an explicit shape and element type.
///
/// The @p shape defines the number of elements along each dimension. All
/// dimensions must be static (non-negative). Pass an empty shape for a scalar
/// channel.
InputType InputType::get(ArrayRef<int64_t> shape, Type elementType)
{ return Base::get(elementType.getContext(), shape, elementType); }

/// @brief Creates an InputType from an existing ShapedType.
///
/// Convenience builder that decomposes @p shapedType into its shape and scalar
/// element type, then constructs the corresponding InputType. Useful when
/// adapting memref or tensor types to DFG channel types.
InputType InputType::get(ShapedType shapedType)
{
    return Base::get(
        shapedType.getContext(),
        shapedType.getShape(),
        shapedType.getElementType());
}

/// @brief Parses an InputType from its assembly format `<[shape x] type>`.
///
/// Expects a `<` delimiter, an optional dimension list (e.g. `2x3x`), a scalar
/// element type, and a closing `>`. Returns an empty Type on any parse failure.
Type InputType::parse(AsmParser &parser)
{
    if (parser.parseLess()) return {};

    SmallVector<int64_t> shape;
    Type elementType;

    if (parser.parseDimensionList(shape, /*allowDynamic=*/false)) return {};
    if (parser.parseType(elementType)) return {};
    if (parser.parseGreater()) return {};

    return InputType::get(shape, elementType);
}

/// @brief Prints an InputType in its assembly format `<[shape x] type>`.
///
/// Emits `<`, the dimension list (omitting the trailing `x` for scalar types),
/// the element type, and `>`.
void InputType::print(AsmPrinter &p) const
{
    p << "<";
    p.printDimensionList(getShape());
    if (!getShape().empty()) p << "x";
    p.printType(getElementType());
    p << ">";
}

/// @brief Verifies the invariants of an InputType.
///
/// Checks that @p elementType is a scalar (not a ShapedType) and that every
/// dimension in @p shape is non-negative (i.e. static).
LogicalResult InputType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape,
    Type elementType)
{
    if (isa<ShapedType>(elementType))
        return emitError()
               << "element type of InputType must be a scalar, not a shaped "
                  "type like '"
               << elementType << "'";
    for (int64_t dim : shape)
        if (dim < 0)
            return emitError() << "dimensions of InputType must be static";
    return success();
}

//===----------------------------------------------------------------------===//
// OutputType
//===----------------------------------------------------------------------===//

/// @brief Creates an OutputType for a scalar element type with no shape
/// dimensions.
///
/// This is the simplest form of OutputType where the channel carries a single
/// scalar value of the given @p elementType. Equivalent to calling
/// OutputType::get({}, elementType).
OutputType OutputType::get(Type elementType)
{
    return Base::get(
        elementType.getContext(),
        ArrayRef<int64_t>{},
        elementType);
}

/// @brief Creates an OutputType with an explicit shape and element type.
///
/// The @p shape defines the number of elements along each dimension. All
/// dimensions must be static (non-negative). Pass an empty shape for a scalar
/// channel.
OutputType OutputType::get(ArrayRef<int64_t> shape, Type elementType)
{ return Base::get(elementType.getContext(), shape, elementType); }

/// @brief Creates an OutputType from an existing ShapedType.
///
/// Convenience builder that decomposes @p shapedType into its shape and scalar
/// element type, then constructs the corresponding OutputType. Useful when
/// adapting memref or tensor types to DFG channel types.
OutputType OutputType::get(ShapedType shapedType)
{
    return Base::get(
        shapedType.getContext(),
        shapedType.getShape(),
        shapedType.getElementType());
}

/// @brief Parses an OutputType from its assembly format `<[shape x] type>`.
///
/// Expects a `<` delimiter, an optional dimension list (e.g. `2x3x`), a scalar
/// element type, and a closing `>`. Returns an empty Type on any parse failure.
Type OutputType::parse(AsmParser &parser)
{
    if (parser.parseLess()) return {};

    SmallVector<int64_t> shape;
    Type elementType;

    if (parser.parseDimensionList(shape, /*allowDynamic=*/false)) return {};
    if (parser.parseType(elementType)) return {};
    if (parser.parseGreater()) return {};

    return OutputType::get(shape, elementType);
}

/// @brief Prints an OutputType in its assembly format `<[shape x] type>`.
///
/// Emits `<`, the dimension list (omitting the trailing `x` for scalar types),
/// the element type, and `>`.
void OutputType::print(AsmPrinter &p) const
{
    p << "<";
    p.printDimensionList(getShape());
    if (!getShape().empty()) p << "x";
    p.printType(getElementType());
    p << ">";
}

/// @brief Verifies the invariants of an OutputType.
///
/// Checks that @p elementType is a scalar (not a ShapedType) and that every
/// dimension in @p shape is non-negative (i.e. static).
LogicalResult OutputType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape,
    Type elementType)
{
    if (isa<ShapedType>(elementType))
        return emitError()
               << "element type of OutputType must be a scalar, not a shaped "
                  "type like '"
               << elementType << "'";
    for (int64_t dim : shape)
        if (dim < 0)
            return emitError() << "dimensions of OutputType must be static";
    return success();
}

//===----------------------------------------------------------------------===//
// DFGDialect
//===----------------------------------------------------------------------===//

void DFGDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.cpp.inc"
        >();
}
