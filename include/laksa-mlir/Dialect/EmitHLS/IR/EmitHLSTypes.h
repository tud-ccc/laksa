/// Declaration of the EmitHLS dialect types.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Types.h"

//===- Generated Includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::emithls {

//===------------------------------------------------------------------===//
// ArrayType
//===------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// ShapedTypeInterface Methods
//===------------------------------------------------------------------===//

/// @brief For HLS, an array should always have rank
inline bool ArrayType::hasRank() const { return true; }

inline ArrayType ArrayType::cloneWith(
    std::optional<ArrayRef<int64_t>> shape,
    Type elementType) const
{ return ArrayType::get(shape.value_or(getShape()), elementType); }

} // namespace mlir::emithls
