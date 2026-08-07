/// Declaration of the FuncToDFG that converts Func to DFG dialect
/// and their instances.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

//===- Generated includes -------------------------------------------------===//

#define GEN_PASS_DECL_CONVERTFUNCTODFG
#include "laksa-mlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//

void populateFuncToDFGConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    DenseMap<Value, Value> &returnValueToOutputPortMap,
    DenseMap<Value, Value> &callResultChannelMap);

std::unique_ptr<Pass> createConvertFuncToDFGPass();

} // namespace mlir
