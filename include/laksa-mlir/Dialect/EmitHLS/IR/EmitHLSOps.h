/// Declaration of the dfg dialect ops.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/EmitHLSEnums.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"
#include "laksa-mlir/Dialect/EmitHLS/Interfaces/PragmaInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::emithls {

//===----------------------------------------------------------------------===//
// Structure operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// FunctionOpInterface Methods
//===------------------------------------------------------------------===//

/// @brief Returns the region on the current operation that is callable.
// This may return null in the case of an external callable object, e.g. an
// external function.
inline Region* FuncOp::getCallableRegion()
{ return isExternal() ? nullptr : &getBody(); }

inline ArrayRef<Type> FuncOp::getArgumentTypes()
{ return getFunctionType().getInputs(); }

inline ArrayRef<Type> FuncOp::getResultTypes()
{ return getFunctionType().getResults(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// CallOpInterface Methods
//===------------------------------------------------------------------===//

inline OperandRange CallOp::getArgOperands()
{ return {arg_operand_begin(), arg_operand_end()}; }

inline MutableOperandRange CallOp::getArgOperandsMutable()
{ return getOperandsMutable(); }

inline OperandRange::iterator CallOp::arg_operand_begin()
{ return operand_begin(); }

inline OperandRange::iterator CallOp::arg_operand_end()
{ return operand_end(); }

inline CallInterfaceCallable CallOp::getCallableForCallee()
{ return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }

inline void CallOp::setCalleeFromCallable(CallInterfaceCallable callee)
{ (*this)->setAttr("callee", cast<SymbolRefAttr>(callee)); }

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

/// @brief Returns true if a variable is defiend as constant.
inline bool VariableOp::hasInit() { return getInitNumber() || getInitValue(); }

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//
inline void UpdateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getVariableMutable(),
        SideEffects::DefaultResource::get());
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getVariableMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ExpressionOp
//===----------------------------------------------------------------------===//

/// @brief Returns the operation that defines the value to be yielded.
inline Operation* ExpressionOp::getRootOp()
{
    auto yieldOp = dyn_cast<YieldOp>((&getBody().front())->getTerminator());
    auto yieldValue = yieldOp.getValue();
    return yieldValue.getDefiningOp();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

inline Value ForOp::getInductionVariable() { return getBody().getArgument(0); }

inline int64_t ForOp::getTripCount()
{
    int64_t lb = getLowerBound().getSExtValue();
    int64_t ub = getUpperBound().getSExtValue();
    int64_t step = getStep().getSExtValue();
    return (ub - lb + step - 1) / step;
}

//===----------------------------------------------------------------------===//
// Arith operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArithFusedOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

inline void ArithFusedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getAccMutable(),
        SideEffects::DefaultResource::get());
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getAccMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayReadOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//
inline void ArrayReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getArrayMutable(),
        SideEffects::DefaultResource::get());
}

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

inline ArrayType ArrayReadOp::getArrayType() { return getArray().getType(); }

//===----------------------------------------------------------------------===//
// ArrayWriteOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//
inline void ArrayWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getArrayMutable(),
        SideEffects::DefaultResource::get());
}

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

inline ArrayType ArrayWriteOp::getArrayType() { return getArray().getType(); }

//===----------------------------------------------------------------------===//
// ArrayPointerReadOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

inline void ArrayPointerReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getPointerMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ArrayPointerWriteOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

inline void ArrayPointerWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getPointerMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Stream operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StreamReadOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

inline void StreamReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getStreamMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// StreamWriteOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

inline void StreamWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Write::get(),
        &getStreamMutable(),
        SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// Pragma operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PragmaArrayPartitionOp
//===----------------------------------------------------------------------===//

inline Type PragmaArrayPartitionOp::getVariableType()
{ return getVariable().getType(); }

//===----------------------------------------------------------------------===//
// PragmaBindStorageOp
//===----------------------------------------------------------------------===//

inline Type PragmaBindStorageOp::getVariableType()
{ return getVariable().getType(); }

//===----------------------------------------------------------------------===//
// PragmaStreamOp
//===----------------------------------------------------------------------===//

inline Type PragmaStreamOp::getVariableType()
{ return getVariable().getType(); }

} // namespace mlir::emithls
