/// Declaration of the dfg dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/DFG/DFGEnums.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.h"
#include "laksa-mlir/Dialect/DFG/Interfaces/EdgeInterface.h"
#include "laksa-mlir/Dialect/DFG/Interfaces/GraphInterface.h"
#include "laksa-mlir/Dialect/DFG/Interfaces/NodeInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dfg {

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// NodeInterface Methods
//===------------------------------------------------------------------===//

inline std::string ProcessOp::getNodeName() { return getSymName().str(); }
inline unsigned ProcessOp::getNumInputPorts()
{ return getFunctionType().getNumInputs(); }
inline Value ProcessOp::getInputPort(unsigned index)
{ return getBody().getArgument(index); }
/// @brief Returns a vector of base types of the input ports.
inline SmallVector<Type> ProcessOp::getInputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getInputs())
        types.push_back(cast<OutputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> ProcessOp::getInputPorts()
{
    SmallVector<Value> inputs;
    for (size_t i = 0; i < getNumInputPorts(); ++i)
        inputs.push_back(getInputPort(i));
    return inputs;
}
inline unsigned ProcessOp::getNumOutputPorts()
{ return getFunctionType().getNumResults(); }
inline Value ProcessOp::getOutputPort(unsigned index)
{
    unsigned idx = index + getNumInputPorts();
    return getBody().getArgument(idx);
}
/// @brief Returns a vector of base types of the output ports.
inline SmallVector<Type> ProcessOp::getOutputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getResults()) {
        auto inTy = cast<InputType>(type);
        types.push_back(inTy.getBaseType());
    }
    return types;
}
inline SmallVector<Value> ProcessOp::getOutputPorts()
{
    SmallVector<Value> outputs;
    for (size_t i = 0; i < getNumOutputPorts(); ++i)
        outputs.push_back(getOutputPort(i));
    return outputs;
}

//===------------------------------------------------------------------===//
// SymbolOpInterface Methods
//===------------------------------------------------------------------===//
inline bool ProcessOp::isDeclaration() { return isExternal(); }

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

/// @brief Returns whether the process is externally defined
inline bool ProcessOp::isExternal() { return getBody().empty(); }

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

/// @brief Constructs a function type from the inputs/outputs types.
inline FunctionType LoopOp::getFunctionType()
{
    SmallVector<Type> inputTypes, outputTypes;
    for (auto input : getInputChannels()) inputTypes.push_back(input.getType());
    for (auto output : getOutputChannels())
        outputTypes.push_back(output.getType());
    return FunctionType::get(getContext(), inputTypes, outputTypes);
}

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// NodeInterface Methods
//===------------------------------------------------------------------===//

inline std::string OperatorOp::getNodeName() { return getSymName().str(); }
inline unsigned OperatorOp::getNumInputPorts()
{ return getFunctionType().getNumInputs(); }
inline Value OperatorOp::getInputPort(unsigned index)
{ return getBody().getArgument(index); }
inline SmallVector<Type> OperatorOp::getInputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getInputs()) types.push_back(type);
    return types;
}
inline SmallVector<Value> OperatorOp::getInputPorts()
{
    SmallVector<Value> inputs;
    for (size_t i = 0; i < getNumInputPorts(); ++i)
        inputs.push_back(getInputPort(i));
    return inputs;
}
inline unsigned OperatorOp::getNumOutputPorts()
{ return getFunctionType().getNumResults(); }
inline Value OperatorOp::getOutputPort(unsigned index)
{
    unsigned idx = index + getNumInputPorts();
    return getBody().getArgument(idx);
}
inline SmallVector<Type> OperatorOp::getOutputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getResults()) types.push_back(type);
    return types;
}
inline SmallVector<Value> OperatorOp::getOutputPorts()
{
    SmallVector<Value> outputs;
    for (size_t i = 0; i < getNumOutputPorts(); ++i)
        outputs.push_back(getOutputPort(i));
    return outputs;
}
inline FunctionType OperatorOp::getBaseFunctionType()
{
    SmallVector<Type> inputs, outputs;
    for (auto inTy : getInputPortTypes()) {
        if (auto shapedTy = dyn_cast<ShapedType>(inTy)) {
            inputs.push_back(
                MemRefType::get(
                    shapedTy.getShape(),
                    shapedTy.getElementType()));
        } else {
            inputs.push_back(inTy);
        }
    }
    for (auto outTy : getOutputPortTypes()) {
        if (auto shapedTy = dyn_cast<ShapedType>(outTy)) {
            outputs.push_back(
                MemRefType::get(
                    shapedTy.getShape(),
                    shapedTy.getElementType()));
        } else {
            outputs.push_back(outTy);
        }
    }
    return FunctionType::get(getContext(), inputs, outputs);
}

//===------------------------------------------------------------------===//
// SymbolOpInterface Methods
//===------------------------------------------------------------------===//

inline bool OperatorOp::isDeclaration() { return isExternal(); }

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

/// @brief Returns whether the operator is externally defined
inline bool OperatorOp::isExternal() { return getBody().empty(); }

//===------------------------------------------------------------------===//
// RegionOp
//===------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// NodeInterface Methods
//===------------------------------------------------------------------===//

inline std::string RegionOp::getNodeName() { return getSymName().str(); }
inline unsigned RegionOp::getNumInputPorts()
{ return getFunctionType().getNumInputs(); }
inline Value RegionOp::getInputPort(unsigned index)
{ return getBody().getArgument(index); }
/// @brief Returns a vector of base types of the input ports.
inline SmallVector<Type> RegionOp::getInputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getInputs())
        types.push_back(cast<OutputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> RegionOp::getInputPorts()
{
    SmallVector<Value> inputs;
    for (size_t i = 0; i < getNumInputPorts(); ++i)
        inputs.push_back(getInputPort(i));
    return inputs;
}
inline unsigned RegionOp::getNumOutputPorts()
{ return getFunctionType().getNumResults(); }
inline Value RegionOp::getOutputPort(unsigned index)
{
    unsigned idx = index + getNumInputPorts();
    return getBody().getArgument(idx);
}
/// @brief Returns a vector of base types of the output ports.
inline SmallVector<Type> RegionOp::getOutputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getFunctionType().getResults())
        types.push_back(cast<InputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> RegionOp::getOutputPorts()
{
    SmallVector<Value> outputs;
    for (size_t i = 0; i < getNumOutputPorts(); ++i)
        outputs.push_back(getOutputPort(i));
    return outputs;
}

//===------------------------------------------------------------------===//
// GraphInterface Methods
//===------------------------------------------------------------------===//

inline std::string RegionOp::getGraphName() { return getNodeName(); }
inline bool RegionOp::isSubGraph()
{
    bool isSub = false;
    auto module = getOperation()->getParentOfType<ModuleOp>();
    module.walk([&](EmbedOp embedOp) {
        if (embedOp.getNodeName() == getGraphName()) {
            isSub = true;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return isSub;
}
inline SmallVector<Operation*> RegionOp::getGraphNodes()
{
    SmallVector<Operation*> nodes;
    for (auto &opi : getBody().getOps())
        if (isa<NodeInterface>(opi) && !isa<GraphInterface>(opi))
            nodes.push_back(&opi);
    return nodes;
}
inline SmallVector<Operation*> RegionOp::getGraphSubGs()
{
    SmallVector<Operation*> subgraphs;
    for (auto &opi : getBody().getOps())
        if (isa<NodeInterface>(opi) && isa<GraphInterface>(opi))
            subgraphs.push_back(&opi);
    return subgraphs;
}
inline SmallVector<Operation*> RegionOp::getGraphEdges()
{
    SmallVector<Operation*> edges;
    for (auto &opi : getBody().getOps())
        if (isa<EdgeInterface>(opi)) edges.push_back(&opi);
    return edges;
}

//===------------------------------------------------------------------===//
// ChannelOp
//===------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// EdgeInterface Methods
//===------------------------------------------------------------------===//

inline Operation* ChannelOp::getInputConnectedOp()
{ return *getInputPort().getUsers().begin(); }

inline Operation* ChannelOp::getOutputConnectedOp()
{ return *getOutputPort().getUsers().begin(); }

//===------------------------------------------------------------------===//
// MemoryEffectsOpInterface Methods
//===------------------------------------------------------------------===//

/// @brief Populates the memory-effect list required by
/// MemoryEffectsOpInterface.
///
/// Both a Read and a Write effect are attached to the channel's `input_port`
/// and `output_port` results on the DefaultResource.  This pair serves two
/// purposes:
///
///  - **Write** marks the channel as a definition site for the FIFO buffer,
///    preventing DCE from removing a channel that has no further SSA uses
///    within the same block.
///  - **Read** prevents CSE from merging two distinct channel declarations that
///    happen to carry the same token type, since each channel is a unique,
///    stateful FIFO object.
///
/// Together they conservatively model the channel as an opaque side-effecting
/// operation, ensuring that standard canonicalization passes do not reorder,
/// deduplicate, or eliminate channel declarations.
inline void ChannelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects)
{
    effects.emplace_back(
        MemoryEffects::Read::get(),
        getOperation()->getOpResult(0),
        SideEffects::DefaultResource::get());
    effects.emplace_back(
        MemoryEffects::Write::get(),
        getOperation()->getOpResult(1),
        SideEffects::DefaultResource::get());
}

//===------------------------------------------------------------------===//
// InstantiateOp
//===------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// NodeInterface Methods
//===------------------------------------------------------------------===//

inline std::string InstantiateOp::getNodeName()
{ return getActor().getRootReference().str(); }
inline unsigned InstantiateOp::getNumInputPorts()
{ return std::ranges::size(getInputs()); }
inline Value InstantiateOp::getInputPort(unsigned index)
{ return getOperation()->getOperand(index); }
inline SmallVector<Type> InstantiateOp::getInputPortTypes()
{
    SmallVector<Type> types;
    // types.clear();
    for (auto type : getInputs().getTypes())
        types.push_back(cast<OutputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> InstantiateOp::getInputPorts() { return getInputs(); }
inline unsigned InstantiateOp::getNumOutputPorts()
{ return std::ranges::size(getOutputs()); }
inline Value InstantiateOp::getOutputPort(unsigned index)
{
    unsigned idx = index + getNumInputPorts();
    return getOperation()->getOperand(idx);
}
inline SmallVector<Type> InstantiateOp::getOutputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getOutputs().getTypes())
        types.push_back(cast<InputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> InstantiateOp::getOutputPorts()
{ return getOutputs(); }

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

/// @brief Finds the instantiated process or operator in this operation.
inline Operation* InstantiateOp::getInstantiatedOperation()
{
    Operation* actor = nullptr;

    auto module = getOperation()->getParentOfType<ModuleOp>();
    if (!module) return actor;

    std::string actorName = getNodeName();
    module->walk([&](Operation* op) {
        if (auto processOp = dyn_cast<ProcessOp>(op)) {
            if (processOp.getNodeName() == actorName) actor = op;
        } else if (auto operatorOp = dyn_cast<OperatorOp>(op)) {
            if (operatorOp.getNodeName() == actorName) actor = op;
        }
    });

    return actor;
}

/// @brief Constructs a function type from the inputs/outputs types.
inline FunctionType InstantiateOp::getFunctionType()
{
    return FunctionType::get(
        getContext(),
        getInputs().getTypes(),
        getOutputs().getTypes());
}

//===------------------------------------------------------------------===//
// EmbedOp
//===------------------------------------------------------------------===//

//===------------------------------------------------------------------===//
// DFGMLIR NodeOpInterface Methods
//===------------------------------------------------------------------===//

inline std::string EmbedOp::getNodeName()
{ return getActor().getRootReference().str(); }
inline unsigned EmbedOp::getNumInputPorts()
{ return std::ranges::size(getInputs()); }
inline Value EmbedOp::getInputPort(unsigned index)
{ return getOperation()->getOperand(index); }
inline SmallVector<Type> EmbedOp::getInputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getInputs().getTypes())
        types.push_back(cast<OutputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> EmbedOp::getInputPorts() { return getInputs(); }
inline unsigned EmbedOp::getNumOutputPorts()
{ return std::ranges::size(getOutputs()); }
inline Value EmbedOp::getOutputPort(unsigned index)
{
    unsigned idx = index + getNumInputPorts();
    return getOperation()->getOperand(idx);
}
inline SmallVector<Type> EmbedOp::getOutputPortTypes()
{
    SmallVector<Type> types;
    for (auto type : getOutputs().getTypes())
        types.push_back(cast<InputType>(type).getBaseType());
    return types;
}
inline SmallVector<Value> EmbedOp::getOutputPorts() { return getOutputs(); }

//===------------------------------------------------------------------===//
// DFGMLIR GraphOpInterface Methods
//===------------------------------------------------------------------===//

inline std::string EmbedOp::getGraphName() { return getNodeName(); }
inline bool EmbedOp::isSubGraph() { return true; }

//===------------------------------------------------------------------===//
// Custom Methods
//===------------------------------------------------------------------===//

/// @brief Finds the embedded region in this operation.
inline Operation* EmbedOp::getEmbeddedOperation()
{
    Operation* actor = nullptr;

    auto module = getOperation()->getParentOfType<ModuleOp>();
    if (!module) return actor;

    std::string actorName = getNodeName();
    module->walk([&](Operation* op) {
        if (auto regionOp = dyn_cast<RegionOp>(op)) {
            if (regionOp.getNodeName() == actorName) actor = op;
        }
    });

    return actor;
}

/// @brief Constructs a function type from the inputs/outputs types.
inline FunctionType EmbedOp::getFunctionType()
{
    return FunctionType::get(
        getContext(),
        getInputs().getTypes(),
        getOutputs().getTypes());
}

} // namespace mlir::dfg
