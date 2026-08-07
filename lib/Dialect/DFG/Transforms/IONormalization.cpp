/// Implementation of IONormalization transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/WalkResult.h>
#include <utility>

#define DEBUG_TYPE "dfg-io-normalization"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[dfg-io-normalization] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::dfg;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGIONORMALIZATION
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

namespace {

// Helper structs and so on
enum class IOStyle {
    Accumulate,
    Parallel,
    Other,
    Unknown,
};
using AxisShape = SmallVector<std::pair<int64_t, int64_t>>;
struct ProcessPortShapes {
    IOStyle style = IOStyle::Unknown;
    SmallVector<AxisShape> inputs;
    SmallVector<AxisShape> outputs;
};
using NormalizedShapeMap = DenseMap<ProcessOp, ProcessPortShapes>;
struct PortConnection {
    ProcessOp process;
    unsigned portIndex = 0;
};
struct ProcessNeighbors {
    SmallVector<PortConnection> inputProducers;
    SmallVector<SmallVector<PortConnection>> outputConsumers;
};
using ConnectivityMap = DenseMap<ProcessOp, ProcessNeighbors>;

// Walks back through any chain of view-like ops to find the defining op.
Operation* findBackingDef(Value memrefVal)
{
    while (auto viewOp = memrefVal.getDefiningOp<ViewLikeOpInterface>())
        memrefVal = viewOp.getViewSource();
    return memrefVal.getDefiningOp();
}
// Walks back through any chain of view-like ops to find the alloc in use.
memref::AllocOp findBackingAlloc(Value memrefVal)
{ return dyn_cast_or_null<memref::AllocOp>(findBackingDef(memrefVal)); }

// Returns the first op of type OpTy found by walking root, or null.
template<typename OpTy>
static OpTy findFirst(Operation* root)
{
    OpTy result = nullptr;
    root->walk([&](OpTy candidate) {
        result = candidate;
        return WalkResult::interrupt();
    });
    return result;
}

// Appends the chain of view-like ops between cursor and landmark to deadOps.
static void collectViewChain(
    Value cursor,
    Operation* landmark,
    SmallVectorImpl<Operation*> &deadOps)
{
    while (Operation* def = cursor.getDefiningOp()) {
        if (def == landmark) break;
        auto viewOp = dyn_cast<ViewLikeOpInterface>(def);
        if (!viewOp) break;
        deadOps.push_back(viewOp);
        cursor = viewOp.getViewSource();
    }
}

// Maps every value in operands through mapper, defaulting to itself.
static SmallVector<Value> mapOperands(IRMapping &mapper, ValueRange operands)
{
    return llvm::map_to_vector(operands, [&](Value v) {
        return mapper.lookupOrDefault(v);
    });
}

// Walks backward through a chain of view-like ops starting at the provided
// value, map the axis index to the one in the root's shape Currently, only
// collapse/expand with statically size-1 dimension is supported
static FailureOr<int64_t>
translateAxisBackwardThroughViews(Operation* op, Value val, int64_t axis)
{
    while (auto viewOp = val.getDefiningOp<ViewLikeOpInterface>()) {
        if (auto expandOp =
                dyn_cast<memref::ExpandShapeOp>(viewOp.getOperation())) {
            SmallVector<ReassociationIndices, 4> reassociation =
                expandOp.getReassociationIndices();
            bool found = false;
            for (auto [srcAxis, group] : llvm::enumerate(reassociation)) {
                if (llvm::is_contained(group, axis)) {
                    axis = srcAxis;
                    found = true;
                    break;
                }
            }
            if (!found)
                return op->emitError(
                    "could not find the kept axis in a "
                    "memref.expand_shape's reassociation");
        } else if (
            auto collapseOp =
                dyn_cast<memref::CollapseShapeOp>(viewOp.getOperation())) {
            SmallVector<ReassociationIndices, 4> reassociation =
                collapseOp.getReassociationIndices();
            ArrayRef<int64_t> group = reassociation[axis];
            MemRefType srcType = collapseOp.getSrcType();
            std::optional<int64_t> realAxis;
            for (int64_t candidate : group) {
                if (srcType.getDimSize(candidate) == 1) continue;
                if (realAxis)
                    return op->emitError(
                        "cannot translate a kept axis through a "
                        "memref.collapse_shape that merges it with more "
                        "than one non-unit-size axis");
                realAxis = candidate;
            }
            // If every axis in the group is statically size 1, the merged
            // axis is itself size 1, so any of them is an equally valid.
            axis = realAxis ? *realAxis : group.front();
        }
        val = viewOp.getViewSource();
    }
    return axis;
}

// Searches axes [0, rank) of val, translating each backward through views via
// translateAxisBackwardThroughViews, for the one that lands on target.
// Returns std::nullopt (not a failure) if none does.
static FailureOr<std::optional<int64_t>> findAxisTranslatingTo(
    Operation* diagOp,
    Value val,
    int64_t rank,
    int64_t target)
{
    for (int64_t candidate = 0; candidate < rank; ++candidate) {
        auto translated =
            translateAxisBackwardThroughViews(diagOp, val, candidate);
        if (failed(translated)) return failure();
        if (*translated == target) return std::optional<int64_t>(candidate);
    }
    return std::optional<int64_t>();
}

// A channel's connected op must be a dfg.instantiate of a dfg.process.
static FailureOr<InstantiateOp> getConnectedInstantiate(Operation* op)
{
    auto instantiateOp = dyn_cast<InstantiateOp>(op);
    if (!instantiateOp)
        return op->emitError(
            "Expected a channel to be connected to a dfg.instantiate");
    if (!isa<ProcessOp>(instantiateOp.getInstantiatedOperation()))
        return instantiateOp.emitError(
            "Expected a dfg.process to be instantiated");
    return instantiateOp;
}
static unsigned findPortIndex(ValueRange ports, Value port)
{
    for (auto [index, candidate] : llvm::enumerate(ports))
        if (candidate == port) return index;
    llvm_unreachable("port not found in its own instantiate's port list");
}
static FailureOr<unsigned>
getBoundaryArgNumber(InstantiateOp instantiateOp, Value port)
{
    auto blockArg = dyn_cast<BlockArgument>(port);
    if (!blockArg)
        return instantiateOp.emitError(
            "Expected a port not driven by a dfg.channel to be a region "
            "block argument");
    return blockArg.getArgNumber();
}
// Build the connectivity map of the input graph
static FailureOr<ConnectivityMap> buildConnectivity(RegionOp regionOp)
{
    ConnectivityMap connectivity;
    for (Operation* node : regionOp.getGraphNodes()) {
        auto instantiateOp = getConnectedInstantiate(node);
        if (failed(instantiateOp)) return failure();
        auto processOp =
            cast<ProcessOp>(instantiateOp->getInstantiatedOperation());

        ProcessNeighbors &neighbors = connectivity[processOp];
        for (Value inputPort : instantiateOp->getInputs()) {
            auto channelOp = inputPort.getDefiningOp<ChannelOp>();
            if (!channelOp) {
                auto argNumber =
                    getBoundaryArgNumber(*instantiateOp, inputPort);
                if (failed(argNumber)) return failure();
                neighbors.inputProducers.push_back({nullptr, *argNumber});
                continue;
            }
            auto producerInstantiate =
                getConnectedInstantiate(channelOp.getInputConnectedOp());
            if (failed(producerInstantiate)) return failure();
            auto producerProcess = cast<ProcessOp>(
                producerInstantiate->getInstantiatedOperation());
            unsigned portIndex = findPortIndex(
                producerInstantiate->getOutputs(),
                channelOp.getInputPort());
            neighbors.inputProducers.push_back({producerProcess, portIndex});
        }
        for (Value outputPort : instantiateOp->getOutputs()) {
            auto channelOp = outputPort.getDefiningOp<ChannelOp>();
            SmallVector<PortConnection> consumers;
            if (!channelOp) {
                auto argNumber =
                    getBoundaryArgNumber(*instantiateOp, outputPort);
                if (failed(argNumber)) return failure();
                consumers.push_back({nullptr, *argNumber});
            } else {
                for (Operation* user : channelOp.getOutputPort().getUsers()) {
                    auto consumerInstantiate = getConnectedInstantiate(user);
                    if (failed(consumerInstantiate)) return failure();
                    auto consumerProcess = cast<ProcessOp>(
                        consumerInstantiate->getInstantiatedOperation());
                    unsigned portIndex = findPortIndex(
                        consumerInstantiate->getInputs(),
                        channelOp.getOutputPort());
                    consumers.push_back({consumerProcess, portIndex});
                }
            }
            neighbors.outputConsumers.push_back(std::move(consumers));
        }
    }
    return connectivity;
}

// Custom print of the helper structs and so on
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, IOStyle style)
{
    switch (style) {
    case IOStyle::Accumulate: return os << "accumulate";
    case IOStyle::Parallel: return os << "parallel";
    case IOStyle::Other: return os << "other";
    case IOStyle::Unknown: return os << "unknown";
    }
    llvm_unreachable("unhandled IOStyle");
}
static void printShapeList(llvm::raw_ostream &os, ArrayRef<AxisShape> shapeList)
{
    os << "[";
    llvm::interleaveComma(shapeList, os, [&](const AxisShape &shape) {
        os << "<";
        llvm::interleaveComma(shape, os, [&](std::pair<int64_t, int64_t> p) {
            os << "idx#" << p.first << "->" << p.second;
        });
        os << ">";
    });
    os << "]";
}
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ProcessPortShapes &shapes)
{
    os << "{style: " << shapes.style << ", inputs: ";
    printShapeList(os, shapes.inputs);
    os << ", outputs: ";
    printShapeList(os, shapes.outputs);
    os << "}";
    return os;
}
static void printPortConnection(llvm::raw_ostream &os, PortConnection port)
{
    if (port.process)
        os << port.process.getNodeName() << "#" << port.portIndex;
    else
        os << "boundary#" << port.portIndex;
}
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ProcessNeighbors &neighbors)
{
    os << "{inputs from: [";
    llvm::interleaveComma(
        neighbors.inputProducers,
        os,
        [&](PortConnection producer) { printPortConnection(os, producer); });
    os << "], outputs to: [";
    llvm::interleaveComma(
        neighbors.outputConsumers,
        os,
        [&](const SmallVector<PortConnection> &consumers) {
            os << "[";
            llvm::interleaveComma(consumers, os, [&](PortConnection consumer) {
                printPortConnection(os, consumer);
            });
            os << "]";
        });
    os << "]}";
    return os;
}

// Precomputes every instantiated process' normalized port shapes
FailureOr<NormalizedShapeMap> computeNormalizedShapes(ModuleOp moduleOp)
{
    NormalizedShapeMap shapes;

    // There is exactly one dfg.region, and it is the last top-level operation,
    // after dfg-inline-embed-region has flattened everything else.
    Block &moduleBody = moduleOp.getBodyRegion().front();
    RegionOp regionOp =
        moduleBody.empty() ? nullptr : dyn_cast<RegionOp>(&moduleBody.back());
    if (!regionOp)
        return moduleOp.emitError(
            "Expected the last top-level operation to be a dfg.region");
    if (llvm::count_if(
            moduleBody,
            [](Operation &opi) { return isa<RegionOp>(opi); })
        != 1)
        return moduleOp.emitError(
            "Expected exactly one `dfg.region` after dfg-inline-embed-region");
    LAKSA_DEBUG(
        llvm::dbgs() << "Analyzing region \"" << regionOp.getNodeName()
                     << "\" for normalized port shapes");

    // Used by code below to resolve shapes that depend on a neighbor.
    auto connectivityOrFailure = buildConnectivity(regionOp);
    if (failed(connectivityOrFailure)) return failure();
    ConnectivityMap connectivity = std::move(*connectivityOrFailure);
    LAKSA_DEBUG(llvm::dbgs() << "  The connectivities are as follow");
    for (auto &[processOp, neighbors] : connectivity)
        LAKSA_DEBUG(
            llvm::dbgs() << "    Process \"" << processOp.getNodeName()
                         << "\" has neighbors: " << neighbors);

    // Classify every process's IOStyle from its own output alloc based on the
    // attribute attached to the definition of this alloc.
    LAKSA_DEBUG(llvm::dbgs() << "  Node shapes are as follow");
    for (Operation* node : regionOp.getGraphNodes()) {
        auto instantiateOp = getConnectedInstantiate(node);
        if (failed(instantiateOp)) return failure();
        auto processOp =
            cast<ProcessOp>(instantiateOp->getInstantiatedOperation());
        if (shapes.contains(processOp)) continue;

        ProcessPortShapes portShapes;
        // A process may push more than one output port, or push the same port
        // using different value.
        SmallVector<PushMemRefOp> pushOps;
        processOp.walk(
            [&](PushMemRefOp candidate) { pushOps.push_back(candidate); });
        if (pushOps.empty())
            return moduleOp.emitError(
                "Expect a push op to flush a memref into output port(s)");

        // For now, every push in a process must flush the same single
        // output alloc.
        auto allocOp = findBackingAlloc(pushOps.front().getTokenMemref());
        for (PushMemRefOp pushOp : pushOps)
            if (findBackingAlloc(pushOp.getTokenMemref()) != allocOp)
                return processOp.emitError(
                    "Expect every push in a process to flush the same "
                    "output alloc");

        if (allocOp && allocOp->hasAttr("accu_at"))
            portShapes.style = IOStyle::Accumulate;
        else if (allocOp && allocOp->hasAttr("parallel"))
            portShapes.style = IOStyle::Parallel;
        else if (allocOp && allocOp->hasAttr("fill_with"))
            portShapes.style = IOStyle::Other;

        // For now, only parallel-style processes are allowed more than one
        // push.
        if (pushOps.size() > 1 && portShapes.style != IOStyle::Parallel)
            return processOp.emitError(
                "Only parallel-style processes may push more than one output "
                "for now");

        LAKSA_DEBUG(
            llvm::dbgs() << "    Node \"" << processOp.getNodeName()
                         << "\" classified as " << portShapes.style);

        // Accumulate style: the output port shape is made up of the alloc's
        // accu_at axes (axis index, size), and an empty one means a scalar
        // port.
        if (portShapes.style == IOStyle::Accumulate) {
            auto accuAt = allocOp->getAttrOfType<ArrayAttr>("accu_at");
            Value pushedVal = pushOps.front().getTokenMemref();
            auto pushedType = cast<MemRefType>(pushedVal.getType());

            // "accu_at" is computed against this node's own raw output alloc,
            // which can differ from what actually crosses the port through a
            // chain of view-like op reshapes. Translate each axis here to be
            // relative to the declared output port type.
            AxisShape outputShape;
            for (Attribute axisAttr : accuAt) {
                int64_t rawAxis = cast<IntegerAttr>(axisAttr).getInt();
                int64_t size = allocOp.getType().getDimSize(rawAxis);
                auto declaredAxis = findAxisTranslatingTo(
                    processOp,
                    pushedVal,
                    pushedType.getRank(),
                    rawAxis);
                if (failed(declaredAxis)) return failure();
                if (!*declaredAxis) {
                    return processOp.emitError()
                           << "could not find a declared output port axis "
                              "that translates down to accu_at axis "
                           << rawAxis;
                }
                outputShape.emplace_back(**declaredAxis, size);
            }
            portShapes.outputs.push_back(outputShape);

            auto linebufOp = findFirst<emithls::HelperLineBufferOp>(
                processOp.getOperation());
            if (!linebufOp)
                return processOp.emitError(
                    "expected an emithls.helper.linebuf in an "
                    "accumulate-style process");

            AxisShape inputShape;
            if (linebufOp.getNumLine() > 0) {
                auto numChan = linebufOp.getNumChan();
                if (numChan.size() != accuAt.size())
                    return processOp.emitError(
                        "expected linebuf's num_chan to have as many "
                        "entries as the output alloc's accu_at");
                // "num_chan" entries line up positionally with "accu_at" axes.
                for (auto [chanAttr, outAxis] : llvm::zip(numChan, outputShape))
                    inputShape.emplace_back(
                        outAxis.first,
                        cast<IntegerAttr>(chanAttr).getInt());
            } else {
                // Zero line buffering to get only a vector. This axis is a
                // genuine position in linebuf's own token memref, so
                // translate it back up to the declared input port type.
                auto shapedType =
                    cast<ShapedType>(linebufOp.getTokenRef().getType());
                int64_t lastAxis = shapedType.getRank() - 1;
                auto declaredAxis = translateAxisBackwardThroughViews(
                    processOp,
                    linebufOp.getTokenRef(),
                    lastAxis);
                if (failed(declaredAxis)) return failure();
                inputShape.emplace_back(
                    *declaredAxis,
                    shapedType.getShape().back());
            }
            portShapes.inputs.push_back(std::move(inputShape));
            LAKSA_DEBUG(
                llvm::dbgs() << "    Node \"" << processOp.getNodeName()
                             << "\" shapes: " << portShapes);
        }
        shapes[processOp] = std::move(portShapes);
    }

    // Retargets an AxisShape to a different port by matching each kept axis to
    // the axis of the same size in that port's declared shape. Ports can differ
    // in rank, so axis indices don't transfer directly.
    auto retargetAxisShape =
        [](Operation* diagOp,
           const AxisShape &shape,
           ArrayRef<int64_t> targetPortShape) -> FailureOr<AxisShape> {
        AxisShape result;
        for (auto [axis, size] : shape) {
            std::optional<int64_t> found;
            for (auto [candidate, candidateSize] :
                 llvm::enumerate(targetPortShape)) {
                if (candidateSize != size) continue;
                if (found)
                    return diagOp->emitError()
                           << "ambiguous axis translation: more than one "
                              "axis of size "
                           << size << " in the target port's declared type";
                found = candidate;
            }
            if (!found)
                return diagOp->emitError()
                       << "could not find an axis of size " << size
                       << " in the target port's declared type";
            result.emplace_back(*found, size);
        }
        return result;
    };

    // Resolves one parallel/other-style process' port shapes from whichever
    // neighbor is already resolved: an upstream producer's output, or a
    // downstream consumer's input.
    auto resolveFromAnyResolvedNeighbor =
        [&](ProcessOp processOp) -> FailureOr<bool> {
        ProcessPortShapes &portShapes = shapes[processOp];
        if (!portShapes.inputs.empty() || !portShapes.outputs.empty())
            return true;
        const ProcessNeighbors &neighbors = connectivity.lookup(processOp);

        const AxisShape* resolvedShape = nullptr;
        for (PortConnection producer : neighbors.inputProducers) {
            if (!producer.process) continue;
            const ProcessPortShapes &producerShapes =
                shapes.at(producer.process);
            if (producer.portIndex < producerShapes.outputs.size()) {
                resolvedShape = &producerShapes.outputs[producer.portIndex];
                break;
            }
        }
        if (!resolvedShape) {
            for (const SmallVector<PortConnection> &consumers :
                 neighbors.outputConsumers) {
                for (PortConnection consumer : consumers) {
                    if (!consumer.process) continue;
                    const ProcessPortShapes &consumerShapes =
                        shapes.at(consumer.process);
                    if (consumer.portIndex < consumerShapes.inputs.size()) {
                        resolvedShape =
                            &consumerShapes.inputs[consumer.portIndex];
                        break;
                    }
                }
                if (resolvedShape) break;
            }
        }
        if (!resolvedShape) return false;

        // getInputPortTypes()/getOutputPortTypes() return each port's base
        // type: a MemRefType for shaped ports, or the bare element type for
        // scalar ports (see DFG_TypeInterface::getBaseType).
        auto portShape = [](Type baseType) -> ArrayRef<int64_t> {
            if (auto shapedType = dyn_cast<ShapedType>(baseType))
                return shapedType.getShape();
            return {};
        };

        SmallVector<Type> inputPortTypes = processOp.getInputPortTypes();
        portShapes.inputs.resize(inputPortTypes.size());
        for (auto [i, portType] : llvm::enumerate(inputPortTypes)) {
            auto translated = retargetAxisShape(
                processOp,
                *resolvedShape,
                portShape(portType));
            if (failed(translated)) return failure();
            portShapes.inputs[i] = std::move(*translated);
        }
        SmallVector<Type> outputPortTypes = processOp.getOutputPortTypes();
        portShapes.outputs.resize(outputPortTypes.size());
        for (auto [j, portType] : llvm::enumerate(outputPortTypes)) {
            auto translated = retargetAxisShape(
                processOp,
                *resolvedShape,
                portShape(portType));
            if (failed(translated)) return failure();
            portShapes.outputs[j] = std::move(*translated);
        }
        return true;
    };

    // Resolves every process of one IOStyle by repeatedly sweeping it,
    // resolving whichever ones a pass can , until a pass makes no more
    // progress.
    auto resolvePhase = [&](IOStyle style) -> LogicalResult {
        SmallVector<ProcessOp> pending;
        for (auto &[processOp, portShapes] : shapes)
            if (portShapes.style == style) pending.push_back(processOp);

        for (bool progress = true; progress && !pending.empty();) {
            progress = false;
            for (unsigned i = 0; i < pending.size();) {
                auto resolved = resolveFromAnyResolvedNeighbor(pending[i]);
                if (failed(resolved)) return failure();
                if (*resolved) {
                    pending.erase(pending.begin() + i);
                    progress = true;
                } else {
                    ++i;
                }
            }
        }
        for (ProcessOp processOp : pending)
            return processOp.emitError(
                "expected a parallel/other-style process' shape to be "
                "resolvable from an already-resolved producer or consumer");
        return success();
    };

    // Accumulate-style is already resolved by this point; resolve
    // parallel-style next, then other-style last.
    if (failed(resolvePhase(IOStyle::Parallel))) return failure();
    if (failed(resolvePhase(IOStyle::Other))) return failure();

    for (auto &[processOp, portShapes] : shapes) {
        if (portShapes.style != IOStyle::Parallel
            && portShapes.style != IOStyle::Other)
            continue;
        LAKSA_DEBUG(
            llvm::dbgs() << "    Node \"" << processOp.getNodeName()
                         << "\" shapes: " << portShapes);
    }

    return shapes;
}

} // namespace

namespace {
// Extracts an AxisShape's kept sizes, in order, as a plain shape.
static SmallVector<int64_t> axisShapeSizes(const AxisShape &axisShape)
{
    return llvm::map_to_vector(axisShape, [](std::pair<int64_t, int64_t> axis) {
        return axis.second;
    });
}
// Builds a normalized dfg port type, keeping the original element type but
// replacing the shape with axisShape's sizes, or a scalar type if it's empty.
template<typename PortType>
static PortType
buildNormalizedPortType(Type elementType, const AxisShape &axisShape)
{
    if (axisShape.empty()) return PortType::get(elementType);
    return PortType::get(axisShapeSizes(axisShape), elementType);
}

// Builds a process' normalized function type from its old (declared) one and
// its precomputed normalization target.
static FunctionType buildNormalizedFunctionType(
    ProcessOp op,
    const ProcessPortShapes &target,
    PatternRewriter &rewriter)
{
    SmallVector<Type> newInputTypes, newOutputTypes;
    for (auto [portIndex, axisShape] : llvm::enumerate(target.inputs))
        newInputTypes.push_back(
            buildNormalizedPortType<OutputType>(
                cast<OutputType>(op.getFunctionType().getInput(portIndex))
                    .getElementType(),
                axisShape));
    for (auto [portIndex, axisShape] : llvm::enumerate(target.outputs))
        newOutputTypes.push_back(
            buildNormalizedPortType<InputType>(
                cast<InputType>(op.getFunctionType().getResult(portIndex))
                    .getElementType(),
                axisShape));
    return rewriter.getFunctionType(newInputTypes, newOutputTypes);
}

// Rewrites a single process' body according to its precomputed normalization
// target. Not a pattern itself, but called directly, once per process, from
// NormalizeRegion's matchAndRewrite so a region's processes and wiring
// normalize atomically in one step.
struct ProcessBodyRewriter {
    ProcessBodyRewriter(NormalizedShapeMap &shapes) : shapes(shapes) {}

    LogicalResult rewriteProcess(
        ProcessOp op,
        const ProcessPortShapes &target,
        PatternRewriter &rewriter,
        ProcessOp &newProcessOpOut) const
    {
        switch (target.style) {
        case IOStyle::Accumulate:
            LAKSA_DEBUG(llvm::dbgs() << "  Using accumulate style body rewrite");
            return rewriteAccumulateStyle(
                op,
                target,
                rewriter,
                newProcessOpOut);
        case IOStyle::Parallel:
            LAKSA_DEBUG(llvm::dbgs() << "  Using parallel style body rewrite");
            return rewriteParallelStyle(op, target, rewriter, newProcessOpOut);
        case IOStyle::Other:
            LAKSA_DEBUG(llvm::dbgs() << "  Using other style body rewrite");
            return rewriteOtherStyle(op, target, rewriter, newProcessOpOut);
        case IOStyle::Unknown:
            return op.emitError(
                "output alloc has no recognized IO-normalization pattern");
        }
        llvm_unreachable("unhandled IOStyle");
    }

private:
    LogicalResult rewriteAccumulateStyle(
        ProcessOp op,
        const ProcessPortShapes &target,
        PatternRewriter &rewriter,
        ProcessOp &newProcessOpOut) const
    {
        auto newFunctionType =
            buildNormalizedFunctionType(op, target, rewriter);
        LAKSA_DEBUG(llvm::dbgs() << "  The new signature is " << newFunctionType);

        auto newProcessOp = ProcessOp::create(
            rewriter,
            op.getLoc(),
            op.getSymName(),
            newFunctionType);
        shapes[newProcessOp] = target;
        Block* newBlock = &newProcessOp.getBody().front();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created new process \"" << newProcessOp.getSymName()
            << "\" with new signature");

        // Find the ops to be rewritten in original process.
        auto pullOp = findFirst<PullAsMemRefOp>(op.getOperation());
        auto pushOp = findFirst<PushMemRefOp>(op.getOperation());
        auto linebufOp =
            findFirst<emithls::HelperLineBufferOp>(op.getOperation());
        auto accumulateOp =
            findFirst<emithls::HelperAccumulateOp>(op.getOperation());
        if (!pullOp || !pushOp || !linebufOp || !accumulateOp)
            return op.emitError(
                "expected an accumulate-style process to contain "
                "pull_as_memref, linebuf, accumulate and push_memref");

        memref::AllocOp allocOp = findBackingAlloc(pushOp.getTokenMemref());
        if (!allocOp)
            return op.emitError(
                "expected the pushed memref to trace back to a memref.alloc");
        LAKSA_DEBUG(llvm::dbgs() << "  Backing alloc for push: " << allocOp);

        auto streamingLoop =
            dyn_cast<affine::AffineForOp>(linebufOp->getParentOp());
        if (!streamingLoop)
            return op.emitError(
                "expected linebuf's parent loop to be an affine.for");
        LAKSA_DEBUG(
            llvm::dbgs() << "  Operation linebuf's streaming affine.for is at "
                         << streamingLoop.getLoc());

        LoopOp oldLoopOp = pullOp->getParentOfType<LoopOp>();
        if (!oldLoopOp)
            return op.emitError(
                "expected pull_as_memref to be inside a dfg.loop");

        // Mark dead ops
        SmallVector<Operation*> deadOps = {pullOp, pushOp, allocOp};
        // The view-like operations that works on the token memref are also dead
        collectViewChain(
            linebufOp.getTokenRef(),
            pullOp.getOperation(),
            deadOps);
        collectViewChain(
            pushOp.getTokenMemref(),
            allocOp.getOperation(),
            deadOps);

        // New alloc memref type
        auto newAllocType = MemRefType::get(
            axisShapeSizes(target.outputs[0]),
            allocOp.getType().getElementType());
        LAKSA_DEBUG(llvm::dbgs() << "  New alloc type: " << newAllocType);

        // Block arguments mapping
        IRMapping mapper;
        for (auto [oldArg, newArg] :
             llvm::zip(op.getBody().getArguments(), newBlock->getArguments()))
            mapper.map(oldArg, newArg);

        // Find where accumulate's "at" indices go out of scope on the *old*
        // tree -- that's where a fresh alloc must be (re)created and pushed
        // right after, so every output element gets its own zero-initialized
        // accumulator instead of all of them sharing one hoisted above the
        // whole loop nest.
        affine::AffineForOp pushLoop = nullptr;
        affine::AffineIfOp pushGuard = nullptr;
        {
            ValueRange atIndices = accumulateOp.getIndices();
            for (Operation* ancestor = accumulateOp->getParentOp();
                 ancestor && ancestor != streamingLoop.getOperation();
                 ancestor = ancestor->getParentOp()) {
                if (auto forOp = dyn_cast<affine::AffineForOp>(ancestor)) {
                    if (llvm::is_contained(
                            atIndices,
                            forOp.getInductionVar())) {
                        // Innermost at-indexed loop.
                        pushLoop = forOp;
                        break;
                    }
                    continue;
                }
                // If the buffer is scalar, "at" is empty, mark the if to be
                // the guard of the alloc/push position.
                if (auto ifOp = dyn_cast<affine::AffineIfOp>(ancestor))
                    pushGuard = ifOp;
            }
        }
        Operation* allocPushLandmark = pushLoop    ? pushLoop.getOperation()
                                       : pushGuard ? pushGuard.getOperation()
                                                   : nullptr;
        if (pushLoop)
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  alloc+push will live inside innermost at-indexed "
                   "affine.for at "
                << pushLoop.getLoc());
        else if (pushGuard)
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  alloc+push will live inside affine.if guard at "
                << pushGuard.getLoc());
        else
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  alloc+push will live at the end of the streaming "
                   "loop body");

        // Creates a fresh (shrunk) alloc and maps the old one to it. Called
        // once per landmark body below, so it re-executes -- and, thanks to
        // "fill_with", re-zeros -- every time that body does.
        auto emitFreshAlloc = [&](OpBuilder &b) {
            auto newAllocOp =
                memref::AllocOp::create(b, allocOp.getLoc(), newAllocType);
            for (NamedAttribute attr : allocOp->getAttrs())
                if (attr.getName() != "accu_at")
                    newAllocOp->setAttr(attr.getName(), attr.getValue());
            mapper.map(allocOp.getResult(), newAllocOp.getResult());
            LAKSA_DEBUG(llvm::dbgs() << "    New alloc created: " << newAllocOp);
        };
        auto emitPush = [&](OpBuilder &b) {
            PushMemRefOp::create(
                b,
                pushOp.getLoc(),
                mapper.lookup(allocOp.getResult()),
                newBlock->getArgument(1));
        };

        // The streamingLoop may be nested arbitrarily deep below dfg.loop,
        // and allocPushLandmark arbitrarily deep below streamingLoop, so
        // recurse through every affine.for/affine.if ancestor of either.
        LogicalResult bodyResult = success();
        auto isProperAncestor = [](Operation* candidate,
                                   Operation* descendant) {
            for (Operation* ancestor = descendant->getParentOp(); ancestor;
                 ancestor = ancestor->getParentOp())
                if (ancestor == candidate) return true;
            return false;
        };
        std::function<Operation*(Operation*, OpBuilder &)> cloneRebuildingPull =
            [&](Operation* oldOp, OpBuilder &builder) -> Operation* {
            if (oldOp == streamingLoop.getOperation()) {
                // The narrowed pull goes right before linebuf per iteration.
                SmallVector<Value> lbOperands =
                    mapOperands(mapper, streamingLoop.getLowerBoundOperands());
                SmallVector<Value> ubOperands =
                    mapOperands(mapper, streamingLoop.getUpperBoundOperands());
                auto newLoop = affine::AffineForOp::create(
                    builder,
                    streamingLoop.getLoc(),
                    lbOperands,
                    streamingLoop.getLowerBoundMap(),
                    ubOperands,
                    streamingLoop.getUpperBoundMap(),
                    streamingLoop.getStepAsInt(),
                    /*iterArgs=*/ValueRange{},
                    [&](OpBuilder &loopBuilder,
                        Location,
                        Value inductionVar,
                        ValueRange) {
                        mapper.map(
                            streamingLoop.getInductionVar(),
                            inductionVar);

                        Value tokenMemrefValue;
                        if (target.inputs[0].empty()) {
                            // If dfg.pull yields a scalar token, bridge with a
                            // 0-d memref by cast.
                            LAKSA_DEBUG(
                                llvm::dbgs()
                                << "    Scalar input port: using dfg.pull + "
                                   "cast to a 0-d memref before linebuf");
                            Value scalarToken = PullOp::create(
                                loopBuilder,
                                pullOp.getLoc(),
                                newBlock->getArgument(0));
                            auto wrapperType = MemRefType::get(
                                {},
                                cast<OutputType>(
                                    op.getFunctionType().getInput(0))
                                    .getElementType());
                            tokenMemrefValue =
                                UnrealizedConversionCastOp::create(
                                    loopBuilder,
                                    pullOp.getLoc(),
                                    wrapperType,
                                    scalarToken)
                                    .getResult(0);
                        } else {
                            LAKSA_DEBUG(
                                llvm::dbgs()
                                << "    Shaped input port: using "
                                   "dfg.pull_as_memref before linebuf");
                            auto newPullOp = PullAsMemRefOp::create(
                                loopBuilder,
                                pullOp.getLoc(),
                                newBlock->getArgument(0));
                            tokenMemrefValue = newPullOp.getResult();
                        }
                        mapper.map(linebufOp.getTokenRef(), tokenMemrefValue);

                        // With no dedicated at-indexed loop/guard below,
                        // alloc+push happen once per streaming-loop
                        // iteration instead.
                        bool allocPushHere = allocPushLandmark == nullptr;
                        if (allocPushHere) emitFreshAlloc(loopBuilder);
                        for (Operation &innerOp :
                             streamingLoop.getBody()->without_terminator())
                            cloneRebuildingPull(&innerOp, loopBuilder);
                        if (allocPushHere) emitPush(loopBuilder);
                        affine::AffineYieldOp::create(
                            loopBuilder,
                            streamingLoop.getLoc());
                    });
                return newLoop.getOperation();
            }

            if (allocPushLandmark && oldOp == allocPushLandmark) {
                if (auto forOp = dyn_cast<affine::AffineForOp>(oldOp)) {
                    SmallVector<Value> lbOperands =
                        mapOperands(mapper, forOp.getLowerBoundOperands());
                    SmallVector<Value> ubOperands =
                        mapOperands(mapper, forOp.getUpperBoundOperands());
                    auto newLoop = affine::AffineForOp::create(
                        builder,
                        forOp.getLoc(),
                        lbOperands,
                        forOp.getLowerBoundMap(),
                        ubOperands,
                        forOp.getUpperBoundMap(),
                        forOp.getStepAsInt(),
                        /*iterArgs=*/ValueRange{},
                        [&](OpBuilder &loopBuilder,
                            Location,
                            Value inductionVar,
                            ValueRange) {
                            mapper.map(forOp.getInductionVar(), inductionVar);
                            emitFreshAlloc(loopBuilder);
                            for (Operation &child :
                                 forOp.getBody()->without_terminator())
                                cloneRebuildingPull(&child, loopBuilder);
                            emitPush(loopBuilder);
                            affine::AffineYieldOp::create(
                                loopBuilder,
                                forOp.getLoc());
                        });
                    return newLoop.getOperation();
                }
                auto ifOp = cast<affine::AffineIfOp>(oldOp);
                if (ifOp.hasElse()) {
                    op.emitError(
                        "expected the alloc/push landmark's affine.if to "
                        "have no else region");
                    bodyResult = failure();
                    return nullptr;
                }
                auto newIf = affine::AffineIfOp::create(
                    builder,
                    ifOp.getLoc(),
                    ifOp.getIntegerSet(),
                    mapOperands(mapper, ifOp.getOperands()),
                    /*withElseRegion=*/false);
                OpBuilder thenBuilder = newIf.getThenBodyBuilder();
                emitFreshAlloc(thenBuilder);
                for (Operation &child :
                     ifOp.getThenBlock()->without_terminator())
                    cloneRebuildingPull(&child, thenBuilder);
                emitPush(thenBuilder);
                return newIf.getOperation();
            }

            bool isOnPathToLandmark =
                isProperAncestor(oldOp, streamingLoop.getOperation())
                || (allocPushLandmark
                    && isProperAncestor(oldOp, allocPushLandmark));
            if (!isOnPathToLandmark) return builder.clone(*oldOp, mapper);

            if (auto ifOp = dyn_cast<affine::AffineIfOp>(oldOp)) {
                if (ifOp.hasElse()) {
                    op.emitError(
                        "expected an affine.if ancestor on the way to the "
                        "alloc/push landmark to have no else region");
                    bodyResult = failure();
                    return nullptr;
                }
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Rebuilding ancestor affine.if at " << ifOp.getLoc()
                    << " on the way down to the alloc/push landmark");
                auto newIf = affine::AffineIfOp::create(
                    builder,
                    ifOp.getLoc(),
                    ifOp.getIntegerSet(),
                    mapOperands(mapper, ifOp.getOperands()),
                    /*withElseRegion=*/false);
                OpBuilder thenBuilder = newIf.getThenBodyBuilder();
                for (Operation &child :
                     ifOp.getThenBlock()->without_terminator())
                    cloneRebuildingPull(&child, thenBuilder);
                return newIf.getOperation();
            }

            auto forOp = dyn_cast<affine::AffineForOp>(oldOp);
            if (!forOp) {
                op.emitError(
                    "expected an affine.for or affine.if between linebuf's "
                    "loop and dfg.loop");
                bodyResult = failure();
                return nullptr;
            }
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    Rebuilding ancestor affine.for at " << forOp.getLoc()
                << " on the way down to the streaming loop");
            SmallVector<Value> lbOperands =
                mapOperands(mapper, forOp.getLowerBoundOperands());
            SmallVector<Value> ubOperands =
                mapOperands(mapper, forOp.getUpperBoundOperands());
            auto newLoop = affine::AffineForOp::create(
                builder,
                forOp.getLoc(),
                lbOperands,
                forOp.getLowerBoundMap(),
                ubOperands,
                forOp.getUpperBoundMap(),
                forOp.getStepAsInt(),
                /*iterArgs=*/ValueRange{},
                [&](OpBuilder &loopBuilder,
                    Location,
                    Value inductionVar,
                    ValueRange) {
                    mapper.map(forOp.getInductionVar(), inductionVar);
                    for (Operation &child :
                         forOp.getBody()->without_terminator())
                        cloneRebuildingPull(&child, loopBuilder);
                    affine::AffineYieldOp::create(loopBuilder, forOp.getLoc());
                });
            return newLoop.getOperation();
        };

        rewriter.setInsertionPointToStart(newBlock);
        for (Operation &oldTopOp : op.getBody().front()) {
            if (&oldTopOp != oldLoopOp.getOperation()) {
                rewriter.clone(oldTopOp, mapper);
                continue;
            }
            // Rebuild dfg.loop body.
            LoopOp::create(
                rewriter,
                oldLoopOp.getLoc(),
                newProcessOp.getInputPorts(),
                newProcessOp.getOutputPorts(),
                [&](OpBuilder &bodyBuilder, Location /*loc*/) {
                    LAKSA_DEBUG(llvm::dbgs() << "  Rebuilding dfg.loop body");
                    for (Operation &oldChild : oldLoopOp.getBody().front()) {
                        if (llvm::is_contained(deadOps, &oldChild)) continue;
                        cloneRebuildingPull(&oldChild, bodyBuilder);
                    }
                });
        }
        if (failed(bodyResult)) return failure();

        rewriter.replaceOp(op, newProcessOp);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Replace op with new process " << newProcessOp);
        newProcessOpOut = newProcessOp;
        return success();
    }
    LogicalResult rewriteParallelStyle(
        ProcessOp op,
        const ProcessPortShapes &target,
        PatternRewriter &rewriter,
        ProcessOp &newProcessOpOut) const
    {
        auto newFunctionType =
            buildNormalizedFunctionType(op, target, rewriter);
        LAKSA_DEBUG(llvm::dbgs() << "  The new signature is " << newFunctionType);

        auto newProcessOp = ProcessOp::create(
            rewriter,
            op.getLoc(),
            op.getSymName(),
            newFunctionType);
        shapes[newProcessOp] = target;
        Block* newBlock = &newProcessOp.getBody().front();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created new process \"" << newProcessOp.getSymName()
            << "\" with new signature");

        // Find one pull_as_memref per input port, and the affine.load
        // reading each one.
        struct InputPortInfo {
            PullAsMemRefOp pullOp = nullptr;
            affine::AffineLoadOp loadOp = nullptr;
            SmallVector<Value> loadOperands;
            SmallVector<int64_t> localAxes;
        };
        SmallVector<InputPortInfo> inputPorts(op.getNumInputPorts());
        op.walk([&](PullAsMemRefOp candidate) {
            for (unsigned i = 0; i < op.getNumInputPorts(); ++i)
                if (candidate.getReadPort() == op.getInputPort(i)) {
                    inputPorts[i].pullOp = candidate;
                    break;
                }
        });
        for (unsigned i = 0; i < op.getNumInputPorts(); ++i) {
            if (!inputPorts[i].pullOp)
                return op.emitError()
                       << "expected a pull_as_memref reading input port " << i;
            PullAsMemRefOp pullOp = inputPorts[i].pullOp;
            op.walk([&](affine::AffineLoadOp candidate) {
                if (findBackingDef(candidate.getMemRef())
                    != pullOp.getOperation())
                    return WalkResult::advance();
                inputPorts[i].loadOp = candidate;
                return WalkResult::interrupt();
            });
            if (!inputPorts[i].loadOp)
                return op.emitError()
                       << "expected an affine.load reading the memref "
                          "pulled from input port "
                       << i;
            if (!inputPorts[i].loadOp.getAffineMap().isIdentity())
                return op.emitError(
                    "expected an identity affine map on the affine.load "
                    "touching a pulled memref");
            inputPorts[i].loadOperands =
                llvm::to_vector(inputPorts[i].loadOp.getMapOperands());
        }
        LAKSA_DEBUG(
            llvm::dbgs() << "  Located " << op.getNumInputPorts()
                         << " pull_as_memref/affine.load pairs, one per "
                            "input port");

        // Find one push_memref per output port, all sharing the same
        // backing alloc, and the single affine.store writing it.
        struct OutputPortInfo {
            PushMemRefOp pushOp = nullptr;
            SmallVector<int64_t> localAxes;
        };
        SmallVector<OutputPortInfo> outputPorts(op.getNumOutputPorts());
        op.walk([&](PushMemRefOp candidate) {
            for (unsigned j = 0; j < op.getNumOutputPorts(); ++j)
                if (candidate.getWritePort() == op.getOutputPort(j)) {
                    outputPorts[j].pushOp = candidate;
                    break;
                }
        });
        for (unsigned j = 0; j < op.getNumOutputPorts(); ++j)
            if (!outputPorts[j].pushOp)
                return op.emitError()
                       << "expected a push_memref writing output port " << j;

        memref::AllocOp allocOp =
            findBackingAlloc(outputPorts[0].pushOp.getTokenMemref());
        if (!allocOp)
            return op.emitError(
                "expected the pushed memref to trace back to a memref.alloc");
        for (OutputPortInfo &outputPort : outputPorts)
            if (findBackingAlloc(outputPort.pushOp.getTokenMemref()) != allocOp)
                return op.emitError(
                    "expected every push in a parallel-style process to "
                    "flush the same output alloc");
        LAKSA_DEBUG(llvm::dbgs() << "  Backing alloc for push: " << allocOp);

        affine::AffineStoreOp storeOp = nullptr;
        op.walk([&](affine::AffineStoreOp candidate) {
            if (findBackingDef(candidate.getMemRef()) != allocOp.getOperation())
                return WalkResult::advance();
            storeOp = candidate;
            return WalkResult::interrupt();
        });
        if (!storeOp)
            return op.emitError(
                "expected an affine.store writing the pushed memref");
        if (!storeOp.getAffineMap().isIdentity())
            return op.emitError(
                "expected an identity affine map on the affine.store "
                "touching the pushed memref");
        auto storeOperands = llvm::to_vector(storeOp.getMapOperands());
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Located " << op.getNumOutputPorts()
            << " push_memref ops sharing affine.store at " << storeOp.getLoc());

        LoopOp oldLoopOp = inputPorts[0].pullOp->getParentOfType<LoopOp>();
        if (!oldLoopOp)
            return op.emitError(
                "expected pull_as_memref to be inside a dfg.loop");

        // The axis positions of target can be different from the load/store's
        // memref because of view-like operations that change shape. The nested
        // affine.for loops are copied over unchanged, with only the translation
        // of axis to the local one, which used in pull/push operation.
        for (unsigned i = 0; i < op.getNumInputPorts(); ++i) {
            InputPortInfo &inputPort = inputPorts[i];
            for (std::pair<int64_t, int64_t> axis : target.inputs[i]) {
                auto localAxis = findAxisTranslatingTo(
                    op,
                    inputPort.loadOp.getMemRef(),
                    inputPort.loadOperands.size(),
                    axis.first);
                if (failed(localAxis)) return failure();
                if (!*localAxis) {
                    op.emitError()
                        << "could not find input port " << i << "'s kept axis "
                        << axis.first
                        << " among the affine.load's own memref axes";
                    return failure();
                }
                inputPort.localAxes.push_back(**localAxis);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Input port " << i << " axis " << axis.first
                    << " (declared port type) maps to local axis "
                    << **localAxis << " of the affine.load");
            }
        }
        for (unsigned j = 0; j < op.getNumOutputPorts(); ++j) {
            OutputPortInfo &outputPort = outputPorts[j];
            for (std::pair<int64_t, int64_t> axis : target.outputs[j]) {
                auto translated = translateAxisBackwardThroughViews(
                    op,
                    outputPort.pushOp.getTokenMemref(),
                    axis.first);
                if (failed(translated)) return failure();
                outputPort.localAxes.push_back(*translated);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Output port " << j << " axis " << axis.first
                    << " (declared port type) maps to local axis "
                    << *translated << " of the affine.store");
            }
        }

        // Mark dead ops
        SmallVector<Operation*> deadOps = {allocOp};
        for (InputPortInfo &inputPort : inputPorts) {
            deadOps.push_back(inputPort.pullOp);
            collectViewChain(
                inputPort.loadOp.getMemRef(),
                inputPort.pullOp.getOperation(),
                deadOps);
        }
        collectViewChain(storeOp.getMemRef(), allocOp.getOperation(), deadOps);
        for (OutputPortInfo &outputPort : outputPorts) {
            deadOps.push_back(outputPort.pushOp);
            // The view-like reshapes between the pushed value and allocOp
            // (if any) are also dead, since Parallel style pushes scalars
            // directly at the store site instead of the whole reshaped
            // alloc.
            collectViewChain(
                outputPort.pushOp.getTokenMemref(),
                allocOp.getOperation(),
                deadOps);
        }

        // Block arguments mapping
        IRMapping mapper;
        for (auto [oldArg, newArg] :
             llvm::zip(op.getBody().getArguments(), newBlock->getArguments()))
            mapper.map(oldArg, newArg);

        // Recursively rebuilds the loop body.
        std::function<Operation*(Operation*, OpBuilder &)> rewriteLoopNest =
            [&](Operation* oldOp, OpBuilder &builder) -> Operation* {
            for (auto [i, inputPort] : llvm::enumerate(inputPorts)) {
                if (oldOp != inputPort.loadOp.getOperation()) continue;
                SmallVector<Value> indices;
                for (int64_t localAxis : inputPort.localAxes)
                    indices.push_back(mapper.lookupOrDefault(
                        inputPort.loadOperands[localAxis]));
                LAKSA_DEBUG(
                    llvm::dbgs() << "    Replacing affine.load for input port "
                                 << i << " with dfg.pull");
                auto newPull = PullOp::create(
                    builder,
                    inputPort.loadOp.getLoc(),
                    newBlock->getArgument(i),
                    indices);
                mapper.map(inputPort.loadOp.getResult(), newPull.getResult());
                return newPull.getOperation();
            }
            if (oldOp == storeOp.getOperation()) {
                Value valueToStore =
                    mapper.lookupOrDefault(storeOp.getValueToStore());
                Operation* lastPush = nullptr;
                for (auto [j, outputPort] : llvm::enumerate(outputPorts)) {
                    SmallVector<Value> indices;
                    for (int64_t localAxis : outputPort.localAxes)
                        indices.push_back(
                            mapper.lookupOrDefault(storeOperands[localAxis]));
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "    Replacing affine.store with dfg.push for "
                           "output port "
                        << j);
                    lastPush = PushOp::create(
                        builder,
                        storeOp.getLoc(),
                        valueToStore,
                        newBlock->getArgument(op.getNumInputPorts() + j),
                        indices);
                }
                return lastPush;
            }

            auto forOp = dyn_cast<affine::AffineForOp>(oldOp);
            if (!forOp) return builder.clone(*oldOp, mapper);

            SmallVector<Value> lbOperands =
                mapOperands(mapper, forOp.getLowerBoundOperands());
            SmallVector<Value> ubOperands =
                mapOperands(mapper, forOp.getUpperBoundOperands());

            auto newLoop = affine::AffineForOp::create(
                builder,
                forOp.getLoc(),
                lbOperands,
                forOp.getLowerBoundMap(),
                ubOperands,
                forOp.getUpperBoundMap(),
                forOp.getStepAsInt(),
                /*iterArgs=*/ValueRange{},
                [&](OpBuilder &loopBuilder,
                    Location,
                    Value inductionVar,
                    ValueRange) {
                    mapper.map(forOp.getInductionVar(), inductionVar);
                    for (Operation &child :
                         forOp.getBody()->without_terminator())
                        rewriteLoopNest(&child, loopBuilder);
                    affine::AffineYieldOp::create(loopBuilder, forOp.getLoc());
                });
            return newLoop.getOperation();
        };

        rewriter.setInsertionPointToStart(newBlock);
        for (Operation &oldTopOp : op.getBody().front()) {
            if (&oldTopOp != oldLoopOp.getOperation()) {
                rewriter.clone(oldTopOp, mapper);
                continue;
            }
            // Rebuild dfg.loop body.
            LoopOp::create(
                rewriter,
                oldLoopOp.getLoc(),
                newProcessOp.getInputPorts(),
                newProcessOp.getOutputPorts(),
                [&](OpBuilder &bodyBuilder, Location /*loc*/) {
                    LAKSA_DEBUG(llvm::dbgs() << "  Rebuilding dfg.loop body");
                    for (Operation &oldChild : oldLoopOp.getBody().front()) {
                        if (llvm::is_contained(deadOps, &oldChild)) continue;
                        rewriteLoopNest(&oldChild, bodyBuilder);
                    }
                });
        }

        rewriter.replaceOp(op, newProcessOp);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Replace op with new process" << newProcessOp);
        newProcessOpOut = newProcessOp;
        return success();
    }
    LogicalResult rewriteOtherStyle(
        ProcessOp op,
        const ProcessPortShapes &target,
        PatternRewriter &rewriter,
        ProcessOp &newProcessOpOut) const
    {
        auto newFunctionType =
            buildNormalizedFunctionType(op, target, rewriter);
        LAKSA_DEBUG(llvm::dbgs() << "  The new signature is " << newFunctionType);

        auto newProcessOp = ProcessOp::create(
            rewriter,
            op.getLoc(),
            op.getSymName(),
            newFunctionType);
        shapes[newProcessOp] = target;
        Block* newBlock = &newProcessOp.getBody().front();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created new process \"" << newProcessOp.getSymName()
            << "\" with new signature");

        // Find the ops to be rewritten in the original process.
        auto pullOp = findFirst<PullAsMemRefOp>(op.getOperation());
        auto pushOp = findFirst<PushMemRefOp>(op.getOperation());
        if (!pullOp || !pushOp)
            return op.emitError(
                "expected an other-style process to contain pull_as_memref "
                "and push_memref");
        LAKSA_DEBUG(
            llvm::dbgs() << "  Located pull_as_memref and push_memref "
                            "landmarks in the old body");

        LoopOp oldLoopOp = pullOp->getParentOfType<LoopOp>();
        if (!oldLoopOp)
            return op.emitError(
                "expected pull_as_memref to be inside a dfg.loop");

        // Block arguments mapping.
        IRMapping mapper;
        for (auto [oldArg, newArg] :
             llvm::zip(op.getBody().getArguments(), newBlock->getArguments()))
            mapper.map(oldArg, newArg);

        rewriter.setInsertionPointToStart(newBlock);
        for (Operation &oldTopOp : op.getBody().front()) {
            if (&oldTopOp != oldLoopOp.getOperation()) {
                rewriter.clone(oldTopOp, mapper);
                continue;
            }
            // Rebuild dfg.loop body: everything is cloned unchanged except
            // pull_as_memref/push_memref, which are re-pointed at the
            // normalized port and bridged to/from the original shape with
            // unrealized_conversion_cast.
            LoopOp::create(
                rewriter,
                oldLoopOp.getLoc(),
                newProcessOp.getInputPorts(),
                newProcessOp.getOutputPorts(),
                [&](OpBuilder &bodyBuilder, Location /*loc*/) {
                    LAKSA_DEBUG(llvm::dbgs() << "  Rebuilding dfg.loop body");
                    for (Operation &oldChild : oldLoopOp.getBody().front()) {
                        if (&oldChild == pullOp.getOperation()) {
                            Value bridged;
                            if (target.inputs[0].empty()) {
                                Value scalarToken = PullOp::create(
                                    bodyBuilder,
                                    pullOp.getLoc(),
                                    newBlock->getArgument(0));
                                // Bridge the scalar straight to the original
                                // (shaped) pulled type in one cast - no need
                                // to go through an intermediate 0-d memref.
                                bridged = UnrealizedConversionCastOp::create(
                                              bodyBuilder,
                                              pullOp.getLoc(),
                                              pullOp.getResult().getType(),
                                              scalarToken)
                                              .getResult(0);
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "    Scalar input port: pulled via "
                                       "dfg.pull and cast straight to the "
                                       "original pulled shape "
                                    << pullOp.getResult().getType());
                            } else {
                                Value normalizedToken =
                                    PullAsMemRefOp::create(
                                        bodyBuilder,
                                        pullOp.getLoc(),
                                        newBlock->getArgument(0))
                                        .getResult();
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "    Shaped input port: pulled via "
                                       "dfg.pull_as_memref from the "
                                       "normalized port");
                                bridged = UnrealizedConversionCastOp::create(
                                              bodyBuilder,
                                              pullOp.getLoc(),
                                              pullOp.getResult().getType(),
                                              normalizedToken)
                                              .getResult(0);
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "    Bridged the normalized pull "
                                       "back to the original pulled shape "
                                    << pullOp.getResult().getType());
                            }
                            mapper.map(pullOp.getResult(), bridged);
                            continue;
                        }
                        if (&oldChild == pushOp.getOperation()) {
                            Value origToken =
                                mapper.lookupOrDefault(pushOp.getTokenMemref());
                            if (target.outputs[0].empty()) {
                                auto elementType =
                                    cast<InputType>(
                                        newFunctionType.getResult(0))
                                        .getElementType();
                                // Bridge the original (shaped) pushed value
                                // straight to a scalar in one cast - no need
                                // to go through an intermediate 0-d memref
                                // and a real memref.load.
                                Value scalarVal =
                                    UnrealizedConversionCastOp::create(
                                        bodyBuilder,
                                        pushOp.getLoc(),
                                        elementType,
                                        origToken)
                                        .getResult(0);
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "    Scalar output port: cast "
                                       "straight to the element type and "
                                       "pushed via dfg.push");
                                PushOp::create(
                                    bodyBuilder,
                                    pushOp.getLoc(),
                                    scalarVal,
                                    newBlock->getArgument(1));
                            } else {
                                auto newOutputPortType = cast<InputType>(
                                    newFunctionType.getResult(0));
                                Value normalizedMemref =
                                    UnrealizedConversionCastOp::create(
                                        bodyBuilder,
                                        pushOp.getLoc(),
                                        MemRefType::get(
                                            newOutputPortType.getShape(),
                                            newOutputPortType.getElementType()),
                                        origToken)
                                        .getResult(0);
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "    Shaped output port: cast to "
                                       "the normalized shape and pushed "
                                       "via dfg.push_memref");
                                PushMemRefOp::create(
                                    bodyBuilder,
                                    pushOp.getLoc(),
                                    normalizedMemref,
                                    newBlock->getArgument(1));
                            }
                            continue;
                        }
                        bodyBuilder.clone(oldChild, mapper);
                    }
                });
        }

        rewriter.replaceOp(op, newProcessOp);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Replace op with new process " << newProcessOp);
        newProcessOpOut = newProcessOp;
        return success();
    }

    NormalizedShapeMap &shapes;
};
} // namespace

namespace {
// Normalizes an entire dfg.region atomically: every instantiated process, plus
// the region's boundary signature, channels, and instantiate wiring, all
// rewritten in one matchAndRewrite call.
struct NormalizeRegion : public OpRewritePattern<RegionOp> {
    NormalizeRegion(MLIRContext* context, NormalizedShapeMap &shapes)
            : OpRewritePattern<RegionOp>(context),
              shapes(shapes),
              bodyRewriter(shapes)
    {}

    LogicalResult
    matchAndRewrite(RegionOp op, PatternRewriter &rewriter) const override
    {
        LAKSA_DEBUG(
            llvm::dbgs() << "Visiting region \"" << op.getNodeName() << "\"");

        auto connectivityOrFailure = buildConnectivity(op);
        if (failed(connectivityOrFailure)) return failure();
        ConnectivityMap connectivity = std::move(*connectivityOrFailure);

        // Rewrite every process instantiated in this region.
        llvm::MapVector<ProcessOp, ProcessOp> newProcessFor;
        for (Operation* node : op.getGraphNodes()) {
            auto instantiateOp = dyn_cast<InstantiateOp>(node);
            if (!instantiateOp) continue;
            auto processOp = dyn_cast_or_null<ProcessOp>(
                instantiateOp.getInstantiatedOperation());
            if (!processOp || newProcessFor.contains(processOp)) continue;

            auto it = shapes.find(processOp);
            if (it == shapes.end())
                return processOp.emitError(
                    "expected a precomputed normalized shape");

            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Rewriting process \"" << processOp.getNodeName() << "\"");
            // Each rewritee helper leaves the insertion point inside the new
            // process' body, so reset it before every call here or the next
            // process would nest inside the previous one instead of landing as
            // a sibling of the region.
            rewriter.setInsertionPoint(op);
            ProcessOp newProcessOp;
            if (failed(bodyRewriter.rewriteProcess(
                    processOp,
                    it->second,
                    rewriter,
                    newProcessOp)))
                return failure();
            newProcessFor[processOp] = newProcessOp;
        }

        // Determine, for every old process' ports, where each input pulls from
        // and where each output pushes to, before rewriting the region's own
        // IR.
        struct PortSource {
            bool isBoundary = false;
            unsigned boundaryIndex = 0;
            ProcessOp producerProcess = nullptr;
            unsigned producerPort = 0;
        };
        // Boundary block args are numbered globally (inputs then outputs), so
        // offset an output boundary connection's portIndex back down to an
        // output-relative index.
        unsigned numBoundaryInputs = op.getNumInputPorts();
        DenseMap<ProcessOp, SmallVector<PortSource>> inputSources;
        DenseMap<ProcessOp, SmallVector<PortSource>> outputSinks;
        for (auto &[oldProcessOp, newProcessOp] : newProcessFor) {
            const ProcessNeighbors &neighbors =
                connectivity.lookup(oldProcessOp);
            SmallVector<PortSource> &ins = inputSources[oldProcessOp];
            for (PortConnection producer : neighbors.inputProducers) {
                PortSource source;
                if (producer.process) {
                    source.producerProcess = producer.process;
                    source.producerPort = producer.portIndex;
                } else {
                    source.isBoundary = true;
                    source.boundaryIndex = producer.portIndex;
                }
                ins.push_back(source);
            }
            SmallVector<PortSource> &outs = outputSinks[oldProcessOp];
            for (const SmallVector<PortConnection> &consumers :
                 neighbors.outputConsumers) {
                bool anyBoundary = false, anyProcess = false;
                PortSource sink;
                for (PortConnection consumer : consumers) {
                    if (consumer.process) {
                        anyProcess = true;
                    } else {
                        anyBoundary = true;
                        sink.isBoundary = true;
                        sink.boundaryIndex =
                            consumer.portIndex - numBoundaryInputs;
                    }
                }
                if (anyBoundary && anyProcess)
                    return op.emitError(
                        "expected an output port to be either fully "
                        "boundary-connected or fully process-connected, "
                        "not both");
                if (anyBoundary && consumers.size() != 1)
                    return op.emitError(
                        "expected a boundary-connected output port to have "
                        "exactly one consumer");
                outs.push_back(sink); // !isBoundary means "shared channel".
            }
        }

        // Determine the region's new boundary port types from whichever
        // (already-normalized) process port each boundary arg connects to.
        unsigned numBoundaryOutputs = op.getNumOutputPorts();
        SmallVector<Type> newBoundaryInputTypes(numBoundaryInputs);
        SmallVector<Type> newBoundaryOutputTypes(numBoundaryOutputs);
        SmallVector<bool> boundaryInputFound(numBoundaryInputs, false);
        SmallVector<bool> boundaryOutputFound(numBoundaryOutputs, false);
        for (auto &[oldProcessOp, newProcessOp] : newProcessFor) {
            for (auto [i, source] :
                 llvm::enumerate(inputSources[oldProcessOp])) {
                if (!source.isBoundary) continue;
                newBoundaryInputTypes[source.boundaryIndex] =
                    newProcessOp.getFunctionType().getInput(i);
                boundaryInputFound[source.boundaryIndex] = true;
            }
            for (auto [j, sink] : llvm::enumerate(outputSinks[oldProcessOp])) {
                if (!sink.isBoundary) continue;
                newBoundaryOutputTypes[sink.boundaryIndex] =
                    newProcessOp.getFunctionType().getResult(j);
                boundaryOutputFound[sink.boundaryIndex] = true;
            }
        }
        for (unsigned k = 0; k < numBoundaryInputs; ++k)
            if (!boundaryInputFound[k])
                return op.emitError()
                       << "boundary input " << k
                       << " is not connected to any normalized process";
        for (unsigned k = 0; k < numBoundaryOutputs; ++k)
            if (!boundaryOutputFound[k])
                return op.emitError()
                       << "boundary output " << k
                       << " is not connected to any normalized process";

        auto newRegionFunctionType = rewriter.getFunctionType(
            newBoundaryInputTypes,
            newBoundaryOutputTypes);
        LAKSA_DEBUG(
            llvm::dbgs() << "  The region's new boundary signature is "
                         << newRegionFunctionType);

        // Same reset as before each process rewrite above: the last
        // rewrite call left the insertion point inside that process' own body.
        rewriter.setInsertionPoint(op);
        auto newRegionOp = RegionOp::create(
            rewriter,
            op.getLoc(),
            op.getSymName(),
            newRegionFunctionType,
            [&](OpBuilder &bodyBuilder, Location, ValueRange newBoundaryArgs) {
                ValueRange newBoundaryInputs =
                    newBoundaryArgs.take_front(numBoundaryInputs);
                ValueRange newBoundaryOutputs =
                    newBoundaryArgs.drop_front(numBoundaryInputs);

                // One shared channel per (producer process, producer port)
                // that feeds another process - memoized so fan-out to
                // multiple consumers reuses the same channel.
                DenseMap<
                    std::pair<Operation*, unsigned>,
                    std::pair<Value, Value>>
                    channelHalves;
                auto getOrCreateChannel = [&](ProcessOp producerOld,
                                              unsigned producerPort) {
                    auto key = std::make_pair(
                        producerOld.getOperation(),
                        producerPort);
                    auto it = channelHalves.find(key);
                    if (it != channelHalves.end()) return it->second;
                    ProcessOp producerNew = newProcessFor.lookup(producerOld);
                    auto portType = cast<InputType>(
                        producerNew.getFunctionType().getResult(producerPort));
                    auto channelOp = ChannelOp::create(
                        bodyBuilder,
                        op.getLoc(),
                        portType.getShape(),
                        portType.getElementType(),
                        std::nullopt);
                    std::pair<Value, Value> halves(
                        channelOp.getInputPort(),
                        channelOp.getOutputPort());
                    channelHalves[key] = halves;
                    return halves;
                };

                for (auto &[oldProcessOp, newProcessOp] : newProcessFor) {
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Rebuilding wiring around \""
                                     << newProcessOp.getNodeName() << "\"");
                    SmallVector<Value> instInputs, instOutputs;
                    for (const PortSource &source :
                         inputSources[oldProcessOp]) {
                        if (source.isBoundary) {
                            instInputs.push_back(
                                newBoundaryInputs[source.boundaryIndex]);
                        } else {
                            instInputs.push_back(getOrCreateChannel(
                                                     source.producerProcess,
                                                     source.producerPort)
                                                     .second);
                        }
                    }
                    for (auto [j, sink] :
                         llvm::enumerate(outputSinks[oldProcessOp])) {
                        if (sink.isBoundary) {
                            instOutputs.push_back(
                                newBoundaryOutputs[sink.boundaryIndex]);
                        } else {
                            instOutputs.push_back(
                                getOrCreateChannel(oldProcessOp, j).first);
                        }
                    }
                    InstantiateOp::create(
                        bodyBuilder,
                        op.getLoc(),
                        newProcessOp,
                        instInputs,
                        instOutputs);
                }
            });

        rewriter.replaceOp(op, newRegionOp);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Replaced region \"" << op.getNodeName()
                         << "\" with a normalized region");
        return success();
    }

private:
    NormalizedShapeMap &shapes;
    ProcessBodyRewriter bodyRewriter;
};
} // namespace

namespace {
struct DFGIONormalizationPass
        : public dfg::impl::DFGIONormalizationBase<DFGIONormalizationPass> {
    void runOnOperation() override
    {
        auto moduleOp = cast<ModuleOp>(getOperation());
        auto shapesOrFailure = computeNormalizedShapes(moduleOp);
        if (failed(shapesOrFailure)) return signalPassFailure();
        NormalizedShapeMap shapes = std::move(*shapesOrFailure);

        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<NormalizeRegion>(&getContext(), shapes);

        target.addLegalDialect<DFGDialect>();
        // ProcessOp normalization is driven entirely as a side effect of
        // NormalizeRegion, so there is no pattern registered for ProcessOp
        target.addLegalOp<ProcessOp>();
        // A region is legal only when every dfg.instantiate's operand/result
        // shapes already match the referenced process' precomputed target
        // shape.
        target.addDynamicallyLegalOp<RegionOp>([&shapes](RegionOp op) {
            auto shapeMatches = [](ArrayRef<int64_t> actualShape,
                                   const AxisShape &axisShape) {
                if (actualShape.size() != axisShape.size()) return false;
                for (auto [size, axis] : llvm::zip(actualShape, axisShape))
                    if (size != axis.second) return false;
                return true;
            };
            for (Operation* node : op.getGraphNodes()) {
                auto instantiateOp = dyn_cast<InstantiateOp>(node);
                if (!instantiateOp) continue;
                auto processOp = dyn_cast_or_null<ProcessOp>(
                    instantiateOp.getInstantiatedOperation());
                if (!processOp) continue;
                auto it = shapes.find(processOp);
                if (it == shapes.end()) continue;
                const ProcessPortShapes &target = it->second;

                if (instantiateOp.getInputs().size() != target.inputs.size()
                    || instantiateOp.getOutputs().size()
                           != target.outputs.size()) {
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "'" << instantiateOp.getNodeName()
                        << "' has a different port count than '"
                        << processOp.getNodeName() << "'s target shape");
                    return false;
                }
                for (auto [operand, axisShape] :
                     llvm::zip(instantiateOp.getInputs(), target.inputs))
                    if (!shapeMatches(
                            cast<OutputType>(operand.getType()).getShape(),
                            axisShape)) {
                        LAKSA_DEBUG(
                            llvm::dbgs() << "'" << instantiateOp.getNodeName()
                                         << "' has a stale input type for '"
                                         << processOp.getNodeName() << "'");
                        return false;
                    }
                for (auto [result, axisShape] :
                     llvm::zip(instantiateOp.getOutputs(), target.outputs))
                    if (!shapeMatches(
                            cast<InputType>(result.getType()).getShape(),
                            axisShape)) {
                        LAKSA_DEBUG(
                            llvm::dbgs() << "'" << instantiateOp.getNodeName()
                                         << "' has a stale output type for '"
                                         << processOp.getNodeName() << "'");
                        return false;
                    }
            }
            return true;
        });
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(
                applyPartialConversion(moduleOp, target, std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createDFGIONormalizationPass()
{ return std::make_unique<DFGIONormalizationPass>(); }
