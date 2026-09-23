/// Implementation of the CollapseUnitDims transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "dfg-collapse-unit-dims"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[dfg-collapse-unit-dims] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::dfg;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGCOLLAPSEUNITDIMS
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

namespace {

// Old-axis indices that survive the collapse, in order.
using KeptAxes = SmallVector<int64_t>;
static KeptAxes computeKeptAxes(ArrayRef<int64_t> shape)
{
    KeptAxes kept;
    for (auto [axis, size] : llvm::enumerate(shape))
        if (size != 1) kept.push_back(static_cast<int64_t>(axis));
    return kept;
}
static void setIdentity(KeptAxes &kept, int64_t rank)
{
    kept.clear();
    kept.reserve(rank);
    for (int64_t axis = 0; axis < rank; ++axis) kept.push_back(axis);
}
static bool isIdentity(const KeptAxes &kept, int64_t rank)
{
    if (static_cast<int64_t>(kept.size()) != rank) return false;
    for (auto [axis, value] : llvm::enumerate(kept))
        if (static_cast<int64_t>(axis) != value) return false;
    return true;
}
// Picks the kept-axis entries , remapping each one through mapper
static SmallVector<Value>
mapAndSelect(IRMapping &mapper, ValueRange values, const KeptAxes &kept)
{
    SmallVector<Value> result;
    result.reserve(kept.size());
    for (int64_t axis : kept)
        result.push_back(mapper.lookupOrDefault(values[axis]));
    return result;
}
static SmallVector<int32_t>
selectInt32(ArrayRef<int32_t> values, const KeptAxes &kept)
{
    SmallVector<int32_t> result;
    result.reserve(kept.size());
    for (int64_t axis : kept) result.push_back(values[axis]);
    return result;
}
// Drops axes from a ShapedType (memref or emithls::ArrayType)
static Type dropAxes(Type type, const KeptAxes &kept)
{
    auto shaped = cast<ShapedType>(type);
    SmallVector<int64_t> newShape;
    newShape.reserve(kept.size());
    for (int64_t axis : kept) newShape.push_back(shaped.getShape()[axis]);
    // rebuild new memref
    if (auto memrefTy = dyn_cast<MemRefType>(type))
        return MemRefType::get(newShape, memrefTy.getElementType());
    return shaped.clone(newShape);
}
// Drops axes from a dfg::OutputType/InputType port, collapsing to a scalar port
// when every axis is dropped.
template<typename PortTy>
static PortTy dropAxesPort(PortTy port, const KeptAxes &kept)
{
    if (kept.empty()) return PortTy::get(port.getElementType());
    SmallVector<int64_t> newShape;
    newShape.reserve(kept.size());
    for (int64_t axis : kept) newShape.push_back(port.getShape()[axis]);
    return PortTy::get(newShape, port.getElementType());
}
static unsigned findPortIndex(ValueRange ports, Value v)
{
    for (auto [index, candidate] : llvm::enumerate(ports))
        if (candidate == v) return index;
    llvm_unreachable("port not found in its own port list");
}
static SmallVector<Value> mapValues(IRMapping &mapper, ValueRange values)
{
    return llvm::map_to_vector(values, [&](Value v) {
        return mapper.lookupOrDefault(v);
    });
}

// Per-process analysis
struct ProcessCollapsePlan {
    SmallVector<KeptAxes> inputKept;  // parallel to op.getInputPorts()
    SmallVector<KeptAxes> outputKept; // parallel to op.getOutputPorts()
    DenseMap<Operation*, KeptAxes> allocKept;
    bool anyDrop = false;
};
// Conservatively verifies that every op consuming a port-derived shaped value
// is one this pass knows how to rewrite.
static bool isSupportedProcessBody(ProcessOp op)
{
    DenseSet<Value> tainted;
    for (Value v : op.getInputPorts()) tainted.insert(v);
    for (Value v : op.getOutputPorts()) tainted.insert(v);

    bool ok = true;
    op.walk([&](Operation* o) {
        if (o == op.getOperation()) return WalkResult::advance();
        bool touchesTainted = llvm::any_of(o->getOperands(), [&](Value v) {
            return tainted.contains(v);
        });
        if (!touchesTainted) return WalkResult::advance();

        if (isa<LoopOp,
                emithls::ForOp,
                emithls::IfOp,
                emithls::ExpressionOp,
                emithls::YieldOp,
                PullOp,
                PushOp,
                PushMemRefOp,
                emithls::HelperAccumulateOp,
                emithls::ArrayReadOp,
                emithls::ArrayWriteOp>(o))
            return WalkResult::advance();

        if (auto pull = dyn_cast<PullAsMemRefOp>(o)) {
            tainted.insert(pull.getResult());
            return WalkResult::advance();
        }
        if (auto linebuf = dyn_cast<emithls::HelperLineBufferOp>(o)) {
            auto tokenRank =
                cast<ShapedType>(linebuf.getTokenRef().getType()).getRank();
            if (static_cast<int64_t>(linebuf.getNumChan().size())
                != tokenRank) {
                linebuf.emitRemark(
                    "dfg-collapse-unit-dims: num_chan does not cover the "
                    "full rank of its token operand; process left "
                    "unchanged");
                ok = false;
                return WalkResult::interrupt();
            }
            tainted.insert(linebuf.getBufRef());
            return WalkResult::advance();
        }
        if (auto window = dyn_cast<emithls::HelperWindowOp>(o)) {
            tainted.insert(window.getWinRef());
            return WalkResult::advance();
        }

        o->emitRemark(
            "dfg-collapse-unit-dims: unsupported op consumes a port-derived "
            "shaped value; process left unchanged");
        ok = false;
        return WalkResult::interrupt();
    });
    return ok;
}

static LogicalResult buildAllocKeptMap(
    ProcessOp op,
    ValueRange outputPorts,
    ArrayRef<KeptAxes> outputKept,
    DenseMap<Operation*, KeptAxes> &allocKept)
{
    LogicalResult result = success();
    op.walk([&](PushMemRefOp pushOp) {
        unsigned idx = findPortIndex(outputPorts, pushOp.getWritePort());
        int64_t rank = static_cast<int64_t>(
            cast<InputType>(pushOp.getWritePort().getType()).getShape().size());
        if (isIdentity(outputKept[idx], rank)) return WalkResult::advance();

        Value tokenMemref = pushOp.getTokenMemref();
        if (tokenMemref.getDefiningOp<PullAsMemRefOp>())
            return WalkResult::advance();

        auto allocOp = tokenMemref.getDefiningOp<memref::AllocOp>();
        if (!allocOp) {
            pushOp.emitRemark(
                "dfg-collapse-unit-dims: push_memref operand is not a "
                "directly-defined memref.alloc or dfg.pull_as_memref; process "
                "left unchanged");
            result = failure();
            return WalkResult::interrupt();
        }
        allocKept[allocOp.getOperation()] = outputKept[idx];
        return WalkResult::advance();
    });
    return result;
}

static FailureOr<ProcessCollapsePlan> analyzeProcess(ProcessOp op)
{
    LAKSA_DEBUG(
        llvm::dbgs() << "Analyzing process \"" << op.getNodeName() << "\"");

    ProcessCollapsePlan plan;
    FunctionType fnType = op.getFunctionType();

    for (Type t : fnType.getInputs())
        plan.inputKept.push_back(
            computeKeptAxes(cast<OutputType>(t).getShape()));
    for (Type t : fnType.getResults())
        plan.outputKept.push_back(
            computeKeptAxes(cast<InputType>(t).getShape()));

    SmallVector<Value> inputPorts = op.getInputPorts();
    SmallVector<Value> outputPorts = op.getOutputPorts();

    // A port ever pulled/pushed as a whole memref cannot collapse to a scalar
    op.walk([&](PullAsMemRefOp pull) {
        unsigned idx = findPortIndex(inputPorts, pull.getReadPort());
        if (plan.inputKept[idx].empty())
            setIdentity(
                plan.inputKept[idx],
                static_cast<int64_t>(
                    cast<OutputType>(fnType.getInput(idx)).getShape().size()));
    });
    op.walk([&](PushMemRefOp push) {
        unsigned idx = findPortIndex(outputPorts, push.getWritePort());
        if (plan.outputKept[idx].empty())
            setIdentity(
                plan.outputKept[idx],
                static_cast<int64_t>(
                    cast<InputType>(fnType.getResult(idx)).getShape().size()));
    });
    for (auto [i, kept] : llvm::enumerate(plan.inputKept))
        if (!isIdentity(
                kept,
                static_cast<int64_t>(
                    cast<OutputType>(fnType.getInput(i)).getShape().size())))
            plan.anyDrop = true;
    for (auto [i, kept] : llvm::enumerate(plan.outputKept))
        if (!isIdentity(
                kept,
                static_cast<int64_t>(
                    cast<InputType>(fnType.getResult(i)).getShape().size())))
            plan.anyDrop = true;

    // A linebuf's own num_chan can carry a trivial single-channel entry that
    // has nothing to do with any port shape.
    if (!plan.anyDrop) {
        op.walk([&](emithls::HelperLineBufferOp linebuf) {
            for (Attribute a : linebuf.getNumChan())
                if (cast<IntegerAttr>(a).getInt() == 1) plan.anyDrop = true;
        });
    }

    if (!plan.anyDrop) {
        LAKSA_DEBUG(
            llvm::dbgs() << "  no unit-size axes found, nothing to collapse");
        return plan;
    }

    if (!isSupportedProcessBody(op)) {
        LAKSA_DEBUG(
            llvm::dbgs() << "  body contains unsupported ops, skipping");
        return failure();
    }
    if (failed(buildAllocKeptMap(
            op,
            outputPorts,
            plan.outputKept,
            plan.allocKept))) {
        LAKSA_DEBUG(llvm::dbgs() << "  alloc bookkeeping incomplete, skipping");
        return failure();
    }

    LAKSA_DEBUG(llvm::dbgs() << "  plan accepted");
    return plan;
}

// Per-process body rewrite
class ProcessBodyRewriter {
public:
    explicit ProcessBodyRewriter(const ProcessCollapsePlan &plan) : plan(plan)
    {}

    ProcessOp rewriteProcess(ProcessOp op, IRRewriter &rewriter);

private:
    void rewriteBlock(Block &oldBlock, OpBuilder &b);
    void rewritePull(PullOp old, OpBuilder &b);
    void rewritePush(PushOp old, OpBuilder &b);
    void rewritePullAsMemRef(PullAsMemRefOp old, OpBuilder &b);
    void rewritePushMemRef(PushMemRefOp old, OpBuilder &b);
    void rewriteLinebuf(emithls::HelperLineBufferOp old, OpBuilder &b);
    void rewriteWindow(emithls::HelperWindowOp old, OpBuilder &b);
    void rewriteArrayRead(emithls::ArrayReadOp old, OpBuilder &b);
    void rewriteArrayWrite(emithls::ArrayWriteOp old, OpBuilder &b);
    void rewriteAccumulate(emithls::HelperAccumulateOp old, OpBuilder &b);
    void rewriteAlloc(memref::AllocOp old, OpBuilder &b);
    KeptAxes traceOrIdentity(Value v) const;

    const ProcessCollapsePlan &plan;
    IRMapping mapper;
    DenseMap<Value, KeptAxes> trace;
    SmallVector<Value> oldInputPorts;
    SmallVector<Value> oldOutputPorts;
};

KeptAxes ProcessBodyRewriter::traceOrIdentity(Value v) const
{
    auto it = trace.find(v);
    if (it != trace.end()) return it->second;
    KeptAxes identity;
    setIdentity(identity, cast<ShapedType>(v.getType()).getRank());
    return identity;
}

void ProcessBodyRewriter::rewritePull(PullOp old, OpBuilder &b)
{
    unsigned idx = findPortIndex(oldInputPorts, old.getReadPort());
    const KeptAxes &kept = plan.inputKept[idx];
    Value newPort = mapper.lookupOrDefault(old.getReadPort());
    SmallVector<Value> newIndices =
        mapAndSelect(mapper, old.getIndices(), kept);
    auto newOp = PullOp::create(b, old.getLoc(), newPort, newIndices);
    mapper.map(old.getResult(), newOp.getResult());
}

void ProcessBodyRewriter::rewritePush(PushOp old, OpBuilder &b)
{
    unsigned idx = findPortIndex(oldOutputPorts, old.getWritePort());
    const KeptAxes &kept = plan.outputKept[idx];
    Value newToken = mapper.lookupOrDefault(old.getToken());
    Value newPort = mapper.lookupOrDefault(old.getWritePort());
    SmallVector<Value> newIndices =
        mapAndSelect(mapper, old.getIndices(), kept);
    PushOp::create(b, old.getLoc(), newToken, newPort, newIndices);
}

void ProcessBodyRewriter::rewritePullAsMemRef(PullAsMemRefOp old, OpBuilder &b)
{
    unsigned idx = findPortIndex(oldInputPorts, old.getReadPort());
    const KeptAxes &kept = plan.inputKept[idx];
    Value newPort = mapper.lookupOrDefault(old.getReadPort());
    auto newOp = PullAsMemRefOp::create(b, old.getLoc(), newPort);
    mapper.map(old.getResult(), newOp.getResult());
    trace[old.getResult()] = kept;
}

void ProcessBodyRewriter::rewritePushMemRef(PushMemRefOp old, OpBuilder &b)
{
    Value newMemref = mapper.lookupOrDefault(old.getTokenMemref());
    Value newPort = mapper.lookupOrDefault(old.getWritePort());
    PushMemRefOp::create(b, old.getLoc(), newMemref, newPort);
}

void ProcessBodyRewriter::rewriteLinebuf(
    emithls::HelperLineBufferOp old,
    OpBuilder &b)
{
    Value oldTokenRef = old.getTokenRef();
    KeptAxes tokKept = traceOrIdentity(oldTokenRef);
    int64_t oldTokenRank = cast<ShapedType>(oldTokenRef.getType()).getRank();
    int64_t oldResultRank =
        cast<ShapedType>(old.getBufRef().getType()).getRank();

    SmallVector<int32_t> oldNumChan;
    for (Attribute a : old.getNumChan())
        oldNumChan.push_back(
            static_cast<int32_t>(cast<IntegerAttr>(a).getInt()));

    // A channel count of 1 is redundant on its own, whether or not the token
    // itself was already narrowed by port-shape tracing， drop it here too so
    // num_chan and the linebuf's own result shape never carry a dead
    // single-channel axis.
    KeptAxes chanKept;
    for (int64_t pos : tokKept)
        if (oldNumChan[pos] != 1) chanKept.push_back(pos);

    KeptAxes resultKept = chanKept;
    for (int64_t axis = oldTokenRank; axis < oldResultRank; ++axis)
        resultKept.push_back(axis);

    SmallVector<int32_t> newNumChan = selectInt32(oldNumChan, chanKept);

    Value newTokenRef = mapper.lookupOrDefault(oldTokenRef);
    Value newIndex =
        old.getIndex() ? mapper.lookupOrDefault(old.getIndex()) : Value();
    Type newResultTy = dropAxes(old.getBufRef().getType(), resultKept);

    auto newOp = emithls::HelperLineBufferOp::create(
        b,
        old.getLoc(),
        newTokenRef,
        newIndex,
        newResultTy,
        newNumChan,
        old.getNumLine());
    mapper.map(old.getBufRef(), newOp.getBufRef());
    trace[old.getBufRef()] = resultKept;
}

void ProcessBodyRewriter::rewriteWindow(
    emithls::HelperWindowOp old,
    OpBuilder &b)
{
    Value oldBufRef = old.getBufRef();
    KeptAxes bufKept = traceOrIdentity(oldBufRef);
    Value newBufRef = mapper.lookupOrDefault(oldBufRef);
    // Window indices are spatial offsets, not one-per-axis, so they never need
    // a dropped entry
    SmallVector<Value> newIndices = mapValues(mapper, old.getIndices());
    Type newWinTy = dropAxes(old.getWinRef().getType(), bufKept);

    auto newOp = emithls::HelperWindowOp::create(
        b,
        old.getLoc(),
        newBufRef,
        newIndices,
        newWinTy);
    mapper.map(old.getWinRef(), newOp.getWinRef());
    trace[old.getWinRef()] = bufKept;
}

void ProcessBodyRewriter::rewriteArrayRead(
    emithls::ArrayReadOp old,
    OpBuilder &b)
{
    Value oldArray = old.getArray();
    KeptAxes arrKept = traceOrIdentity(oldArray);
    int64_t rank = cast<ShapedType>(oldArray.getType()).getRank();
    if (isIdentity(arrKept, rank)) {
        b.clone(*old.getOperation(), mapper);
        return;
    }
    Value newArray = mapper.lookupOrDefault(oldArray);
    SmallVector<Value> newIndices =
        mapAndSelect(mapper, old.getIndices(), arrKept);
    auto newOp = emithls::ArrayReadOp::create(
        b,
        old.getLoc(),
        old.getResult().getType(),
        newArray,
        newIndices);
    mapper.map(old.getResult(), newOp.getResult());
}

void ProcessBodyRewriter::rewriteArrayWrite(
    emithls::ArrayWriteOp old,
    OpBuilder &b)
{
    Value oldArray = old.getArray();
    KeptAxes arrKept = traceOrIdentity(oldArray);
    int64_t rank = cast<ShapedType>(oldArray.getType()).getRank();
    if (isIdentity(arrKept, rank)) {
        b.clone(*old.getOperation(), mapper);
        return;
    }
    Value newValue = mapper.lookupOrDefault(old.getValue());
    Value newArray = mapper.lookupOrDefault(oldArray);
    SmallVector<Value> newIndices =
        mapAndSelect(mapper, old.getIndices(), arrKept);
    emithls::ArrayWriteOp::create(
        b,
        old.getLoc(),
        newValue,
        newArray,
        newIndices);
}

void ProcessBodyRewriter::rewriteAccumulate(
    emithls::HelperAccumulateOp old,
    OpBuilder &b)
{
    Value oldAccuRef = old.getAccuRef();
    KeptAxes accuKept = traceOrIdentity(oldAccuRef);
    int64_t rank = cast<ShapedType>(oldAccuRef.getType()).getRank();
    if (isIdentity(accuKept, rank)) {
        b.clone(*old.getOperation(), mapper);
        return;
    }
    Value newAccuRef = mapper.lookupOrDefault(oldAccuRef);
    Value newValue = mapper.lookupOrDefault(old.getValue());
    SmallVector<Value> newIndices =
        mapAndSelect(mapper, old.getIndices(), accuKept);
    emithls::HelperAccumulateOp::create(
        b,
        old.getLoc(),
        newAccuRef,
        newIndices,
        old.getOpCode(),
        newValue);
}

void ProcessBodyRewriter::rewriteAlloc(memref::AllocOp old, OpBuilder &b)
{
    auto it = plan.allocKept.find(old.getOperation());
    int64_t rank = old.getType().getRank();
    if (it == plan.allocKept.end() || isIdentity(it->second, rank)) {
        b.clone(*old.getOperation(), mapper);
        return;
    }
    Type newType = dropAxes(old.getType(), it->second);
    auto newOp =
        memref::AllocOp::create(b, old.getLoc(), cast<MemRefType>(newType));
    newOp->setAttrs(old->getAttrs());
    mapper.map(old.getResult(), newOp.getResult());
    trace[old.getResult()] = it->second;
}

void ProcessBodyRewriter::rewriteBlock(Block &oldBlock, OpBuilder &b)
{
    for (Operation &oldOp : oldBlock) {
        if (auto forOp = dyn_cast<emithls::ForOp>(&oldOp)) {
            emithls::ForOp::create(
                b,
                forOp.getLoc(),
                forOp.getLowerBoundAttr(),
                forOp.getUpperBoundAttr(),
                forOp.getStepAttr(),
                [&](OpBuilder &nb, Location, ValueRange ivs) {
                    mapper.map(forOp.getInductionVariable(), ivs.front());
                    rewriteBlock(forOp.getBody().front(), nb);
                });
        } else if (auto ifOp = dyn_cast<emithls::IfOp>(&oldOp)) {
            Value newCond = mapper.lookupOrDefault(ifOp.getCondition());
            bool hasElse = !ifOp.getElseRegion().empty();
            emithls::IfOp::create(
                b,
                ifOp.getLoc(),
                newCond,
                [&](OpBuilder &nb, Location) {
                    rewriteBlock(ifOp.getThenRegion().front(), nb);
                },
                hasElse
                    ? function_ref<void(OpBuilder &, Location)>(
                          [&](OpBuilder &nb, Location) {
                              rewriteBlock(ifOp.getElseRegion().front(), nb);
                          })
                    : function_ref<void(OpBuilder &, Location)>(nullptr));
        } else if (auto exprOp = dyn_cast<emithls::ExpressionOp>(&oldOp)) {
            auto newOp = emithls::ExpressionOp::create(
                b,
                exprOp.getLoc(),
                exprOp.getResult().getType(),
                [&](OpBuilder &nb, Location) {
                    rewriteBlock(exprOp.getBody().front(), nb);
                });
            mapper.map(exprOp.getResult(), newOp.getResult());
        } else if (auto pullOp = dyn_cast<PullOp>(&oldOp)) {
            rewritePull(pullOp, b);
        } else if (auto pushOp = dyn_cast<PushOp>(&oldOp)) {
            rewritePush(pushOp, b);
        } else if (auto pullMemrefOp = dyn_cast<PullAsMemRefOp>(&oldOp)) {
            rewritePullAsMemRef(pullMemrefOp, b);
        } else if (auto pushMemrefOp = dyn_cast<PushMemRefOp>(&oldOp)) {
            rewritePushMemRef(pushMemrefOp, b);
        } else if (
            auto linebufOp = dyn_cast<emithls::HelperLineBufferOp>(&oldOp)) {
            rewriteLinebuf(linebufOp, b);
        } else if (auto windowOp = dyn_cast<emithls::HelperWindowOp>(&oldOp)) {
            rewriteWindow(windowOp, b);
        } else if (auto accOp = dyn_cast<emithls::HelperAccumulateOp>(&oldOp)) {
            rewriteAccumulate(accOp, b);
        } else if (auto readOp = dyn_cast<emithls::ArrayReadOp>(&oldOp)) {
            rewriteArrayRead(readOp, b);
        } else if (auto writeOp = dyn_cast<emithls::ArrayWriteOp>(&oldOp)) {
            rewriteArrayWrite(writeOp, b);
        } else if (auto allocOp = dyn_cast<memref::AllocOp>(&oldOp)) {
            rewriteAlloc(allocOp, b);
        } else {
            b.clone(oldOp, mapper);
        }
    }
}

ProcessOp
ProcessBodyRewriter::rewriteProcess(ProcessOp op, IRRewriter &rewriter)
{
    LAKSA_DEBUG(
        llvm::dbgs()
        << "Collapsing unit dims in process \"" << op.getNodeName() << "\"");

    FunctionType fnType = op.getFunctionType();
    SmallVector<Type> newInputs, newOutputs;
    for (auto [i, ty] : llvm::enumerate(fnType.getInputs()))
        newInputs.push_back(
            dropAxesPort(cast<OutputType>(ty), plan.inputKept[i]));
    for (auto [i, ty] : llvm::enumerate(fnType.getResults()))
        newOutputs.push_back(
            dropAxesPort(cast<InputType>(ty), plan.outputKept[i]));

    auto newFnType = rewriter.getFunctionType(newInputs, newOutputs);
    LAKSA_DEBUG(llvm::dbgs() << "  new signature: " << newFnType);

    rewriter.setInsertionPoint(op);
    auto newProcessOp = ProcessOp::create(
        rewriter,
        op.getLoc(),
        op.getSymName(),
        newFnType,
        op.getMultiplicity());

    oldInputPorts = op.getInputPorts();
    oldOutputPorts = op.getOutputPorts();
    for (auto [oldArg, newArg] : llvm::zip(
             op.getBody().getArguments(),
             newProcessOp.getBody().getArguments()))
        mapper.map(oldArg, newArg);

    rewriter.setInsertionPointToStart(&newProcessOp.getBody().front());
    for (Operation &top : op.getBody().front()) {
        if (auto loopOp = dyn_cast<LoopOp>(&top)) {
            LoopOp::create(
                rewriter,
                loopOp.getLoc(),
                newProcessOp.getInputPorts(),
                newProcessOp.getOutputPorts(),
                [&](OpBuilder &nb, Location) {
                    rewriteBlock(loopOp.getBody().front(), nb);
                });
        } else {
            rewriter.clone(top, mapper);
        }
    }

    LAKSA_DEBUG(llvm::dbgs() << "  replaced process body");
    rewriter.replaceOp(op, newProcessOp);
    return newProcessOp;
}

// Region/channel wiring
//
// Topology and op order never change here, only per-port shapes shrink
// identically on both ends of a channel, so this walks the original body in
// place.
class RegionRewriter {
public:
    explicit RegionRewriter(RegionOp op) : op(op) {}

    LogicalResult rewrite(IRRewriter &rewriter);

private:
    bool needsRewrite();
    static FailureOr<InputType> resolveNewChannelType(ChannelOp channelOp);

    RegionOp op;
};

// A channel's new type is derived from its producer only - both ends of a
// channel always agree, since axis-dropping is a pure function of a shape
// and a channel's two ports necessarily start out with the same shape.
FailureOr<InputType> RegionRewriter::resolveNewChannelType(ChannelOp channelOp)
{
    auto producerInst =
        dyn_cast<InstantiateOp>(channelOp.getInputConnectedOp());
    if (!producerInst)
        return channelOp.emitError(
            "dfg-collapse-unit-dims: expected a channel to be connected to "
            "a dfg.instantiate");
    auto producerProcess =
        dyn_cast_or_null<ProcessOp>(producerInst.getInstantiatedOperation());
    if (!producerProcess)
        return producerInst.emitError(
            "dfg-collapse-unit-dims: expected a dfg.process to be "
            "instantiated");
    unsigned portIndex =
        findPortIndex(producerInst.getOutputs(), channelOp.getInputPort());
    return cast<InputType>(
        producerProcess.getFunctionType().getResult(portIndex));
}

// Check instantiate op's signature
bool RegionRewriter::needsRewrite()
{
    for (Operation* node : op.getGraphNodes()) {
        auto instantiateOp = dyn_cast<InstantiateOp>(node);
        if (!instantiateOp) continue;
        auto processOp = dyn_cast_or_null<ProcessOp>(
            instantiateOp.getInstantiatedOperation());
        if (!processOp) continue;
        FunctionType fnType = processOp.getFunctionType();
        for (auto [i, v] : llvm::enumerate(instantiateOp.getInputs()))
            if (v.getType() != fnType.getInput(i)) return true;
        for (auto [i, v] : llvm::enumerate(instantiateOp.getOutputs()))
            if (v.getType() != fnType.getResult(i)) return true;
    }
    return false;
}

LogicalResult RegionRewriter::rewrite(IRRewriter &rewriter)
{
    if (!needsRewrite()) {
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Region \"" << op.getNodeName() << "\" unchanged, skipping");
        return success();
    }

    LAKSA_DEBUG(
        llvm::dbgs() << "Rewiring region \"" << op.getNodeName()
                     << "\" for collapsed process port shapes");

    unsigned numBoundaryInputs = op.getNumInputPorts();
    unsigned numBoundaryOutputs = op.getNumOutputPorts();
    SmallVector<Type> newBoundaryInputTypes(numBoundaryInputs);
    SmallVector<Type> newBoundaryOutputTypes(numBoundaryOutputs);
    SmallVector<bool> boundaryInputFound(numBoundaryInputs, false);
    SmallVector<bool> boundaryOutputFound(numBoundaryOutputs, false);
    for (Operation* node : op.getGraphNodes()) {
        auto instantiateOp = dyn_cast<InstantiateOp>(node);
        if (!instantiateOp) continue;
        auto processOp = dyn_cast_or_null<ProcessOp>(
            instantiateOp.getInstantiatedOperation());
        if (!processOp) continue;
        FunctionType fnType = processOp.getFunctionType();
        for (auto [i, v] : llvm::enumerate(instantiateOp.getInputs())) {
            auto blockArg = dyn_cast<BlockArgument>(v);
            if (!blockArg) continue;
            newBoundaryInputTypes[blockArg.getArgNumber()] = fnType.getInput(i);
            boundaryInputFound[blockArg.getArgNumber()] = true;
        }
        for (auto [j, v] : llvm::enumerate(instantiateOp.getOutputs())) {
            auto blockArg = dyn_cast<BlockArgument>(v);
            if (!blockArg) continue;
            unsigned idx = blockArg.getArgNumber() - numBoundaryInputs;
            newBoundaryOutputTypes[idx] = fnType.getResult(j);
            boundaryOutputFound[idx] = true;
        }
    }
    for (unsigned k = 0; k < numBoundaryInputs; ++k)
        if (!boundaryInputFound[k])
            return op.emitError() << "dfg-collapse-unit-dims: boundary input "
                                  << k << " is not connected to any process";
    for (unsigned k = 0; k < numBoundaryOutputs; ++k)
        if (!boundaryOutputFound[k])
            return op.emitError() << "dfg-collapse-unit-dims: boundary output "
                                  << k << " is not connected to any process";

    auto newRegionFnType =
        rewriter.getFunctionType(newBoundaryInputTypes, newBoundaryOutputTypes);
    LAKSA_DEBUG(
        llvm::dbgs() << "  new boundary signature: " << newRegionFnType);

    rewriter.setInsertionPoint(op);
    LogicalResult bodyResult = success();
    auto newRegionOp = RegionOp::create(
        rewriter,
        op.getLoc(),
        op.getSymName(),
        newRegionFnType,
        [&](OpBuilder &b, Location, ValueRange newBoundaryArgs) {
            IRMapping mapper;
            for (auto [oldArg, newArg] :
                 llvm::zip(op.getBody().getArguments(), newBoundaryArgs))
                mapper.map(oldArg, newArg);

            for (Operation &oldOp : op.getBody().front()) {
                auto channelOp = dyn_cast<ChannelOp>(&oldOp);
                if (!channelOp) {
                    b.clone(oldOp, mapper);
                    continue;
                }
                auto newTypeOrFailure = resolveNewChannelType(channelOp);
                if (failed(newTypeOrFailure)) {
                    bodyResult = failure();
                    continue;
                }
                InputType newType = *newTypeOrFailure;
                auto newChannelOp = ChannelOp::create(
                    b,
                    oldOp.getLoc(),
                    newType.getShape(),
                    newType.getElementType(),
                    channelOp.getBufferSize());
                mapper.map(
                    channelOp.getInputPort(),
                    newChannelOp.getInputPort());
                mapper.map(
                    channelOp.getOutputPort(),
                    newChannelOp.getOutputPort());
            }
        });
    if (failed(bodyResult)) return failure();

    LAKSA_DEBUG(llvm::dbgs() << "  replaced region body");
    if (Attribute rootAttr = op->getAttr(laksa::kRootAttrName))
        newRegionOp->setAttr(laksa::kRootAttrName, rootAttr);
    rewriter.replaceOp(op, newRegionOp);
    return success();
}

struct DFGCollapseUnitDimsPass
        : public dfg::impl::DFGCollapseUnitDimsBase<DFGCollapseUnitDimsPass> {
    void runOnOperation() override
    {
        auto moduleOp = cast<ModuleOp>(getOperation());
        IRRewriter rewriter(&getContext());

        SmallVector<ProcessOp> processes;
        moduleOp.walk([&](ProcessOp op) { processes.push_back(op); });
        for (ProcessOp proc : processes) {
            auto planOrFailure = analyzeProcess(proc);
            if (failed(planOrFailure)) continue;
            if (!planOrFailure->anyDrop) continue;
            ProcessBodyRewriter(*planOrFailure).rewriteProcess(proc, rewriter);
        }

        SmallVector<RegionOp> regions;
        moduleOp.walk([&](RegionOp op) { regions.push_back(op); });
        for (RegionOp region : regions)
            if (failed(RegionRewriter(region).rewrite(rewriter)))
                return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::dfg::createDFGCollapseUnitDimsPass()
{ return std::make_unique<DFGCollapseUnitDimsPass>(); }
