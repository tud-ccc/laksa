/// Implementation of LoopFusion transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "emithls-loop-fusion"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-loop-fusion] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSLOOPFUSION
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

// A single random-access read or write of a memref/array-like resource
struct MemAccess {
    Value resource;
    SmallVector<Value> indices;
    bool isWrite;
    Operation* op;
};

// Collects every random-access memory access rooted at root (including nested
// regions such as emithls.if/emithls.for bodies).
static void
collectAccesses(Operation* root, SmallVectorImpl<MemAccess> &accesses)
{
    root->walk([&](Operation* op) {
        if (auto load = dyn_cast<memref::LoadOp>(op))
            accesses.push_back(
                {load.getMemRef(),
                 SmallVector<Value>(load.getIndices()),
                 false,
                 op});
        else if (auto store = dyn_cast<memref::StoreOp>(op))
            accesses.push_back(
                {store.getMemRef(),
                 SmallVector<Value>(store.getIndices()),
                 true,
                 op});
        else if (auto read = dyn_cast<ArrayReadOp>(op))
            accesses.push_back(
                {read.getArray(),
                 SmallVector<Value>(read.getIndices()),
                 false,
                 op});
        else if (auto write = dyn_cast<ArrayWriteOp>(op))
            accesses.push_back(
                {write.getArray(),
                 SmallVector<Value>(write.getIndices()),
                 true,
                 op});
        else if (auto update = dyn_cast<UpdateOp>(op)) {
            accesses.push_back(
                {update.getVariable(),
                 SmallVector<Value>(update.getIndices()),
                 true,
                 op});
            if (!update.getNewValueIndices().empty())
                accesses.push_back(
                    {update.getNewValue(),
                     SmallVector<Value>(update.getNewValueIndices()),
                     false,
                     op});
        }
    });
}

// Whether index lists "a" and "b" are the same once "from" is treated as an
// alias for "to", which is used to compare accesses across the two loop bodies
// after identifying the fused-away induction variable with the surviving one.
static bool sameIndices(ValueRange a, ValueRange b, Value from, Value to)
{
    if (a.size() != b.size()) return false;
    for (auto [x, y] : llvm::zip(a, b))
        if ((x == from ? to : x) != (y == from ? to : y)) return false;
    return true;
}

// Fusing "first" and "second" one iteration at a time is only equivalent to
// running "first" to completion before "second" if every resource they both
// touch, with at least one side writing it, is accessed at corresponding
// indices in both loops.
static bool conflicts(ForOp first, ForOp second)
{
    SmallVector<MemAccess> firstAccesses, secondAccesses;
    collectAccesses(first, firstAccesses);
    collectAccesses(second, secondAccesses);

    Value firstIV = first.getInductionVariable();
    Value secondIV = second.getInductionVariable();

    for (auto &a : firstAccesses) {
        for (auto &b : secondAccesses) {
            if (a.resource != b.resource) continue;
            if (!a.isWrite && !b.isWrite) continue;
            if (!sameIndices(a.indices, b.indices, secondIV, firstIV)) {
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "  Conflict on " << a.resource << " between "
                    << a.op->getName() << " and " << b.op->getName());
                return true;
            }
        }
    }
    LAKSA_DEBUG(llvm::dbgs() << "  No conflicts between the two loop bodies");
    return false;
}

// After fusing, a memref store whose value is immediately read back by a load
// at the exact same address is a redundant round trip through memory. Sp it
// forwards the stored value directly to the load's uses and erase the load.
static void forwardStoresToLoads(Block &body, PatternRewriter &rewriter)
{
    LAKSA_DEBUG(llvm::dbgs() << "  Forwarding stores to loads in fused body");
    struct PendingStore {
        Value memref;
        SmallVector<Value> indices;
        Value value;
    };
    SmallVector<PendingStore> pendingStores;
    SmallVector<memref::LoadOp> deadLoads;

    body.walk([&](Operation* op) {
        if (auto store = dyn_cast<memref::StoreOp>(op)) {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    Recording pending store of " << store.getValue()
                << " to " << store.getMemRef());
            pendingStores.push_back(
                {store.getMemRef(),
                 SmallVector<Value>(store.getIndices()),
                 store.getValue()});
        } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
            SmallVector<Value> loadIndices(load.getIndices());
            for (auto &pending : llvm::reverse(pendingStores)) {
                if (pending.memref != load.getMemRef()
                    || pending.indices != loadIndices)
                    continue;
                LAKSA_DEBUG(
                    llvm::dbgs() << "    Forwarding " << pending.value
                                 << " to load " << load.getResult());
                rewriter.replaceAllUsesWith(load.getResult(), pending.value);
                deadLoads.push_back(load);
                break;
            }
        }
    });
    for (auto load : deadLoads) rewriter.eraseOp(load);
    LAKSA_DEBUG(
        llvm::dbgs()
        << "    Forwarded and erased " << deadLoads.size() << " load(s)");

    DenseSet<Operation*> handledAllocs;
    for (auto &pending : pendingStores) {
        auto alloc = pending.memref.getDefiningOp<memref::AllocOp>();
        if (!alloc || !handledAllocs.insert(alloc).second) continue;
        bool stillRead = llvm::any_of(
            alloc.getResult().getUsers(),
            [](Operation* user) { return isa<memref::LoadOp>(user); });
        if (stillRead) continue;
        LAKSA_DEBUG(
            llvm::dbgs() << "    Removing dead alloc " << alloc.getResult());
        for (Operation* user :
             llvm::make_early_inc_range(alloc.getResult().getUsers()))
            rewriter.eraseOp(user);
        rewriter.eraseOp(alloc);
    }
}

struct FuseAdjacentLoops : OpRewritePattern<ForOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ForOp first, PatternRewriter &rewriter) const override
    {
        auto second = dyn_cast_or_null<ForOp>(first->getNextNode());
        if (!second) return failure();
        if (first.getLowerBound() != second.getLowerBound()
            || first.getUpperBound() != second.getUpperBound()
            || first.getStep() != second.getStep())
            return failure();

        LAKSA_DEBUG(
            llvm::dbgs() << "Considering fusing loop at " << first.getLoc()
                         << " with loop at " << second.getLoc());
        if (conflicts(first, second))
            return rewriter.notifyMatchFailure(
                first,
                "loop bodies conflict, cannot fuse");

        Value firstIV = first.getInductionVariable();
        rewriter.mergeBlocks(
            &second.getBody().front(),
            &first.getBody().front(),
            firstIV);
        rewriter.eraseOp(second);
        LAKSA_DEBUG(llvm::dbgs() << "  Fused loop bodies");

        forwardStoresToLoads(first.getBody().front(), rewriter);

        return success();
    }
};

} // namespace

namespace {
struct EmitHLSLoopFusionPass
        : public emithls::impl::EmitHLSLoopFusionBase<EmitHLSLoopFusionPass> {
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<FuseAdjacentLoops>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::emithls::createEmitHLSLoopFusionPass()
{ return std::make_unique<EmitHLSLoopFusionPass>(); }
