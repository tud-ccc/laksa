/// Implementation of FuncOutlineComputationLeaf transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/Func/Transforms/Passes.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>

#define DEBUG_TYPE "func-outline-computation-leaf"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[func-outline-computation-leaf] "; X;                 \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::func;

namespace mlir {
namespace func {
#define GEN_PASS_DEF_FUNCOUTLINECOMPUTATIONLEAF
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h.inc"
} // namespace func
} // namespace mlir

namespace {
struct FuncOutlineComputationLeafPass
        : public func::impl::FuncOutlineComputationLeafBase<
              FuncOutlineComputationLeafPass> {

    SmallVector<FuncOp> nodeFunctions;
    SmallPtrSet<Operation*, 8> storedGenericOps;
    SmallPtrSet<Operation*, 8> consumedReshapeOps;
    // Ops that have been outlined as their own node in previous outlineNode
    // calls.  Their results must be treated as inter-function values rather
    // than dep ops whenever they appear as operands of a later node.
    SmallPtrSet<Operation*, 8> outlinedOps;
    // Parallel to nodeFunctions: the original-scope values each node takes
    // as arguments, and the original-scope values it returns.
    SmallVector<SmallVector<Value>> nodeArgValues;
    SmallVector<SmallVector<Value>> nodeEffectiveResults;

    // Traces operands of a node op to collect function argument types/values
    // and dependency ops, then creates an outlined FuncOp containing clones
    // of all collected ops with a correct return.
    LogicalResult outlineNode(
        IRRewriter &rewriter,
        Operation* nodeOp,
        ValueRange operandsToTrace)
    {
        SmallVector<Value> funcArgValues;
        SmallVector<Value> effectiveResults;
        SmallVector<Operation*> nodeFunctionOps;
        DenseSet<Value> seenArgValues;
        DenseSet<Operation*> addedToNode;

        // Y-combinator avoids the overhead of std::function type-erasure.
        auto collectOperandInfo = [&](auto &self, Value val) -> void {
            if (auto blkArg = dyn_cast<BlockArgument>(val)) {
                if (!seenArgValues.insert(val).second) return;
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "  Block arg #" << blkArg.getArgNumber() << " ("
                    << blkArg.getType() << ") -> function argument");
                funcArgValues.push_back(val);
            } else {
                auto* defOp = val.getDefiningOp();
                if (auto defGenericOp = dyn_cast<linalg::GenericOp>(defOp)) {
                    if (storedGenericOps.contains(defGenericOp)) {
                        if (!addedToNode.insert(defOp).second) return;
                        LAKSA_DEBUG(
                            llvm::dbgs() << "  Consuming stored generic at "
                                         << defGenericOp.getLoc());
                        // Do NOT erase from storedGenericOps: a fill/broadcast
                        // may be shared by multiple computation nodes, each of
                        // which gets its own independent clone.
                        for (auto input : defGenericOp.getInputs())
                            self(self, input);
                        for (auto output : defGenericOp.getOutputs())
                            self(self, output);
                        nodeFunctionOps.push_back(defOp);
                    } else {
                        if (!seenArgValues.insert(val).second) return;
                        LAKSA_DEBUG(
                            llvm::dbgs()
                            << "  Inter-function result (" << val.getType()
                            << ") from generic at " << defGenericOp.getLoc()
                            << " -> function argument");
                        funcArgValues.push_back(val);
                    }
                } else {
                    // An op already outlined as its own node (e.g. tensor.pad)
                    // must not be re-cloned; pass its result as an argument.
                    if (outlinedOps.contains(defOp)) {
                        if (!seenArgValues.insert(val).second) return;
                        LAKSA_DEBUG(
                            llvm::dbgs()
                            << "  Already-outlined op result (" << val.getType()
                            << ") from " << defOp->getName() << " at "
                            << defOp->getLoc() << " -> function argument");
                        funcArgValues.push_back(val);
                        return;
                    }
                    // A reshape op whose result was already returned by a
                    // previous outlined node must not be re-cloned; treat
                    // its result as an inter-function argument instead.
                    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
                            defOp)
                        && consumedReshapeOps.contains(defOp)) {
                        if (!seenArgValues.insert(val).second) return;
                        LAKSA_DEBUG(
                            llvm::dbgs()
                            << "  Consumed reshape result (" << val.getType()
                            << ") -> function argument");
                        funcArgValues.push_back(val);
                        return;
                    }
                    if (!addedToNode.insert(defOp).second) return;
                    for (Value operand : defOp->getOperands())
                        self(self, operand);
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Adding dep op " << defOp->getName()
                                     << " at " << defOp->getLoc());
                    nodeFunctionOps.push_back(defOp);
                }
            }
        };

        for (Value operand : operandsToTrace)
            collectOperandInfo(collectOperandInfo, operand);
        // Keep addedToNode in sync: nodeOp is pushed outside the lambda.
        nodeFunctionOps.push_back(nodeOp);
        addedToNode.insert(nodeOp);

        // Fold a chain of trailing reshapes into the node as long as each is
        // the sole consumer of the previous value (e.g. a collapse_shape
        // immediately followed by an expand_shape).
        for (auto result : nodeOp->getResults()) {
            if (result.use_empty()) continue;
            Value current = result;
            while (current.hasOneUse()) {
                auto* user = *current.user_begin();
                if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user))
                    break;
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "  Result consumed by " << user->getName() << " at "
                    << user->getLoc() << " -> folding reshape into node");
                nodeFunctionOps.push_back(user);
                addedToNode.insert(user);
                consumedReshapeOps.insert(user);
                current = user->getResult(0);
            }
            effectiveResults.push_back(current);
        }

        // Find values used inside op regions but defined outside the ops
        // scheduled for cloning (constants captured directly by a body region
        // rather than passed through the op's operands).
        // addedToNode already holds every op in nodeFunctionOps; reuse it as
        // the "in-node" guard instead of rebuilding a separate set.
        // For each def, walk parent ops up until we either hit an op in
        // addedToNode (def is nested inside a node op — not a capture) or
        // exhaust the chain (def is truly external).
        {
            SmallVector<Operation*> captured;
            for (auto* opi : nodeFunctionOps)
                for (auto &region : opi->getRegions())
                    region.walk([&](Operation* inner) {
                        for (Value operand : inner->getOperands()) {
                            auto* def = operand.getDefiningOp();
                            if (!def) continue;
                            auto* ancestor = def->getParentOp();
                            while (ancestor && !addedToNode.contains(ancestor))
                                ancestor = ancestor->getParentOp();
                            if (ancestor) continue; // nested inside a node op
                            if (!addedToNode.insert(def).second) continue;
                            LAKSA_DEBUG(
                                llvm::dbgs()
                                << "  Captured external op " << def->getName()
                                << " at " << def->getLoc());
                            captured.push_back(def);
                        }
                    });

            if (!captured.empty()) {
                LAKSA_DEBUG(
                    llvm::dbgs() << "Prepending " << captured.size()
                                 << " captured op(s) to node op list");
                SmallVector<Operation*> reordered;
                reordered.reserve(captured.size() + nodeFunctionOps.size());
                llvm::append_range(reordered, captured);
                llvm::append_range(reordered, nodeFunctionOps);
                nodeFunctionOps = std::move(reordered);
            }
        }

        auto parentFuncOp = cast<FuncOp>(nodeOp->getParentOp());
        auto funcType = rewriter.getFunctionType(
            TypeRange(ValueRange(funcArgValues)),
            TypeRange(ValueRange(effectiveResults)));
        std::string funcName = parentFuncOp.getSymName().str() + "_node_"
                               + std::to_string(nodeFunctions.size());
        LAKSA_DEBUG(
            llvm::dbgs() << "Creating node function '" << funcName
                         << "' with type " << funcType);

        rewriter.setInsertionPoint(parentFuncOp);
        auto newFuncOp =
            FuncOp::create(rewriter, nodeOp->getLoc(), funcName, funcType);
        Block* block = newFuncOp.addEntryBlock();

        IRMapping mapping;
        for (auto [origVal, newArg] :
             llvm::zip(funcArgValues, block->getArguments()))
            mapping.map(origVal, newArg);

        LAKSA_DEBUG(
            llvm::dbgs() << "Cloning " << nodeFunctionOps.size()
                         << " ops into '" << funcName << "':");
        rewriter.setInsertionPointToEnd(block);
        for (auto* opi : nodeFunctionOps) {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  " << opi->getName() << " at " << opi->getLoc());
            rewriter.clone(*opi, mapping);
        }

        SmallVector<Value> returnVals;
        for (auto val : effectiveResults)
            returnVals.push_back(mapping.lookupOrDefault(val));
        ReturnOp::create(rewriter, nodeOp->getLoc(), returnVals);
        LAKSA_DEBUG(
            llvm::dbgs() << "Return " << returnVals.size() << " value(s) from '"
                         << funcName << "'");

        nodeFunctions.push_back(newFuncOp);
        nodeArgValues.push_back(std::move(funcArgValues));
        nodeEffectiveResults.push_back(std::move(effectiveResults));
        outlinedOps.insert(nodeOp);
        return success();
    }

    LogicalResult outlineComputationOp(IRRewriter &rewriter, Operation* op)
    {
        LAKSA_DEBUG(
            llvm::dbgs() << "Processing operation " << op->getName() << " at "
                         << op->getLoc());

        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            if (linalg::isaFillOpInterface(genericOp)
                || linalg::isaBroadcastOpInterface(genericOp)) {
                storedGenericOps.insert(genericOp);
                LAKSA_DEBUG(
                    llvm::dbgs() << "Store fill/broadcast for later use.");
            } else {
                LAKSA_DEBUG(
                    llvm::dbgs() << "Outlining computation generic at "
                                 << genericOp.getLoc());
                SmallVector<Value> operands;
                llvm::append_range(operands, genericOp.getInputs());
                llvm::append_range(operands, genericOp.getOutputs());
                return outlineNode(rewriter, op, operands);
            }
        } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
            LAKSA_DEBUG(
                llvm::dbgs() << "Outlining pad op at " << padOp.getLoc());
            SmallVector<Value> operands;
            operands.push_back(padOp.getSource());
            llvm::append_range(operands, padOp.getLow());
            llvm::append_range(operands, padOp.getHigh());
            return outlineNode(rewriter, op, operands);
        } else {
            LAKSA_DEBUG(llvm::dbgs() << "Skip " << op->getName());
        }

        return success();
    }

    void runOnOperation() override
    {
        nodeFunctions.clear();
        storedGenericOps.clear();
        consumedReshapeOps.clear();
        outlinedOps.clear();
        nodeArgValues.clear();
        nodeEffectiveResults.clear();

        func::FuncOp funcOp = getOperation();
        IRRewriter rewriter(funcOp->getContext());

        for (auto &opFunc : funcOp.getBody().getOps())
            if (failed(outlineComputationOp(rewriter, &opFunc)))
                return signalPassFailure();

        if (nodeFunctions.empty()) return;

        // Rebuild funcOp's body as a wrapper that calls the outlined nodes.
        Block &funcBlock = funcOp.getBody().front();

        // Snapshot the old ops; the terminator is always last.
        SmallVector<Operation*> oldOps;
        for (auto &op : funcBlock.getOperations()) oldOps.push_back(&op);

        // Block arguments map to themselves (they survive the body rewrite).
        DenseMap<Value, Value> valueMap;
        for (Value blkArg : funcBlock.getArguments()) valueMap[blkArg] = blkArg;

        // Insert calls to each node function, just before the old terminator.
        rewriter.setInsertionPoint(oldOps.back());
        for (auto [nodeFunc, argVals, effResults] :
             llvm::zip(nodeFunctions, nodeArgValues, nodeEffectiveResults)) {
            SmallVector<Value> callArgs;
            for (Value origVal : argVals) {
                auto it = valueMap.find(origVal);
                if (it == valueMap.end()) {
                    funcOp.emitError("outlined node '")
                        << nodeFunc.getSymName()
                        << "': argument value missing from caller value map";
                    return signalPassFailure();
                }
                callArgs.push_back(it->second);
            }
            LAKSA_DEBUG(
                llvm::dbgs()
                << "Building call to '" << nodeFunc.getSymName() << "'");
            auto callOp =
                CallOp::create(rewriter, funcOp.getLoc(), nodeFunc, callArgs);
            // Map each original effective result to the corresponding call
            // result.
            for (auto [origResult, callResult] :
                 llvm::zip(effResults, callOp.getResults()))
                valueMap[origResult] = callResult;
        }

        // Replace old return with remapped values.
        auto* oldReturn = oldOps.back();
        SmallVector<Value> newReturnVals;
        for (Value origVal : oldReturn->getOperands()) {
            auto it = valueMap.find(origVal);
            if (it == valueMap.end()) {
                funcOp.emitError(
                    "wrapper return: value missing from caller value map");
                return signalPassFailure();
            }
            newReturnVals.push_back(it->second);
        }
        LAKSA_DEBUG(
            llvm::dbgs() << "Building wrapper return with "
                         << newReturnVals.size() << " value(s)");
        rewriter.replaceOpWithNewOp<ReturnOp>(oldReturn, newReturnVals);
        oldOps.pop_back();

        // Erase the now-dead original ops in reverse topological order so
        // each op is def-free by the time it is erased.
        for (auto* op : llvm::reverse(oldOps)) rewriter.eraseOp(op);

        funcOp.setSymName(funcOp.getSymName().str() + "_top");
        funcOp->setAttr(
            laksa::kRootAttrName,
            UnitAttr::get(funcOp.getContext()));

        LAKSA_DEBUG(
            llvm::dbgs() << "Wrapper '" << funcOp.getSymName() << "' complete");
    }
};

} // namespace

std::unique_ptr<Pass> mlir::func::createFuncOutlineComputationLeafPass()
{ return std::make_unique<FuncOutlineComputationLeafPass>(); }
