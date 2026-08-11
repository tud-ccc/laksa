// Implementation of PragmaInsertion transform pass.
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <optional>

#define DEBUG_TYPE "emithls-pragma-insertion"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-pragma-insertion] "; X;                      \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSPRAGMAINSERTION
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

// True if "op" carries any "dse."-prefixed attribute PragmaDSE attached,
// the signal that it still needs a real pragma inserted in its place.
bool hasDSEAttr(Operation* op)
{
    return llvm::any_of(op->getAttrs(), [](NamedAttribute attr) {
        return attr.getName().getValue().starts_with("dse.");
    });
}

// A loop marked "dse.io" with no "dse.factor" is just an outer word loop,
// so the marker is dropped in place. One with "dse.factor" is a channel
// loop whose trip count still reflects the port's old width: a
// "factor"-wide loop replaces it in place, and the leftover
// "tripCount / factor" multiplies into the parent loop's own trip count
// instead of wrapping it, so the parent's index keeps addressing correctly.
struct SplitIOLoop : public OpConversionPattern<ForOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ForOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (!op->hasAttr("dse.io")) return failure();
        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Rewriting loop inside io function at " << loc);

        auto factorAttr = op->getAttrOfType<IntegerAttr>("dse.factor");
        if (!factorAttr) {
            LAKSA_DEBUG(
                llvm::dbgs() << "  Dropping bare dse.io tag at " << loc);
            rewriter.modifyOpInPlace(op, [&]() { op->removeAttr("dse.io"); });
            return success();
        }

        int64_t factor = factorAttr.getInt();
        int64_t tripCount = op.getTripCount();
        int64_t outerTrip = tripCount / factor;
        LAKSA_DEBUG(
            llvm::dbgs() << "  Splitting io loop at " << loc
                         << " (trip=" << tripCount << ", factor=" << factor
                         << ") into " << outerTrip << " x " << factor);

        auto parent = dyn_cast<ForOp>(op->getParentOp());
        if (!parent) {
            return rewriter.notifyMatchFailure(
                loc,
                "expect enclosing loop at outer range");
        }

        // The "factor"-wide loop replaces "op" right where it was.
        rewriter.setInsertionPoint(op);
        auto newInner = ForOp::create(rewriter, loc, 0, factor, 1);
        Block &innerBody = newInner.getBody().front();
        rewriter.inlineBlockBefore(
            &op.getBody().front(),
            &innerBody,
            innerBody.end(),
            ValueRange{newInner.getInductionVariable()});
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created new loop 0..." << newInner.getUpperBound());
        rewriter.eraseOp(op);

        // The split-out "tripCount / factor" multiplies into the parent
        // loop's own trip count instead of wrapping it, so the parent's
        // own index keeps addressing correctly.
        int64_t parentLB = parent.getLowerBound().getSExtValue();
        int64_t parentStep = parent.getStep().getSExtValue();
        int64_t newParentTripCount = parent.getTripCount() * outerTrip;
        int64_t newParentUB = parentLB + newParentTripCount * parentStep;
        rewriter.modifyOpInPlace(parent, [&]() {
            parent->setAttr("upperBound", rewriter.getIndexAttr(newParentUB));
        });
        LAKSA_DEBUG(
            llvm::dbgs() << "  Scaled parent loop at " << parent.getLoc()
                         << " up to trip count " << newParentTripCount);

        auto pipelineAttr = op->getAttrOfType<BoolAttr>("dse.pipelined");
        if (!pipelineAttr) {
            return rewriter.notifyMatchFailure(
                loc,
                "expect a \"dse.pipelined\" attribute");
        }
        if (pipelineAttr.getValue()) {
            rewriter.setInsertionPointToStart(&parent.getBody().front());
            PragmaPipelineOp::create(rewriter, parent.getLoc());
        }

        return success();
    }
};

// A non-I/O loop carrying "dse.factor" is strip-mined into an outer
// "tripCount / splitFactor" loop, pipelined if "dse.pipelined" is set,
// wrapping a "splitFactor"-wide inner loop; "splitFactor" is "dse.factor"
// unless "dse.port" is wider. A port-touching index use takes the inner
// loop's own index, everything else gets the true index rebuilt as
// "outer * splitFactor + inner", and a "splitFactor=1" unpipelined loop is
// left alone with its tags stripped.
struct SplitFactorLoop : public OpConversionPattern<ForOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ForOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (op->hasAttr("dse.io")) return failure();
        auto factorAttr = op->getAttrOfType<IntegerAttr>("dse.factor");
        if (!factorAttr) return failure();

        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Rewriting loop inside node function at " << loc);
        int64_t factor = factorAttr.getInt();
        bool pipelined = false;
        if (auto pipelinedAttr = op->getAttrOfType<BoolAttr>("dse.pipelined"))
            pipelined = pipelinedAttr.getValue();
        int64_t tripCount = op.getTripCount();

        // A pair of loops, one in each branch of the same "if", that both
        // already fully unroll and pipeline don't need splitting: their
        // tags are stripped and the enclosing loop gets the pipeline
        // pragma instead, so Vitis HLS can flatten both branches under one
        // region. Whichever loop is visited first handles both, so the
        // other is already legal by the time the driver reaches it.
        if (pipelined && factor == tripCount) {
            if (auto ifOp = dyn_cast<IfOp>(op->getParentOp())) {
                bool inThen = op->getParentRegion() == &ifOp.getThenRegion();
                Region &siblingRegion =
                    inThen ? ifOp.getElseRegion() : ifOp.getThenRegion();
                ForOp sibling;
                if (!siblingRegion.empty())
                    for (Operation &siblingOp : siblingRegion.front()) {
                        auto siblingLoop = dyn_cast<ForOp>(siblingOp);
                        if (!siblingLoop) continue;
                        auto siblingFactorAttr =
                            siblingLoop->getAttrOfType<IntegerAttr>(
                                "dse.factor");
                        auto siblingPipelinedAttr =
                            siblingLoop->getAttrOfType<BoolAttr>(
                                "dse.pipelined");
                        if (siblingFactorAttr && siblingPipelinedAttr
                            && siblingPipelinedAttr.getValue()
                            && siblingFactorAttr.getInt()
                                   == siblingLoop.getTripCount()) {
                            sibling = siblingLoop;
                            break;
                        }
                    }
                auto parent =
                    sibling ? dyn_cast<ForOp>(ifOp->getParentOp()) : ForOp{};
                if (parent) {
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "  Collapsing matching if/else branch loops at "
                        << loc << " and " << sibling.getLoc()
                        << " under a shared pipeline on " << parent.getLoc());
                    auto stripTags = [&](ForOp loop) {
                        rewriter.modifyOpInPlace(loop, [&]() {
                            loop->removeAttr("dse.factor");
                            loop->removeAttr("dse.pipelined");
                            loop->removeAttr("dse.port");
                        });
                    };
                    stripTags(op);
                    stripTags(sibling);
                    rewriter.setInsertionPointToStart(
                        &parent.getBody().front());
                    PragmaPipelineOp::create(rewriter, ifOp.getLoc());
                    return success();
                }
            }
        }

        auto portAttr = op->getAttrOfType<IntegerAttr>("dse.port");
        int64_t splitFactor = (portAttr && portAttr.getInt() > factor)
                                  ? portAttr.getInt()
                                  : factor;

        if (!pipelined && splitFactor == 1) {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Dropping dse tags at " << loc << ", nothing to split");
            rewriter.modifyOpInPlace(op, [&]() {
                op->removeAttr("dse.factor");
                op->removeAttr("dse.pipelined");
                op->removeAttr("dse.port");
            });
            return success();
        }

        int64_t outerTrip = tripCount / splitFactor;
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Splitting loop at " << loc << " (trip=" << tripCount
            << ", splitFactor=" << splitFactor << ", pipelined=" << pipelined
            << ") into " << outerTrip << " x " << splitFactor);

        // Every direct, port-touching use of the old index will just take
        // the inner loop's own induction variable; everything else needs
        // the true original index rebuilt below, so it's captured now,
        // before "op"'s own induction variable stops existing.
        Value oldIV = op.getInductionVariable();
        struct PortTouch {
            Value port;
            ValueRange indices;
        };
        SmallVector<std::pair<Operation*, unsigned>> nonPortUses;
        for (OpOperand &use : oldIV.getUses()) {
            auto touch =
                llvm::TypeSwitch<Operation*, std::optional<PortTouch>>(
                    use.getOwner())
                    .Case<StreamReadOp, StreamWriteOp>([](auto o) {
                        return PortTouch{o.getStream(), o.getIndices()};
                    })
                    .Case<ArrayPointerReadOp, ArrayPointerWriteOp>([](auto o) {
                        return PortTouch{o.getPointer(), o.getIndices()};
                    })
                    .Default(std::nullopt);
            bool isPortIndex = touch && isa<BlockArgument>(touch->port)
                               && llvm::is_contained(touch->indices, oldIV);
            if (!isPortIndex)
                nonPortUses.emplace_back(
                    use.getOwner(),
                    use.getOperandNumber());
        }

        rewriter.setInsertionPoint(op);
        auto newOuter = ForOp::create(rewriter, loc, 0, outerTrip, 1);
        rewriter.setInsertionPointToStart(&newOuter.getBody().front());
        if (pipelined) PragmaPipelineOp::create(rewriter, loc);
        auto newInner = ForOp::create(rewriter, loc, 0, splitFactor, 1);
        Value outerIV = newOuter.getInductionVariable();
        Value innerIV = newInner.getInductionVariable();

        Block &innerBody = newInner.getBody().front();
        rewriter.inlineBlockBefore(
            &op.getBody().front(),
            &innerBody,
            innerBody.end(),
            ValueRange{innerIV});
        rewriter.eraseOp(op);

        // The inline above pointed every old-index use at "innerIV"; the
        // non-port ones captured above are now corrected back to the true
        // original index, "outer * splitFactor + inner". When "outerTrip"
        // is 1 the outer loop only ever runs once, so "innerIV" alone is
        // already correct and no correction is needed.
        if (!nonPortUses.empty() && outerTrip > 1) {
            rewriter.setInsertionPointToStart(&innerBody);
            Value splitFactorConst = VariableOp::create(
                                         rewriter,
                                         loc,
                                         rewriter.getIndexType(),
                                         rewriter.getIndexAttr(splitFactor),
                                         /*isConst=*/true)
                                         .getVariable();
            Value combined = ExpressionOp::create(
                                 rewriter,
                                 loc,
                                 rewriter.getIndexType(),
                                 [&](OpBuilder &eb, Location el) {
                                     auto mul = ArithMulOp::create(
                                         eb,
                                         el,
                                         outerIV,
                                         splitFactorConst);
                                     auto add = ArithAddOp::create(
                                         eb,
                                         el,
                                         mul.getResult(),
                                         innerIV);
                                     YieldOp::create(eb, el, add.getResult());
                                 })
                                 .getResult();
            for (auto [owner, operandIdx] : nonPortUses)
                rewriter.modifyOpInPlace(owner, [&]() {
                    owner->setOperand(operandIdx, combined);
                });
        }

        return success();
    }
};

// A FuncOp's signature is rewritten from PragmaDSE's "dse.factors": a
// pointer port becomes its new element bit width, an array port becomes
// its new shape. The top function's own pointer ports get the same resize
// since they pass straight into an I/O bridge call. Every callee besides
// top also gets a "#pragma HLS INLINE off"; loops were already marked
// "dse.io" or "dse.port" by PragmaDSE and just ride along through the
// clone below.
struct RewriteFuncSignature : public OpConversionPattern<FuncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FuncOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Rewriting func \"" << op.getSymName() << "\" at " << loc);
        bool isTop = op->hasAttr("dse.top");
        auto factorsAttr = op->getAttrOfType<ArrayAttr>("dse.factors");
        bool isIO = op->hasAttr("dse.io_func");
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  This function is top=" << isTop << " io=" << isIO);

        // Every argument, top function included, is resized using its own
        // per-argument "dse.factors" entry. A pointer port's own new
        // element type is also kept around per index, so the word variable
        // an I/O write function packs into it can be resized to match once
        // the body is cloned below.
        SmallVector<Type> newArgTypes;
        SmallVector<Type> newPointeeTypes(op.getNumArguments());
        for (auto [i, argType] : llvm::enumerate(op.getArgumentTypes())) {
            if (!factorsAttr) {
                newArgTypes.push_back(argType);
                continue;
            }
            auto dims = cast<ArrayAttr>(factorsAttr[i]);
            if (auto ptrType = dyn_cast<PointerType>(argType)) {
                int64_t newBits = cast<IntegerAttr>(dims[0]).getInt();
                Type newPointee = rewriter.getIntegerType(newBits);
                newPointeeTypes[i] = newPointee;
                newArgTypes.push_back(
                    PointerType::get(rewriter.getContext(), newPointee));
            } else if (auto arrType = dyn_cast<emithls::ArrayType>(argType)) {
                SmallVector<int64_t> newShape = llvm::map_to_vector(
                    dims,
                    [](Attribute d) { return cast<IntegerAttr>(d).getInt(); });
                newArgTypes.push_back(
                    emithls::ArrayType::get(
                        newShape,
                        arrType.getElementType()));
            } else {
                // A bare (non-array) stream/scalar port has no shape of its
                // own to resize.
                newArgTypes.push_back(argType);
            }
        }
        auto newFuncType = rewriter.getFunctionType(newArgTypes, {});
        LAKSA_DEBUG(llvm::dbgs() << "  New signature is " << newFuncType);

        rewriter.setInsertionPoint(op);
        auto newFunc =
            FuncOp::create(rewriter, loc, op.getSymName(), newFuncType);
        SmallVector<Location> argLocs(newArgTypes.size(), loc);
        Block* entry = rewriter.createBlock(
            &newFunc.getBody(),
            newFunc.getBody().end(),
            newArgTypes,
            argLocs);

        // Map every old block argument onto its new counterpart up front, so
        // cloning the body below never leaves a reference to the old
        // (now dead) block behind as an unresolved SSA value.
        IRMapping mapping;
        for (auto [oldArg, newArg] :
             llvm::zip(op.getArguments(), entry->getArguments()))
            mapping.map(oldArg, newArg);

        rewriter.setInsertionPointToStart(entry);
        if (isTop) {
            // Marks this function as the design's top for Vitis HLS,
            // emitting a "#pragma HLS INTERFACE" for every port and letting
            // downstream tooling (e.g. the EmitHLSToHLSTcl/EmitHLSToVivadoTcl
            // translations) find the top function by walking for this op.
            PragmaTopInterfaceOp::create(rewriter, loc);
            // The top function's own buffer declarations stay directly in
            // its body, but its calls move into a "#pragma HLS DATAFLOW"
            // region so Vitis HLS runs the pipeline stages concurrently
            // instead of sequentially.
            for (Operation &bodyOp : op.getBody().front())
                if (isa<VariableOp>(bodyOp)) rewriter.clone(bodyOp, mapping);
            PragmaDataflowOp::create(
                rewriter,
                loc,
                [&](OpBuilder &b, Location) {
                    for (Operation &bodyOp : op.getBody().front())
                        if (isa<CallOp>(bodyOp)) b.clone(bodyOp, mapping);
                });
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Cloning calls into a dataflow pragma region.");
        } else {
            PragmaInlineOp::create(rewriter, loc, /*off=*/true);
            for (Operation &bodyOp : op.getBody().front())
                rewriter.clone(bodyOp, mapping);
        }

        // A widened/narrowed pointer port leaves the local "word" variable
        // an I/O write function packs before its own ptr_write stale at its
        // old width (the clone above just copied it verbatim); rebuild it
        // at the pointer's own new element type so the two agree again.
        for (auto [i, newArg] : llvm::enumerate(entry->getArguments())) {
            Type newPointee = newPointeeTypes[i];
            if (!newPointee) continue;
            SmallVector<ArrayPointerWriteOp> staleWrites;
            newFunc.walk([&](ArrayPointerWriteOp writeOp) {
                if (writeOp.getPointer() == newArg
                    && writeOp.getValue().getType() != newPointee)
                    staleWrites.push_back(writeOp);
            });
            for (ArrayPointerWriteOp writeOp : staleWrites) {
                auto wordVar = writeOp.getValue().getDefiningOp<VariableOp>();
                if (!wordVar) continue;
                int64_t initValue = 0;
                if (auto initAttr = dyn_cast_or_null<IntegerAttr>(
                        wordVar.getInitNumberAttr()))
                    initValue = initAttr.getInt();
                LAKSA_DEBUG(
                    llvm::dbgs() << "  Resizing word variable "
                                 << wordVar.getLoc() << " to " << newPointee);
                rewriter.setInsertionPoint(wordVar);
                Value newWordVar = VariableOp::create(
                    rewriter,
                    wordVar.getLoc(),
                    newPointee,
                    rewriter.getIntegerAttr(
                        cast<IntegerType>(newPointee),
                        initValue),
                    wordVar.getIsConst());
                wordVar.getVariable().replaceAllUsesWith(newWordVar);
                rewriter.eraseOp(wordVar);
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

// A local array's "dse.mem" becomes a bind_storage pragma: BRAM maps to
// ram_2p/bram, LUTRAM to rom_1p/lutram if const or ram_2p/lutram otherwise.
// Each dimension's "dse.factors" entry becomes its own array_partition
// pragma, complete if it fills the dimension and cyclic otherwise.
struct InsertArrayPragmas : public OpConversionPattern<VariableOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        VariableOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto memAttr = op->getAttrOfType<StringAttr>("dse.mem");
        if (!memAttr) return failure();

        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Rewriting memory type variable at " << loc);
        Value variable = op.getVariable();
        LAKSA_DEBUG(
            llvm::dbgs() << "  Inserting array pragmas for " << op.getLoc()
                         << ", mem=" << memAttr.getValue());

        BindStorageType storageType;
        BindStorageImpl storageImpl;
        if (memAttr.getValue() == "BRAM") {
            storageType = BindStorageType::ram_2p;
            storageImpl = BindStorageImpl::bram;
        } else {
            storageImpl = BindStorageImpl::lutram;
            storageType = op.getIsConst() ? BindStorageType::rom_1p
                                          : BindStorageType::ram_2p;
        }

        rewriter.setInsertionPointAfter(op);
        PragmaBindStorageOp::create(
            rewriter,
            loc,
            variable,
            storageType,
            storageImpl);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Pragma-ing the array with the bound storage type.");

        if (auto factorsAttr = op->getAttrOfType<ArrayAttr>("dse.factors")) {
            auto arrType = cast<emithls::ArrayType>(variable.getType());
            for (auto [d, factorAttr] : llvm::enumerate(factorsAttr)) {
                int64_t factor = cast<IntegerAttr>(factorAttr).getInt();
                if (factor == 1) continue;
                bool complete = factor == arrType.getShape()[d];
                PragmaArrayPartitionOp::create(
                    rewriter,
                    loc,
                    variable,
                    complete ? ArrayPartitionType::complete
                             : ArrayPartitionType::cyclic,
                    complete
                        ? std::nullopt
                        : std::optional<int32_t>(static_cast<int32_t>(factor)),
                    static_cast<int32_t>(d + 1));
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "  Pragma-ing the array with the partition factor.");
            }
        }

        rewriter.modifyOpInPlace(op, [&]() {
            op->removeAttr("dse.mem");
            op->removeAttr("dse.factors");
        });

        return success();
    }
};

// A "dse.fifo" buffer is rebuilt at its "dse.factors" shape, then gets a
// bind_storage and stream pragma with "dse.depth" as the depth. A bare
// stream has no shape to resize, so it just gets the two pragmas in place.
struct RewriteFIFOVariable : public OpConversionPattern<VariableOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        VariableOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (!op->hasAttr("dse.fifo")) return failure();

        Location loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting FIFO type variable at " << loc);
        int32_t depth = static_cast<int32_t>(
            cast<IntegerAttr>(op->getAttr("dse.depth")).getInt());
        auto factorsAttr = op->getAttrOfType<ArrayAttr>("dse.factors");
        auto arrType = dyn_cast<emithls::ArrayType>(op.getVariable().getType());

        Value variable = op.getVariable();
        if (arrType && factorsAttr) {
            SmallVector<int64_t> newShape = llvm::map_to_vector(
                factorsAttr,
                [](Attribute d) { return cast<IntegerAttr>(d).getInt(); });
            auto newType =
                emithls::ArrayType::get(newShape, arrType.getElementType());
            LAKSA_DEBUG(
                llvm::dbgs() << "  Rewriting FIFO buffer into " << newType);

            rewriter.setInsertionPoint(op);
            auto newOp = VariableOp::create(
                rewriter,
                loc,
                newType,
                /*initValue=*/Value{});
            variable = newOp.getVariable();
            rewriter.setInsertionPointAfter(newOp);
            PragmaBindStorageOp::create(
                rewriter,
                loc,
                variable,
                BindStorageType::fifo,
                BindStorageImpl::srl);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Pragma-ing the FIFO array with the bound storage type.");
            PragmaStreamOp::create(rewriter, loc, variable, depth);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Pragma-ing the FIFO array with stream mark.");
            op.getVariable().replaceAllUsesWith(variable);
            rewriter.eraseOp(op);
            return success();
        }

        LAKSA_DEBUG(
            llvm::dbgs() << "  Pragma-ing FIFO array " << op.getLoc()
                         << " in place, no array shape to resize");
        rewriter.setInsertionPointAfter(op);
        PragmaBindStorageOp::create(
            rewriter,
            loc,
            variable,
            BindStorageType::fifo,
            BindStorageImpl::srl);
        PragmaStreamOp::create(rewriter, loc, variable, depth);

        rewriter.modifyOpInPlace(op, [&]() {
            op->removeAttr("dse.fifo");
            op->removeAttr("dse.depth");
            op->removeAttr("dse.factors");
        });

        return success();
    }
};

} // namespace

namespace {
struct EmitHLSPragmaInsertionPass
        : public emithls::impl::EmitHLSPragmaInsertionBase<
              EmitHLSPragmaInsertionPass> {

    void runOnOperation() override;
};
} // namespace

void EmitHLSPragmaInsertionPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<RewriteFuncSignature>(&getContext());
    patterns.add<InsertArrayPragmas>(&getContext());
    patterns.add<RewriteFIFOVariable>(&getContext());
    patterns.add<SplitIOLoop>(&getContext());
    patterns.add<SplitFactorLoop>(&getContext());

    target.addLegalDialect<EmitHLSDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addDynamicallyLegalOp<FuncOp>(
        [](FuncOp op) { return !hasDSEAttr(op); });
    target.addDynamicallyLegalOp<VariableOp>(
        [](VariableOp op) { return !hasDSEAttr(op); });
    target.addDynamicallyLegalOp<ForOp>(
        [](ForOp op) { return !hasDSEAttr(op); });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSPragmaInsertionPass()
{ return std::make_unique<EmitHLSPragmaInsertionPass>(); }
