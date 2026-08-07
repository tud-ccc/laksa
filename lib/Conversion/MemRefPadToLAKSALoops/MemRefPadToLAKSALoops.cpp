/// Implementation of MemRefPadToLAKSALoops pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/MemRefPadToLAKSALoops/MemRefPadToLAKSALoops.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "memref-pad-to-laksa-loops"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[memref-pad-to-laksa-loops] "; X;                     \
        llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFPADTOLAKSALOOPS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::emithls;

namespace {

// One alloc dim that the subview crops: [offset, offset+size) is real data, the
// rest of [0, allocSize) is padding to be filled with `fill_with`.
struct PadDim {
    unsigned dim;
    int64_t offset;
    int64_t size;
    int64_t allocSize;
};

// Rewrite information collection
struct PadStreamMatch {
    Operation* pullOp;                 // a dfg.pull or dfg.pull_as_memref
    UnrealizedConversionCastOp castIn; // port shape -> real (unpadded) shape
    memref::AllocOp allocOp;           // padded shape, has `fill_with` attr
    memref::SubViewOp subviewOp;       // real-data window into allocOp
    memref::CopyOp copyOp;
    UnrealizedConversionCastOp castOut; // padded shape -> port shape
    Operation* pushOp;                  // a dfg.push or dfg.push_memref
    Type tokenType; // type of the value pulled/pushed per port firing
    SmallVector<PadDim> padDims; // alloc dims where subview < alloc size
};

// Recognizes the pad idiom anchored at `subviewOp` and checks every structural
// precondition the rewrite below relies on.
static FailureOr<PadStreamMatch>
matchPadStreamPattern(memref::SubViewOp subviewOp)
{
    PadStreamMatch match;
    match.subviewOp = subviewOp;

    match.allocOp = subviewOp.getSource().getDefiningOp<memref::AllocOp>();
    if (!match.allocOp || !match.allocOp->hasAttr("fill_with")) {
        subviewOp.emitError() << "expected subview source to be a "
                                 "memref.alloc with a 'fill_with' attribute";
        return failure();
    }

    if (!subviewOp->hasOneUse()) {
        subviewOp.emitError() << "expected subview to have exactly one use";
        return failure();
    }
    match.copyOp = dyn_cast<memref::CopyOp>(*subviewOp->getUsers().begin());
    if (!match.copyOp || match.copyOp.getTarget() != subviewOp.getResult()) {
        subviewOp.emitError()
            << "expected subview's only use to be a memref.copy into it";
        return failure();
    }

    match.castIn =
        match.copyOp.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!match.castIn || match.castIn.getInputs().size() != 1
        || match.castIn.getOutputs().size() != 1
        || !match.castIn->hasOneUse()) {
        subviewOp.emitError()
            << "expected the copy source to be a single-input, "
               "single-result, single-use unrealized_conversion_cast";
        return failure();
    }
    Operation* producer = match.castIn.getInputs()[0].getDefiningOp();
    if (!producer || !isa<dfg::PullOp, dfg::PullAsMemRefOp>(producer)
        || !producer->hasOneUse()) {
        subviewOp.emitError() << "expected the cast input to come from a "
                                 "single-use dfg.pull or dfg.pull_as_memref";
        return failure();
    }
    match.pullOp = producer;

    SmallVector<Operation*> allocUsers(match.allocOp->getUsers());
    if (allocUsers.size() != 2) {
        subviewOp.emitError() << "expected the padded alloc to have exactly "
                                 "two uses: this subview and a narrowing "
                                 "cast";
        return failure();
    }
    Operation* other = allocUsers[0] == subviewOp.getOperation()
                           ? allocUsers[1]
                           : allocUsers[0];
    if (allocUsers[0] != subviewOp.getOperation()
        && allocUsers[1] != subviewOp.getOperation()) {
        subviewOp.emitError() << "expected one of the padded alloc's two "
                                 "uses to be this subview";
        return failure();
    }
    match.castOut = dyn_cast<UnrealizedConversionCastOp>(other);
    if (!match.castOut || match.castOut.getOutputs().size() != 1
        || !match.castOut->hasOneUse()) {
        subviewOp.emitError()
            << "expected the alloc's other use to be a single-result, "
               "single-use unrealized_conversion_cast";
        return failure();
    }

    Operation* consumer = *match.castOut->getUsers().begin();
    if (!isa<dfg::PushOp, dfg::PushMemRefOp>(consumer)
        || consumer->getOperand(0) != match.castOut.getOutputs()[0]) {
        subviewOp.emitError()
            << "expected the narrowing cast's only use to be a dfg.push or "
               "dfg.push_memref consuming its result";
        return failure();
    }
    match.pushOp = consumer;
    match.tokenType = match.castOut.getOutputs()[0].getType();

    Block* block = subviewOp->getBlock();
    for (Operation* op :
         {match.pullOp,
          match.castIn.getOperation(),
          match.allocOp.getOperation(),
          match.copyOp.getOperation(),
          match.castOut.getOperation(),
          match.pushOp}) {
        if (op->getBlock() != block) {
            subviewOp.emitError()
                << "expected the whole pad idiom to live in the same block";
            return failure();
        }
    }
    if (!isa<dfg::LoopOp>(block->getParentOp())) {
        subviewOp.emitError()
            << "expected the pad idiom to sit directly inside a dfg.loop "
               "body";
        return failure();
    }

    if (llvm::is_contained(subviewOp.getStaticOffsets(), ShapedType::kDynamic)
        || llvm::is_contained(
            subviewOp.getStaticSizes(),
            ShapedType::kDynamic)) {
        subviewOp.emitError()
            << "dynamic subview offsets/sizes are not supported yet";
        return failure();
    }

    // Figure out which alloc dims are actually padded
    ArrayRef<int64_t> allocShape = match.allocOp.getType().getShape();
    ArrayRef<int64_t> offsets = subviewOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = subviewOp.getStaticSizes();
    if (sizes.size() != allocShape.size()) {
        subviewOp.emitError()
            << "expected subview rank to match the padded alloc's rank";
        return failure();
    }
    for (unsigned i = 0; i < allocShape.size(); ++i) {
        if (sizes[i] == allocShape[i]) {
            if (offsets[i] != 0) {
                subviewOp.emitError() << "expected subview to span the full "
                                         "extent on non-padded dims";
                return failure();
            }
            continue;
        }
        match.padDims.push_back({i, offsets[i], sizes[i], allocShape[i]});
    }
    if (match.padDims.empty()) {
        subviewOp.emitError() << "expected subview to actually crop at "
                                 "least one dim of the padded alloc";
        return failure();
    }

    // A memref-shaped port token stashes `fill_with` on the filler alloc for
    // some later pass to realize. A scalar one doesn't have a buffer to
    // stash it on, so the attribute must already be directly usable as the
    // pushed value.
    auto fillAttr = cast<TypedAttr>(match.allocOp->getAttr("fill_with"));
    if (!isa<MemRefType>(match.tokenType)
        && fillAttr.getType() != match.tokenType) {
        subviewOp.emitError() << "expected the 'fill_with' attribute's type "
                                 "to match the scalar port token type";
        return failure();
    }

    return match;
}

// Turns a `fill_with` attribute into an actual constant op producing that
// value with scalar-token ports.
static Value
materializeFillValue(OpBuilder &builder, Location loc, TypedAttr fillAttr)
{ return arith::ConstantOp::create(builder, loc, fillAttr); }

struct RewritePadStreamAsLoop : OpConversionPattern<memref::SubViewOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        memref::SubViewOp op,
        OpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto matched = matchPadStreamPattern(op);
        if (failed(matched))
            return rewriter.notifyMatchFailure(op, "not a pad-stream idiom");
        PadStreamMatch m = *matched;
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting pad-stream idiom at " << loc);

        // Filler value pushed for padding positions: a small `fill_with` buffer
        // for a memref-shaped port, or the fill constant itself for a scalar
        // one.
        rewriter.setInsertionPoint(m.pullOp);
        Value fillerValue;
        if (auto fillerType = dyn_cast<MemRefType>(m.tokenType)) {
            auto fillerAlloc =
                memref::AllocOp::create(rewriter, loc, fillerType);
            for (NamedAttribute attr : m.allocOp->getAttrs())
                fillerAlloc->setAttr(attr.getName(), attr.getValue());
            fillerValue = fillerAlloc.getMemref();
            LAKSA_DEBUG(
                llvm::dbgs() << "  filler is a memref alloc: " << fillerValue);
        } else {
            auto fillAttr = cast<TypedAttr>(m.allocOp->getAttr("fill_with"));
            fillerValue = materializeFillValue(rewriter, loc, fillAttr);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  filler is a scalar constant: " << fillerValue);
        }

        // Create the nested affine.for loop
        SmallVector<Value> ivs;
        for (const PadDim &pd : m.padDims) {
            auto forOp = AffineForOp::create(rewriter, loc, 0, pd.allocSize);
            ivs.push_back(forOp.getInductionVar());
            LAKSA_DEBUG(
                llvm::dbgs() << "  affine.for over dim " << pd.dim << ": [0, "
                             << pd.allocSize << "), real window [" << pd.offset
                             << ", " << (pd.offset + pd.size) << ")");
            rewriter.setInsertionPointToStart(&forOp.getRegion().front());
        }

        // Create affine.if to control the output values
        SmallVector<AffineExpr> constraints;
        SmallVector<bool> eqFlags;
        for (auto [i, pd] : llvm::enumerate(m.padDims)) {
            AffineExpr d = getAffineDimExpr(i, rewriter.getContext());
            constraints.push_back(d - pd.offset); // >= 0
            eqFlags.push_back(false);
            constraints.push_back(pd.offset + pd.size - 1 - d); // >= 0
            eqFlags.push_back(false);
        }
        auto condSet =
            IntegerSet::get(m.padDims.size(), 0, constraints, eqFlags);
        auto ifOp = AffineIfOp::create(
            rewriter,
            loc,
            condSet,
            ivs,
            /*withElseRegion=*/true);
        LAKSA_DEBUG({
            llvm::dbgs() << "  affine.if condition set: ";
            condSet.print(llvm::dbgs());
        });

        // Clone the original pull/push with retargeting the pushed value
        rewriter.setInsertionPointToStart(ifOp.getThenBlock());
        Operation* newPull = rewriter.clone(*m.pullOp);
        IRMapping thenMap;
        thenMap.map(m.pushOp->getOperand(0), newPull->getResult(0));
        Operation* newPush = rewriter.clone(*m.pushOp, thenMap);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Cloned " << *newPull << "  and  " << *newPush
                         << " into then region");

        // In else region, push the padded value/buffer
        rewriter.setInsertionPointToStart(ifOp.getElseBlock());
        IRMapping elseMap;
        elseMap.map(m.pushOp->getOperand(0), fillerValue);
        Operation* fillerPush = rewriter.clone(*m.pushOp, elseMap);
        LAKSA_DEBUG(
            llvm::dbgs() << "  Cloned " << *fillerPush << " into else region");

        // Erase old operations in order
        rewriter.eraseOp(m.pushOp);
        rewriter.eraseOp(m.copyOp);
        rewriter.eraseOp(m.castOut);
        rewriter.eraseOp(m.subviewOp);
        rewriter.eraseOp(m.allocOp);
        rewriter.eraseOp(m.castIn);
        rewriter.eraseOp(m.pullOp);
        return success();
    }
};

} // namespace

void mlir::populateMemRefPadToLAKSALoopsConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{ patterns.add<RewritePadStreamAsLoop>(typeConverter, patterns.getContext()); }

namespace {
struct ConvertMemRefPadToLAKSALoopsPass
        : public impl::ConvertMemRefPadToLAKSALoopsBase<
              ConvertMemRefPadToLAKSALoopsPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertMemRefPadToLAKSALoopsPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateMemRefPadToLAKSALoopsConversionPatterns(converter, patterns);

    target.addLegalDialect<
        EmitHLSDialect,
        dfg::DFGDialect,
        affine::AffineDialect>();
    target.addLegalOp<memref::AllocOp, memref::GlobalOp, memref::GetGlobalOp>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertMemRefPadToLAKSALoopsPass()
{ return std::make_unique<ConvertMemRefPadToLAKSALoopsPass>(); }
