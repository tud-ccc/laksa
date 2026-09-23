/// Implementation of InlineEmbedRegion transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Region.h>

#define DEBUG_TYPE "dfg-inline-embed-region"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[dfg-inline-embed-region] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::dfg;

using RegionEmbedMap = DenseMap<Operation*, SmallVector<Operation*>>;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGINLINEEMBEDREGION
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

namespace {
struct InlineRegionPattern : public OpRewritePattern<EmbedOp> {
    using OpRewritePattern<EmbedOp>::OpRewritePattern;

    InlineRegionPattern(MLIRContext* context, RegionEmbedMap &regionEmbedMap)
            : OpRewritePattern<EmbedOp>(context),
              regionEmbedMap(regionEmbedMap)
    {}

    LogicalResult
    matchAndRewrite(EmbedOp op, PatternRewriter &rewriter) const override
    {
        auto embeddedRegion = cast<RegionOp>(op.getEmbeddedOperation());
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Found embedded region " << embeddedRegion.getNodeName()
            << " at " << embeddedRegion.getLoc());

        IRMapping mapper;
        for (auto [blkArg, portArg] : llvm::zip(
                 embeddedRegion.getBody().getArguments(),
                 op->getOperands())) {
            mapper.map(blkArg, portArg);
        }

        rewriter.setInsertionPoint(op);
        LAKSA_DEBUG(llvm::dbgs() << "Embed operations at " << op.getLoc());
        for (auto &opEmbedRegion : embeddedRegion.getBody().getOps())
            rewriter.clone(opEmbedRegion, mapper);

        // Remove this embed op from the tracking map before erasing it.
        auto &embedOps = regionEmbedMap[embeddedRegion];
        embedOps.erase(llvm::find(embedOps, op));
        // Erase this embed
        rewriter.eraseOp(op);
        // Only erase the region once all embed ops referencing it are done.
        if (embedOps.empty()) rewriter.eraseOp(embeddedRegion);

        return success();
    }

private:
    RegionEmbedMap &regionEmbedMap;
};
} // namespace

namespace {
struct DFGInlineEmbedRegionPass
        : public dfg::impl::DFGInlineEmbedRegionBase<DFGInlineEmbedRegionPass> {
    void runOnOperation() override
    {
        // First, find all the embed operations for a region
        Operation* moduleOp = getOperation();
        RegionEmbedMap regionEmbedMap;
        moduleOp->walk([&](EmbedOp embedOp) {
            regionEmbedMap[embedOp.getEmbeddedOperation()].push_back(embedOp);
        });

        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<InlineRegionPattern>(
            patterns.getContext(),
            regionEmbedMap);

        target.addLegalDialect<DFGDialect>();
        target.addIllegalOp<EmbedOp>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(applyPartialConversion(
                getOperation(),
                target,
                std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createDFGInlineEmbedRegionPass()
{ return std::make_unique<DFGInlineEmbedRegionPass>(); }
