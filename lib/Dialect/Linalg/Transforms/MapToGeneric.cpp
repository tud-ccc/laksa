/// Implementation of LinalgMapToGeneric transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "linalg-map-to-generic"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[linalg-map-to-generic] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {
#define GEN_PASS_DEF_LINALGMAPTOGENERIC
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"
} // namespace linalg
} // namespace mlir

namespace {
struct MapToEquivalentGeneric : public OpRewritePattern<MapOp> {
    MapToEquivalentGeneric(MLIRContext* context)
            : OpRewritePattern<MapOp>(context) {};

    LogicalResult
    matchAndRewrite(MapOp op, PatternRewriter &rewriter) const override
    {

        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Transforming map operation at " << loc);

        SmallVector<Value> inputs = op.getDpsInputs();
        // Map operation has only one output
        Value output = op.getDpsInits().front();
        auto outputTy = output.getType();
        LAKSA_DEBUG(llvm::dbgs() << "Output type is " << outputTy;);
        auto outputShapedTy = cast<ShapedType>(outputTy);
        unsigned numOutTyDim = outputShapedTy.getShape().size();
        auto identityMap = AffineMap::getMultiDimIdentityMap(numOutTyDim, ctx);
        auto emptyMap = AffineMap::getMinorIdentityMap(numOutTyDim, 0, ctx);
        LAKSA_DEBUG(
            llvm::dbgs() << "Based on output's type, there are identity map "
                         << identityMap << ", and empty map " << emptyMap);
        SmallVector<AffineMap> affineMaps;

        Value fillValue = nullptr;
        if (inputs.empty()) {
            LAKSA_DEBUG(llvm::dbgs() << "Found fill operation as map.");
            affineMaps.push_back(emptyMap);
            // The yield operation is the block's terminator, not
            // necessarily its first operation (e.g. linalg.index may
            // precede it).
            auto yieldOp =
                cast<YieldOp>(op.getRegion().front().getTerminator());
            fillValue = yieldOp.getOperand(0);
            inputs.push_back(fillValue);
        }
        for (unsigned i = 0; i < op.getNumDpsInputs() + op.getNumDpsInits();
             ++i)
            affineMaps.push_back(identityMap);

        // All iterator should be parallel type
        SmallVector<utils::IteratorType> iteratorTypes;
        for (unsigned i = 0; i < numOutTyDim; ++i)
            iteratorTypes.push_back(utils::IteratorType::parallel);

        bool isTensor = isa<TensorType>(outputTy);
        rewriter.replaceOpWithNewOp<GenericOp>(
            op,
            isTensor ? TypeRange{outputTy} : TypeRange{},
            inputs,
            output,
            affineMaps,
            iteratorTypes,
            /*doc*/ "",
            /*libraryCall*/ "",
            [&](OpBuilder &genericBuilder, Location, ValueRange blockArgs) {
                IRMapping mapper;
                if (fillValue) mapper.map(fillValue, blockArgs[0]);
                for (auto [oldArg, newArg] :
                     llvm::zip(op.getRegion().getArguments(), blockArgs)) {
                    mapper.map(oldArg, newArg);
                }
                for (auto &opMap : op.getRegion().getOps())
                    genericBuilder.clone(opMap, mapper);
            });
        LAKSA_DEBUG(llvm::dbgs() << "Replaced map with generic.");
        return success();
    }
};
} // namespace

namespace {

struct LinalgMapToGenericPass
        : public linalg::impl::LinalgMapToGenericBase<LinalgMapToGenericPass> {
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<MapToEquivalentGeneric>(patterns.getContext());

        target.addLegalDialect<LinalgDialect>();
        target.addIllegalOp<MapOp>();
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

std::unique_ptr<Pass> mlir::linalg::createLinalgMapToGenericPass()
{ return std::make_unique<LinalgMapToGenericPass>(); }
