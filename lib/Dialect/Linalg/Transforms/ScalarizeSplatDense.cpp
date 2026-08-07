/// Implementation of LinalgScalarizeSplatDense transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "linalg-scalarize-splat-dense"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[linalg-scalarize-splat-dense] "; X;                  \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {
#define GEN_PASS_DEF_LINALGSCALARIZESPLATDENSE
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"
} // namespace linalg
} // namespace mlir

namespace {
struct ScalarizeSplatValues : public OpRewritePattern<GenericOp> {
    ScalarizeSplatValues(MLIRContext* context)
            : OpRewritePattern<GenericOp>(context) {};

    LogicalResult
    matchAndRewrite(GenericOp op, PatternRewriter &rewriter) const override
    {
        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Found generic operation using splat dense constant at " << loc);

        SmallVector<Value> inputs = op.getDpsInputs();
        SmallVector<AffineMap> allMaps = op.getIndexingMapsArray();
        unsigned numInputs = inputs.size();

        // For each splat dense tensor input, determine:
        // - The scalar value to replace it with.
        // - Whether any iteration dims are "exclusive" to its map (appear as
        // bare AffineDimExpr only here, not in any other map). Those dims must
        // stay reachable in the combined map, so inject a small dummy tensor
        // whose indexing map covers exactly those exclusive dims.
        struct SplatEntry {
            Value scalarVal;
            AffineMap dummyMap; // null if no exclusive dims
            SmallVector<int64_t> dummyShape;
        };
        SmallVector<std::optional<SplatEntry>> entries(numInputs, std::nullopt);
        bool anyScalarized = false;

        for (auto [idx, input] : llvm::enumerate(inputs)) {
            if (isa<BlockArgument>(input)) continue;
            auto constantOp =
                dyn_cast<arith::ConstantOp>(input.getDefiningOp());
            if (!constantOp) continue;
            auto tensorType = dyn_cast<TensorType>(constantOp.getType());
            if (!tensorType) continue;
            auto denseAttr = cast<DenseElementsAttr>(constantOp.getValueAttr());
            if (!denseAttr.isSplat()) continue;

            // Create scalar constant from the splat value.
            auto elemType = tensorType.getElementType();
            TypedAttr splatAttr;
            LogicalResult status =
                llvm::TypeSwitch<Type, LogicalResult>(elemType)
                    .Case([&](IntegerType t) {
                        splatAttr = rewriter.getIntegerAttr(
                            t,
                            denseAttr.getSplatValue<llvm::APInt>());
                        return success();
                    })
                    .Case([&](FloatType t) {
                        splatAttr = rewriter.getFloatAttr(
                            t,
                            denseAttr.getSplatValue<llvm::APFloat>());
                        return success();
                    })
                    .Default([](Type) { return failure(); });
            if (failed(status))
                return rewriter.notifyMatchFailure(
                    op,
                    "unsupported element type for splat scalarization");

            rewriter.setInsertionPoint(constantOp);
            Value scalarVal = arith::ConstantOp::create(
                rewriter,
                constantOp.getLoc(),
                elemType,
                splatAttr);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "Created scalar constant with splat value " << splatAttr);

            // Collect dims that appear as bare AffineDimExprs in OTHER maps.
            llvm::SmallDenseSet<unsigned> otherSingleDims;
            for (auto [j, map] : llvm::enumerate(allMaps)) {
                if (j == idx) continue;
                for (auto res : map.getResults())
                    if (auto d = dyn_cast<AffineDimExpr>(res))
                        otherSingleDims.insert(d.getPosition());
            }

            // Exclusive dims: appear as bare AffineDimExpr here but not
            // elsewhere. They must be preserved via a small dummy tensor.
            SmallVector<AffineExpr> exclusiveResults;
            SmallVector<int64_t> dummyShape;
            AffineMap inputMap = allMaps[idx];
            for (auto [i, res] : llvm::enumerate(inputMap.getResults())) {
                auto d = dyn_cast<AffineDimExpr>(res);
                if (!d) continue;
                if (!otherSingleDims.count(d.getPosition())) {
                    exclusiveResults.push_back(res);
                    dummyShape.push_back(tensorType.getDimSize(i));
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "Dim d" << d.getPosition()
                        << " is exclusive — covered by dummy tensor.");
                }
            }

            SplatEntry entry;
            entry.scalarVal = scalarVal;
            entry.dummyShape = std::move(dummyShape);
            if (!exclusiveResults.empty())
                entry.dummyMap = AffineMap::get(
                    inputMap.getNumDims(),
                    0,
                    exclusiveResults,
                    ctx);
            entries[idx] = std::move(entry);
            anyScalarized = true;
        }

        if (!anyScalarized)
            return rewriter.notifyMatchFailure(
                op,
                "no scalarizable splat inputs found");

        // Build new inputs / maps lists.
        //
        // Each scalarized input expands to:
        //   scalar (map = ())  [+ dummy tensor (map = exclusive sub-map), if
        //   needed]
        //
        // The old block arg for the scalarized input maps to the scalar's new
        // block arg (same element type).  The dummy's block arg is unused in
        // the body — it only ensures the exclusive dims remain in the combined
        // map so that IndexingMapOpInterface::verifyImpl passes.
        auto emptyMap =
            AffineMap::getMinorIdentityMap(allMaps[0].getNumDims(), 0, ctx);

        SmallVector<Value> newInputs;
        SmallVector<AffineMap> newInputMaps;
        // oldToNewArgIdx[i] = index in newBlockArgs that replaces old input arg
        // i.
        SmallVector<unsigned> oldToNewArgIdx(numInputs);

        rewriter.setInsertionPoint(op);
        unsigned curNewArgIdx = 0;

        for (auto [idx, input] : llvm::enumerate(inputs)) {
            if (!entries[idx].has_value()) {
                newInputs.push_back(input);
                newInputMaps.push_back(allMaps[idx]);
                oldToNewArgIdx[idx] = curNewArgIdx++;
                continue;
            }

            auto &entry = *entries[idx];

            // Scalar replaces the original tensor.  Old block arg → scalar
            // block arg.
            newInputs.push_back(entry.scalarVal);
            newInputMaps.push_back(emptyMap);
            oldToNewArgIdx[idx] = curNewArgIdx++;

            // Dummy tensor to keep exclusive dims reachable in the combined
            // map.
            if (entry.dummyMap) {
                auto elemType =
                    cast<TensorType>(input.getType()).getElementType();
                Value dummy = tensor::EmptyOp::create(
                    rewriter,
                    loc,
                    entry.dummyShape,
                    elemType);
                newInputs.push_back(dummy);
                newInputMaps.push_back(entry.dummyMap);
                curNewArgIdx++; // dummy block arg — present but unused in body
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "Injected dummy tensor with map " << entry.dummyMap);
            }
        }

        // Append unchanged output maps.
        SmallVector<AffineMap> newAffineMaps(newInputMaps);
        for (unsigned i = numInputs; i < allMaps.size(); ++i)
            newAffineMaps.push_back(allMaps[i]);

        unsigned numNewInputs = newInputs.size();

        rewriter.replaceOpWithNewOp<GenericOp>(
            op,
            op.getResultTypes(),
            newInputs,
            op.getDpsInits(),
            newAffineMaps,
            op.getIteratorTypesArray(),
            /*doc*/ "",
            /*libraryCall*/ "",
            [&](OpBuilder &b, Location, ValueRange newBlockArgs) {
                IRMapping mapper;
                // Map each old input block arg to its replacement new block
                // arg.
                for (unsigned i = 0; i < numInputs; ++i)
                    mapper.map(
                        op.getRegion().getArguments()[i],
                        newBlockArgs[oldToNewArgIdx[i]]);
                // Map old output block args to new output block args.
                for (auto [oldArg, newArg] : llvm::zip(
                         op.getRegion().getArguments().drop_front(numInputs),
                         newBlockArgs.drop_front(numNewInputs)))
                    mapper.map(oldArg, newArg);
                for (auto &innerOp : op.getRegion().getOps())
                    b.clone(innerOp, mapper);
            });

        LAKSA_DEBUG(
            llvm::dbgs() << "Created new generic with scalarized input(s).");
        return success();
    }
};
} // namespace

namespace {

struct LinalgScalarizeSplatDensePass
        : public linalg::impl::LinalgScalarizeSplatDenseBase<
              LinalgScalarizeSplatDensePass> {
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<ScalarizeSplatValues>(patterns.getContext());

        target.addLegalDialect<LinalgDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
        target.addDynamicallyLegalOp<GenericOp>([](GenericOp op) {
            for (auto input : op.getDpsInputs()) {
                if (!isa<BlockArgument>(input)) {
                    if (auto constantOp = dyn_cast<arith::ConstantOp>(
                            input.getDefiningOp())) {
                        auto constantType = constantOp.getType();
                        auto constantAttr = constantOp.getValueAttr();
                        if (isa<TensorType>(constantType)) {
                            auto constantDenseAttr =
                                cast<DenseElementsAttr>(constantAttr);
                            if (constantDenseAttr.isSplat()) return false;
                        }
                    }
                }
            }
            return true;
        });

        if (failed(applyPartialConversion(
                getOperation(),
                target,
                std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::linalg::createLinalgScalarizeSplatDensePass()
{ return std::make_unique<LinalgScalarizeSplatDensePass>(); }
