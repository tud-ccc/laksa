/// Implementation of LinalgSoftTranspose transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "linalg-soft-transpose"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[linalg-soft-transpose] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {
#define GEN_PASS_DEF_LINALGSOFTTRANSPOSE
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h.inc"
} // namespace linalg
} // namespace mlir

namespace {

/// Reorders elements of @p data (in row-major flat layout) from @p oldShape to
/// @p newShape according to @p permutation, writing results into @p newData.
/// @p permutation[i] gives the old dimension that maps to new dimension `i`,
/// matching the semantics of `linalg.transpose`.
template<typename T>
void transposeData(
    const std::vector<T> &data,
    ArrayRef<int64_t> oldShape,
    ArrayRef<int64_t> newShape,
    ArrayRef<int64_t> permutation,
    std::vector<T> &newData)
{
    auto numDims = oldShape.size();
    SmallVector<int64_t> oldStrides(numDims, 1);
    for (int i = numDims - 2; i >= 0; --i)
        oldStrides[i] = oldStrides[i + 1] * oldShape[i + 1];
    SmallVector<int64_t> newStrides(numDims, 1);
    for (int i = numDims - 2; i >= 0; --i)
        newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    SmallVector<int64_t> indices(numDims);
    SmallVector<int64_t> newIndices(numDims);
    for (auto [flatIdx, value] : llvm::enumerate(data)) {
        int64_t remaining = flatIdx;
        for (unsigned i = 0; i < numDims; ++i) {
            indices[i] = remaining / oldStrides[i];
            remaining %= oldStrides[i];
        }
        for (unsigned i = 0; i < numDims; ++i)
            newIndices[i] = indices[permutation[i]];
        int64_t newFlatIdx = 0;
        for (unsigned i = 0; i < numDims; ++i)
            newFlatIdx += newIndices[i] * newStrides[i];
        newData[newFlatIdx] = value;
    }
}

struct TransposeConstant : public OpRewritePattern<TransposeOp> {
    TransposeConstant(MLIRContext* context)
            : OpRewritePattern<TransposeOp>(context) {};

    LogicalResult
    matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override
    {
        auto input = op.getInput();
        auto inputType = input.getType();
        auto init = op.getInit();
        auto initType = init.getType();

        LAKSA_DEBUG(
            llvm::dbgs() << "Matching linalg.transpose at " << op.getLoc()
                         << ": " << inputType << " -> " << initType);

        auto defOp = dyn_cast_or_null<arith::ConstantOp>(input.getDefiningOp());
        if (!defOp)
            return rewriter.notifyMatchFailure(op, "input is not a constant");
        auto denseAttr = dyn_cast<DenseElementsAttr>(defOp.getValueAttr());
        if (!denseAttr)
            return rewriter.notifyMatchFailure(op, "expected dense constant");

        auto permutation = op.getPermutation();
        auto elemTy = inputType.getElementType();
        DenseElementsAttr newDenseAttr;
        if (isa<IntegerType>(elemTy)) {
            LAKSA_DEBUG(
                llvm::dbgs() << "Folding integer transpose " << inputType
                             << " -> " << initType);
            auto values = denseAttr.getValues<llvm::APInt>();
            std::vector<llvm::APInt> data(values.begin(), values.end());
            std::vector<llvm::APInt> newData(data.size(), llvm::APInt());
            transposeData(
                data,
                inputType.getShape(),
                initType.getShape(),
                permutation,
                newData);
            newDenseAttr = DenseElementsAttr::get(initType, newData);
        } else if (isa<FloatType>(elemTy)) {
            LAKSA_DEBUG(
                llvm::dbgs() << "Folding float transpose " << inputType
                             << " -> " << initType);
            auto values = denseAttr.getValues<llvm::APFloat>();
            std::vector<llvm::APFloat> data(values.begin(), values.end());
            std::vector<llvm::APFloat> newData(
                data.size(),
                llvm::APFloat::getZero(
                    cast<FloatType>(elemTy).getFloatSemantics()));
            transposeData(
                data,
                inputType.getShape(),
                initType.getShape(),
                permutation,
                newData);
            newDenseAttr = DenseElementsAttr::get(initType, newData);
        } else {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "Unsupported element type " << elemTy << ", skipping");
            return rewriter.notifyMatchFailure(op, "unsupported element type");
        }

        rewriter.setInsertionPoint(defOp);
        auto newConst = arith::ConstantOp::create(
            rewriter,
            defOp.getLoc(),
            initType,
            newDenseAttr);
        LAKSA_DEBUG(
            llvm::dbgs() << "Created transposed constant of type " << initType);
        rewriter.replaceOp(op, newConst.getResult());
        rewriter.eraseOp(defOp);
        if (auto* initOp = init.getDefiningOp()) rewriter.eraseOp(initOp);
        LAKSA_DEBUG(
            llvm::dbgs() << "Erased original linalg.transpose and operands");
        return success();
    }
};

struct LinalgSoftTransposePass : public linalg::impl::LinalgSoftTransposeBase<
                                     LinalgSoftTransposePass> {
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<TransposeConstant>(patterns.getContext());

        target.addLegalDialect<LinalgDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
        target.addDynamicallyLegalOp<TransposeOp>([](TransposeOp op) {
            auto* defOp = op.getInput().getDefiningOp();
            return !defOp || !isa<arith::ConstantOp>(defOp);
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

std::unique_ptr<Pass> mlir::linalg::createLinalgSoftTransposePass()
{ return std::make_unique<LinalgSoftTransposePass>(); }
