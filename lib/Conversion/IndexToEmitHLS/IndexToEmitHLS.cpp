/// Implementation of IndexToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/IndexToEmitHLS/IndexToEmitHLS.h"

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>

#define DEBUG_TYPE "index-to-emithls"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(llvm::dbgs() << "[index-to-emithls] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTINDEXTOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::emithls;

namespace {
struct ConstantVariable : OpConversionPattern<index::ConstantOp> {

    using OpConversionPattern<index::ConstantOp>::OpConversionPattern;

    ConstantVariable(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<index::ConstantOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        index::ConstantOp op,
        index::ConstantOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<emithls::VariableOp>(
            op,
            op.getType(),
            /*init*/ op.getValueAttr(),
            /*is_const*/ true);
        LAKSA_DEBUG(
            llvm::dbgs() << "Replaced constant with a constant variable.");
        return success();
    }
};
template<typename OpFrom, typename OpTo>
struct ConvertBinaryOp : OpConversionPattern<OpFrom> {
    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertBinaryOp(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpFrom op,
        typename OpFrom::Adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(op, op.getLhs(), op.getRhs());
        LAKSA_DEBUG(
            llvm::dbgs() << "Replaced " << op->getName()
                         << " with equivalent emithls operation.");
        return success();
    }
};
template<typename OpFrom, typename OpTo>
struct ConvertCast : OpConversionPattern<OpFrom> {
    using OpConversionPattern<OpFrom>::OpConversionPattern;

    ConvertCast(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<OpFrom>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        OpFrom op,
        typename OpFrom::Adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<OpTo>(
            op,
            op.getOutput().getType(),
            op.getInput());
        LAKSA_DEBUG(
            llvm::dbgs() << "Replaced " << op->getName()
                         << " with equivalent emithls operation.");
        return success();
    }
};
struct ConvertCompare : OpConversionPattern<index::CmpOp> {
    using OpConversionPattern<index::CmpOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        index::CmpOp op,
        index::CmpOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Create map from index predicate to emithls
        using IP = index::IndexCmpPredicate;
        using EP = emithls::CmpPredicate;
        static const DenseMap<IP, EP> predMap = {
            { IP::EQ, EP::eq},
            { IP::NE, EP::ne},
            {IP::SLT, EP::lt},
            {IP::ULT, EP::lt},
            {IP::SLE, EP::le},
            {IP::ULE, EP::le},
            {IP::SGT, EP::gt},
            {IP::UGT, EP::gt},
            {IP::SGE, EP::ge},
            {IP::UGE, EP::ge},
        };

        auto it = predMap.find(op.getPred());
        if (it == predMap.end())
            return rewriter.notifyMatchFailure(op, "unsupported predicate");
        rewriter.replaceOpWithNewOp<ArithCmpOp>(
            op,
            it->second,
            op.getLhs(),
            op.getRhs());
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Replaced compare operation with equivalent emithls operation.");
        return success();
    }
};
} // namespace

void mlir::populateIndexToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConstantVariable>(typeConverter, patterns.getContext());
    patterns.add<ConvertBinaryOp<index::AddOp, emithls::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::SubOp, emithls::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::MulOp, emithls::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::RemUOp, emithls::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::RemSOp, emithls::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::ShlOp, emithls::ArithShlOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::ShrUOp, emithls::ArithShrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::ShrSOp, emithls::ArithShrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::MaxSOp, emithls::ArithMaxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::MaxUOp, emithls::ArithMaxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::MinSOp, emithls::ArithMinOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::MinUOp, emithls::ArithMinOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::AndOp, emithls::ArithAndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<index::OrOp, emithls::ArithOrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<index::CastUOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<index::CastSOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCompare>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertIndexToEmitHLSPass
        : public impl::ConvertIndexToEmitHLSBase<ConvertIndexToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertIndexToEmitHLSPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateIndexToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<EmitHLSDialect>();
    target.addIllegalDialect<index::IndexDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertIndexToEmitHLSPass()
{ return std::make_unique<ConvertIndexToEmitHLSPass>(); }
