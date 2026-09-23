/// Implementation of ArithToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/ArithToEmitHLS/ArithToEmitHLS.h"

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "arith-to-emithls"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(llvm::dbgs() << "[arith-to-emithls] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::emithls;

namespace {
struct ConstantVariable : OpConversionPattern<arith::ConstantOp> {

    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

    ConstantVariable(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::ConstantOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        arith::ConstantOp op,
        arith::ConstantOpAdaptor,
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
            op.getOut().getType(),
            op.getIn());
        LAKSA_DEBUG(
            llvm::dbgs() << "Replaced " << op->getName()
                         << " with equivalent emithls operation.");
        return success();
    }
};
struct ConvertCompare : OpConversionPattern<arith::CmpIOp> {
    using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        arith::CmpIOp op,
        arith::CmpIOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        // Create map from arith predicate to emithls
        using AP = arith::CmpIPredicate;
        using EP = emithls::CmpPredicate;
        static const DenseMap<AP, EP> predMap = {
            { AP::eq, EP::eq},
            { AP::ne, EP::ne},
            {AP::slt, EP::lt},
            {AP::ult, EP::lt},
            {AP::sle, EP::le},
            {AP::ule, EP::le},
            {AP::sgt, EP::gt},
            {AP::ugt, EP::gt},
            {AP::sge, EP::ge},
            {AP::uge, EP::ge},
        };

        auto it = predMap.find(op.getPredicate());
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
struct ConvertSelect : OpConversionPattern<arith::SelectOp> {
    using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

    ConvertSelect(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<arith::SelectOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        arith::SelectOp op,
        arith::SelectOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<emithls::ArithSelectOp>(
            op,
            op.getCondition(),
            op.getTrueValue(),
            op.getFalseValue());
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Replaced select operation with equivalent emithls operation.");
        return success();
    }
};
} // namespace

void mlir::populateArithToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ConstantVariable>(typeConverter, patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::AddIOp, emithls::ArithAddOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::SubIOp, emithls::ArithSubOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::MulIOp, emithls::ArithMulOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::RemUIOp, emithls::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::RemSIOp, emithls::ArithRemOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::ShLIOp, emithls::ArithShlOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::ShRUIOp, emithls::ArithShrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::ShRSIOp, emithls::ArithShrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::MaxSIOp, emithls::ArithMaxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::MaxUIOp, emithls::ArithMaxOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::MinSIOp, emithls::ArithMinOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::MinUIOp, emithls::ArithMinOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::AndIOp, emithls::ArithAndOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertBinaryOp<arith::OrIOp, emithls::ArithOrOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<arith::IndexCastOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<arith::IndexCastUIOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<arith::ExtSIOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<arith::ExtUIOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCast<arith::TruncIOp, emithls::ArithCastOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<ConvertCompare>(typeConverter, patterns.getContext());
    patterns.add<ConvertSelect>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertArithToEmitHLSPass
        : public impl::ConvertArithToEmitHLSBase<ConvertArithToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertArithToEmitHLSPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateArithToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<EmitHLSDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertArithToEmitHLSPass()
{ return std::make_unique<ConvertArithToEmitHLSPass>(); }
