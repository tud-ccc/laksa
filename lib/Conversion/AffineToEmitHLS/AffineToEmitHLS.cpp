/// Implementation of AffineToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/AffineToEmitHLS/AffineToEmitHLS.h"

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "affine-to-emithls"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(llvm::dbgs() << "[affine-to-emithls] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINETOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::emithls;

namespace {

// Recursively retypes an EmitHLS helper op's buffer result (and any upstream
// helper op feeding it) from memref to the equivalent emithls.array, in place.
static void
retypeHelperChainAsArray(ConversionPatternRewriter &rewriter, Value buffer)
{
    auto memrefType = dyn_cast<MemRefType>(buffer.getType());
    if (!memrefType || memrefType.getShape().empty()) return;
    auto arrayType =
        ArrayType::get(memrefType.getShape(), memrefType.getElementType());

    Operation* defOp = buffer.getDefiningOp();
    if (auto linebufOp = dyn_cast_or_null<HelperLineBufferOp>(defOp)) {
        rewriter.modifyOpInPlace(defOp, [&] { buffer.setType(arrayType); });
        retypeHelperChainAsArray(rewriter, linebufOp.getTokenRef());
    } else if (auto windowOp = dyn_cast_or_null<HelperWindowOp>(defOp)) {
        rewriter.modifyOpInPlace(defOp, [&] { buffer.setType(arrayType); });
        retypeHelperChainAsArray(rewriter, windowOp.getBufRef());
    }
}

// Materializes an affine.load/affine.store's memref operand as an emithls.array
// value of type arrayType.
static Value materializeArray(
    ConversionPatternRewriter &rewriter,
    Location loc,
    Value memref,
    ArrayType arrayType)
{
    if (isa_and_nonnull<HelperLineBufferOp, HelperWindowOp>(
            memref.getDefiningOp())) {
        retypeHelperChainAsArray(rewriter, memref);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Retyped helper op chain to " << arrayType << " in place.");
        return memref;
    }
    LAKSA_DEBUG(llvm::dbgs() << "Materialize using unrealized cast.");
    return UnrealizedConversionCastOp::create(rewriter, loc, arrayType, memref)
        .getResult(0);
}

// Builds the emithls.variable based on the used memref.global
static Value materializeGlobalVariable(
    ConversionPatternRewriter &rewriter,
    memref::GetGlobalOp getGlobalOp,
    ArrayType arrayType)
{
    auto globalOp = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobalOp,
        getGlobalOp.getNameAttr());

    Attribute initNumber;
    bool isConst = false;
    if (globalOp) {
        if (auto constInit = globalOp.getConstantInitValue()) {
            if (auto denseInit = dyn_cast<DenseElementsAttr>(constInit)) {
                initNumber = denseInit.reshape(arrayType);
                isConst = true;
            }
        }
    }
    LAKSA_DEBUG(
        llvm::dbgs() << "Materialize memref.get_global @"
                     << getGlobalOp.getName() << " as emithls.variable.");

    // Build the variable where get_global was
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(getGlobalOp);
    return VariableOp::create(
        rewriter,
        getGlobalOp.getLoc(),
        arrayType,
        initNumber,
        isConst);
}

struct ForLoopTransformation : OpConversionPattern<affine::AffineForOp> {
    using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

    ForLoopTransformation(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<affine::AffineForOp>(typeConverter, context)
    { setHasBoundedRewriteRecursion(); };

    LogicalResult matchAndRewrite(
        affine::AffineForOp op,
        affine::AffineForOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto lb = op.getConstantLowerBound();
        auto ub = op.getConstantUpperBound();
        auto step = op.getStepAsInt();
        LAKSA_DEBUG(
            llvm::dbgs() << "Found for loop from " << lb << " to " << ub
                         << " with step " << step << " at " << loc);

        auto newForOp = ForOp::create(
            rewriter,
            loc,
            lb,
            ub,
            step,
            [&](OpBuilder &forBuilder, Location, ValueRange forArgs) {
                IRMapping mapper;
                for (auto [oldArg, newArg] :
                     llvm::zip(op.getBody()->getArguments(), forArgs)) {
                    mapper.map(oldArg, newArg);
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Map " << oldArg << " to " << newArg);
                }

                for (auto &opi : op.getBody()->getOperations()) {
                    if (isa<affine::AffineYieldOp>(opi)) break;
                    forBuilder.clone(opi, mapper);
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Copying " << opi.getName()
                                     << " into new for loop");
                }
            });
        rewriter.replaceOp(op, newForOp);

        return success();
    }
};
struct IfTransformation : OpConversionPattern<affine::AffineIfOp> {
    using OpConversionPattern<affine::AffineIfOp>::OpConversionPattern;

    IfTransformation(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<affine::AffineIfOp>(typeConverter, context)
    { setHasBoundedRewriteRecursion(); };

    LogicalResult matchAndRewrite(
        affine::AffineIfOp op,
        affine::AffineIfOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (op.getNumResults() != 0)
            return rewriter.notifyMatchFailure(
                op,
                "affine.if with results is not supported");

        auto loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Found affine.if with " << op.getIntegerSet().getNumConstraints()
            << " constraint(s) at " << loc);

        auto exprOp = ExpressionOp::create(
            rewriter,
            loc,
            rewriter.getI1Type(),
            [&](OpBuilder &exprBuilder, Location) {
                auto integerSet = op.getIntegerSet();
                SmallVector<Value> operands(op.getOperands());
                ArrayRef<Value> operandsRef(operands);
                unsigned numDims = integerSet.getNumDims();

                Value zero =
                    arith::ConstantIndexOp::create(exprBuilder, loc, 0);
                SmallVector<Value> cmpVals;
                for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e;
                     ++i) {
                    Value affineResult = affine::expandAffineExpr(
                        exprBuilder,
                        loc,
                        integerSet.getConstraint(i),
                        operandsRef.take_front(numDims),
                        operandsRef.drop_front(numDims));
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Expanded the affine constraint to "
                                        "arith operation.");
                    auto predicate = integerSet.isEq(i) ? CmpPredicate::eq
                                                        : CmpPredicate::ge;
                    cmpVals.push_back(
                        ArithCmpOp::create(
                            exprBuilder,
                            loc,
                            predicate,
                            affineResult,
                            zero)
                            .getResult());
                    LAKSA_DEBUG(
                        llvm::dbgs() << "  Created comparison operation "
                                        "according to constraint.");
                }
                Value cond = cmpVals.size() > 1 ? ArithLogicalAndOp::create(
                                                      exprBuilder,
                                                      loc,
                                                      exprBuilder.getI1Type(),
                                                      cmpVals)
                                                      .getResult()
                                                : cmpVals.front();
                LAKSA_DEBUG(
                    llvm::dbgs() << "  Insert logical and operation to combine "
                                    "the affine constraint values.");
                YieldOp::create(exprBuilder, loc, cond);
            });

        auto cloneBlockBody = [](Block* srcBlock, OpBuilder &builder) {
            IRMapping mapper;
            for (auto &opi : srcBlock->getOperations()) {
                if (isa<affine::AffineYieldOp>(opi)) break;
                builder.clone(opi, mapper);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "  Copying " << opi.getName() << " into new if op");
            }
        };

        Block* thenBlock = op.getThenBlock();
        IfOp newIf;
        if (op.hasElse()) {
            Block* elseBlock = op.getElseBlock();
            newIf = IfOp::create(
                rewriter,
                loc,
                exprOp.getResult(),
                [&](OpBuilder &builder, Location) {
                    cloneBlockBody(thenBlock, builder);
                },
                [&](OpBuilder &builder, Location) {
                    cloneBlockBody(elseBlock, builder);
                });
        } else {
            newIf = IfOp::create(
                rewriter,
                loc,
                exprOp.getResult(),
                [&](OpBuilder &builder, Location) {
                    cloneBlockBody(thenBlock, builder);
                });
        }
        rewriter.replaceOp(op, newIf);

        return success();
    }
};
struct EraseYield : OpConversionPattern<affine::AffineYieldOp> {
    using OpConversionPattern<affine::AffineYieldOp>::OpConversionPattern;

    EraseYield(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<affine::AffineYieldOp>(
                  typeConverter,
                  context) {};

    LogicalResult matchAndRewrite(
        affine::AffineYieldOp op,
        affine::AffineYieldOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (op->getNumOperands() != 0)
            return rewriter.notifyMatchFailure(
                op.getLoc(),
                "Expect zero operand.");
        rewriter.eraseOp(op);
        return success();
    }
};
struct LoadTransformation : OpConversionPattern<affine::AffineLoadOp> {
    using OpConversionPattern<affine::AffineLoadOp>::OpConversionPattern;

    LoadTransformation(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<affine::AffineLoadOp>(
                  typeConverter,
                  context) {};

    LogicalResult matchAndRewrite(
        affine::AffineLoadOp op,
        affine::AffineLoadOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto indices = affine::expandAffineMap(
            rewriter,
            loc,
            op.getAffineMap(),
            op.getMapOperands());
        if (!indices)
            return rewriter.notifyMatchFailure(
                op,
                "failed to expand affine map into indices");
        LAKSA_DEBUG(llvm::dbgs() << "Converting affine.load at " << loc);

        auto memrefType = op.getMemRefType();
        auto arrayType =
            ArrayType::get(memrefType.getShape(), memrefType.getElementType());

        auto getGlobalOp = op.getMemref().getDefiningOp<memref::GetGlobalOp>();
        Value array =
            getGlobalOp
                ? materializeGlobalVariable(rewriter, getGlobalOp, arrayType)
                : materializeArray(rewriter, loc, op.getMemref(), arrayType);

        rewriter.replaceOpWithNewOp<ArrayReadOp>(op, array, *indices);

        return success();
    }
};
} // namespace

void mlir::populateAffineToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<ForLoopTransformation>(typeConverter, patterns.getContext());
    patterns.add<IfTransformation>(typeConverter, patterns.getContext());
    patterns.add<EraseYield>(typeConverter, patterns.getContext());
    patterns.add<LoadTransformation>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertAffineToEmitHLSPass
        : public impl::ConvertAffineToEmitHLSBase<ConvertAffineToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertAffineToEmitHLSPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateAffineToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<EmitHLSDialect>();
    target.addIllegalDialect<affine::AffineDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
        return;
    }

    // Clean the dead global and get_global operations after the patterns
    IRRewriter rewriter(&getContext());
    SmallVector<memref::GetGlobalOp> deadGetGlobals;
    getOperation()->walk([&](memref::GetGlobalOp op) {
        if (op->use_empty()) deadGetGlobals.push_back(op);
    });
    SmallVector<memref::GlobalOp> maybeDeadGlobals;
    for (auto op : deadGetGlobals) {
        if (auto globalOp =
                SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
                    op,
                    op.getNameAttr()))
            maybeDeadGlobals.push_back(globalOp);
        rewriter.eraseOp(op);
        LAKSA_DEBUG(
            llvm::dbgs() << "Erase the dead get_global at " << op.getLoc());
    }
    for (auto globalOp : maybeDeadGlobals)
        if (SymbolTable::symbolKnownUseEmpty(globalOp, getOperation())) {
            rewriter.eraseOp(globalOp);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "Erase the dead global at " << globalOp.getLoc());
        }
}

std::unique_ptr<Pass> mlir::createConvertAffineToEmitHLSPass()
{ return std::make_unique<ConvertAffineToEmitHLSPass>(); }
