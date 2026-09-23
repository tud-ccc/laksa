// Implementation of ResolveHelpers transform pass.
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "emithls-resolve-helpers"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-resolve-helpers] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSRESOLVEHELPERS
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

// Bookkeeping handed off from ResolveLineBuffer to ResolveWindow for buffers
// that still need work once the window is known: a 1-D buffer's loop body is
// left empty here, and a multi-dimensional buffer's token is loaded but not yet
// stored. Pure line buffers are fully resolved on the spot and never get an
// entry here, since no HelperWindowOp ever references them.
struct LineBufferInfo {
    bool isOneD = false;
    // The VariableOp result that replaced the original buf_ref
    Value bufVar;
    // The original token_ref operand.
    Value tokenRef;
    // The original num_line attribute value.
    int32_t numLine = 0;
    // Loop nest created at the line buffer's original position, outer to inner,
    // one level per num_chan entry.
    SmallVector<ForOp> loops;
    // For the multi-dimensional case: the token value already loaded inside the
    // innermost loop, waiting to be stored by ResolveWindow pattern.
    Value loadedToken;
    // The original `keep` index operand (the column position); only set for
    // the multi-dimensional case.
    Value keepIndex;
};
using LineBufferInfoMap = DenseMap<Value, LineBufferInfo>;

// Prints a ForOp's bounds as `[lb, ub, step)`, dropping `step` when it's 1.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ForOp forOp)
{
    os << "[" << forOp.getLowerBound().getSExtValue() << ", "
       << forOp.getUpperBound().getSExtValue();
    if (auto step = forOp.getStep().getSExtValue(); step != 1)
        os << ", " << step;
    os << ")";
    return os;
}
// Prints a loop nest as `[[lb, ub, step), ...]`, outer to inner.
static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, ArrayRef<ForOp> loops)
{
    os << "[";
    for (auto [i, forOp] : llvm::enumerate(loops)) {
        if (i) os << ", ";
        os << forOp;
    }
    os << "]";
    return os;
}
// Print a LineBufferInfo struct
static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const LineBufferInfo &info)
{
    os << "LineBufferInfo: isOneD=" << info.isOneD << ", bufVar=" << info.bufVar
       << ", tokenRef=" << info.tokenRef << ", numLine=" << info.numLine
       << ", loops=" << ArrayRef<ForOp>(info.loops);
    if (info.loadedToken) os << ", loadedToken=" << info.loadedToken;
    return os;
}

struct ResolveLineBuffer : OpRewritePattern<HelperLineBufferOp> {
    ResolveLineBuffer(MLIRContext* context, LineBufferInfoMap &infos)
            : OpRewritePattern<HelperLineBufferOp>(context),
              infos(infos) {};

    LineBufferInfoMap &infos;

    LogicalResult matchAndRewrite(
        HelperLineBufferOp op,
        PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting line buffer at " << loc);
        auto parentFunc = op->getParentOfType<FuncOp>();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Found parent function " << parentFunc.getSymName());
        auto resultType = dyn_cast<ArrayType>(op.getResult().getType());
        if (!resultType)
            return rewriter.notifyMatchFailure(
                loc,
                "expect array type at this phase");
        auto resultShape = resultType.getShape();
        rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
        auto linebufVar = VariableOp::create(
            rewriter,
            loc,
            resultType,
            /*initValue*/ Value{});

        rewriter.setInsertionPoint(op);

        enum class Kind { Pure, OneD, MultiDim };
        Kind kind;
        SmallVector<int32_t> loopUpperBounds;
        if (op.getNumLine() == 0) {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  This is line buffer without usage for sliding window");
            if (resultShape.size() != 1)
                return rewriter.notifyMatchFailure(
                    loc,
                    "expect pure line buffer only has one dimension");
            kind = Kind::Pure;
            loopUpperBounds.push_back(resultShape[0]);
        } else if (!op.getIndex()) {
            LAKSA_DEBUG(llvm::dbgs() << "  This is a 1-D line buffer");
            if (resultShape.size() != 1)
                return rewriter.notifyMatchFailure(
                    loc,
                    "expect 1-D line buffer only has one dimension");
            kind = Kind::OneD;
            loopUpperBounds.push_back(resultShape[0]);
        } else {
            LAKSA_DEBUG(
                llvm::dbgs() << "  This is a multi demensional line buffer");
            kind = Kind::MultiDim;
            for (auto numAttr : op.getNumChan()) {
                auto numIntAttr = cast<IntegerAttr>(numAttr);
                loopUpperBounds.push_back(numIntAttr.getInt());
            }
        }

        SmallVector<ForOp> loops;
        for (auto dimSize : loopUpperBounds) {
            auto forOp = ForOp::create(rewriter, loc, 0, dimSize);
            rewriter.setInsertionPointToStart(&forOp.getBody().front());
            loops.push_back(forOp);
            LAKSA_DEBUG(
                llvm::dbgs() << "  Created a loop of [0, " << dimSize << ")");
        }

        auto tokenRef = op.getTokenRef();
        auto loadToken = [&](ValueRange indices) -> Value {
            if (isa<MemRefType>(tokenRef.getType()))
                return memref::LoadOp::create(rewriter, loc, tokenRef, indices);
            return ArrayReadOp::create(
                rewriter,
                loc,
                resultType.getElementType(),
                tokenRef,
                indices);
        };

        switch (kind) {
        case Kind::Pure:
        {
            // No HelperWindowOp ever consumes this buffer, so read the token
            // and store it into the array right away.
            Value idx = loops.front().getInductionVariable();
            Value token = loadToken(idx);
            UpdateOp::create(
                rewriter,
                loc,
                linebufVar.getVariable(),
                token,
                SmallVector<Value>{idx});
            break;
        }
        case Kind::OneD:
        {
            // Leave the loop body empty; ResolveWindow fills it in once it
            // knows how the sliding window consumes this buffer.
            LineBufferInfo info;
            info.isOneD = true;
            info.bufVar = linebufVar.getVariable();
            info.tokenRef = tokenRef;
            info.numLine = op.getNumLine();
            info.loops = loops;
            infos[linebufVar.getVariable()] = std::move(info);
            break;
        }
        case Kind::MultiDim:
        {
            // Load the token now, but leave storing it into the line-buffer
            // array to ResolveWindow.
            SmallVector<Value> chanIndices;
            for (auto &forOp : loops)
                chanIndices.push_back(forOp.getInductionVariable());
            Value token = loadToken(chanIndices);
            LAKSA_DEBUG(llvm::dbgs() << "  Loaded token " << token);

            LineBufferInfo info;
            info.isOneD = false;
            info.bufVar = linebufVar.getVariable();
            info.tokenRef = tokenRef;
            info.numLine = op.getNumLine();
            info.loops = loops;
            info.loadedToken = token;
            info.keepIndex = op.getIndex();
            infos[linebufVar.getVariable()] = std::move(info);
            break;
        }
        }

        rewriter.replaceOp(op, linebufVar);
        return success();
    }
};
// Builds a zero-valued attribute usable as a VariableOp's const initNumber
// for a line-buffer element type.
static Attribute buildZeroAttr(Builder &builder, Type type)
{
    if (isa<IndexType>(type)) return builder.getIndexAttr(0);
    if (auto intType = dyn_cast<IntegerType>(type))
        return builder.getIntegerAttr(intType, 0);
    if (auto floatType = dyn_cast<FloatType>(type))
        return builder.getFloatAttr(floatType, 0.0);
    llvm_unreachable("unsupported line-buffer element type for zero-fill");
}

// Recovers the original scalar value behind a rank-0 token_ref.
static Value unwrapScalarToken(OpBuilder &builder, Location loc, Value tokenRef)
{
    if (auto castOp = tokenRef.getDefiningOp<UnrealizedConversionCastOp>())
        if (castOp.getInputs().size() == 1) return castOp.getInputs().front();
    if (isa<MemRefType>(tokenRef.getType()))
        return memref::LoadOp::create(builder, loc, tokenRef, ValueRange{});
    auto arrayType = cast<ArrayType>(tokenRef.getType());
    return ArrayReadOp::create(
        builder,
        loc,
        arrayType.getElementType(),
        tokenRef,
        ValueRange{});
}

struct ResolveWindow : OpConversionPattern<HelperWindowOp> {
    ResolveWindow(MLIRContext* context, LineBufferInfoMap &infos)
            : OpConversionPattern<HelperWindowOp>(context),
              infos(infos) {};

    LineBufferInfoMap &infos;

    LogicalResult matchAndRewrite(
        HelperWindowOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting window at " << loc);

        auto it = infos.find(adaptor.getBufRef());
        if (it == infos.end()) {
            return rewriter.notifyMatchFailure(
                loc,
                "expect line buffer info if there is a window helper");
        }
        LineBufferInfo &info = it->second;
        LAKSA_DEBUG(llvm::dbgs() << "  Found " << info);

        // Builds a const index VariableOp
        auto constIndex = [](OpBuilder &b, Location l, int64_t value) {
            return VariableOp::create(
                       b,
                       l,
                       b.getIndexType(),
                       b.getIndexAttr(value),
                       /*is_const=*/true)
                .getVariable();
        };

        // Every HelperWindowOp result gets replaced by a VariableOp created
        // at the top of the function, mirroring what ResolveLineBuffer does
        // for buf_ref.
        auto parentFunc = op->getParentOfType<FuncOp>();
        auto windowResultType = op.getResult().getType();
        rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
        auto windowVar = VariableOp::create(
            rewriter,
            loc,
            windowResultType,
            /*initValue*/ Value{});
        rewriter.setInsertionPoint(op);

        if (info.isOneD) {
            // 1-D case: windowVar is itself the persistent shift register;
            // info.bufVar goes unused. Reuse the (still-empty) loop
            // ResolveLineBuffer already created. The freshly read token always
            // becomes the newest (last) slot, appended after the loop.
            LAKSA_DEBUG(llvm::dbgs() << "  This is a 1-D window");
            auto shiftLoop = info.loops.back();
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    Reusing existing shift loop " << shiftLoop);
            rewriter.setInsertionPoint(shiftLoop);

            Value zeroIndex = constIndex(rewriter, loc, 0);
            Value rowIndex = adaptor.getIndices().front();
            Value isFirstRow = ExpressionOp::create(
                                   rewriter,
                                   loc,
                                   rewriter.getI1Type(),
                                   [&](OpBuilder &b, Location l) {
                                       Value cmp = ArithCmpOp::create(
                                           b,
                                           l,
                                           b.getI1Type(),
                                           CmpPredicate::eq,
                                           rowIndex,
                                           zeroIndex);
                                       YieldOp::create(b, l, cmp);
                                   })
                                   .getResult();
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    Created index check expression before shift loop");

            rewriter.setInsertionPointToStart(&shiftLoop.getBody().front());
            LAKSA_DEBUG(llvm::dbgs() << "    Inside shift loop");
            Value slot = shiftLoop.getInductionVariable();
            auto elementType =
                cast<ArrayType>(windowVar.getVariable().getType())
                    .getElementType();

            IfOp::create(
                rewriter,
                loc,
                isFirstRow,
                [&](OpBuilder &b, Location l) {
                    Value zeroElem = VariableOp::create(
                                         b,
                                         l,
                                         elementType,
                                         buildZeroAttr(b, elementType),
                                         /*is_const=*/true)
                                         .getVariable();
                    UpdateOp::create(
                        b,
                        l,
                        windowVar.getVariable(),
                        zeroElem,
                        SmallVector<Value>{slot});
                },
                [&](OpBuilder &b, Location l) {
                    Value one = constIndex(b, l, 1);
                    Value nextSlot = ExpressionOp::create(
                                         b,
                                         l,
                                         b.getIndexType(),
                                         [&](OpBuilder &b2, Location l2) {
                                             Value sum = ArithAddOp::create(
                                                 b2,
                                                 l2,
                                                 b2.getIndexType(),
                                                 slot,
                                                 one);
                                             YieldOp::create(b2, l2, sum);
                                         })
                                         .getResult();
                    UpdateOp::create(
                        b,
                        l,
                        windowVar.getVariable(),
                        windowVar.getVariable(),
                        SmallVector<Value>{slot},
                        SmallVector<Value>{nextSlot});
                });
            LAKSA_DEBUG(llvm::dbgs() << "      Created window update logic");

            rewriter.setInsertionPointAfter(shiftLoop);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    Appending fresh token at slot " << info.numLine);
            Value newestSlot = constIndex(rewriter, loc, info.numLine);
            Value freshToken = unwrapScalarToken(rewriter, loc, info.tokenRef);
            UpdateOp::create(
                rewriter,
                loc,
                windowVar.getVariable(),
                freshToken,
                SmallVector<Value>{newestSlot});

            rewriter.replaceOp(op, windowVar);
            return success();
        }

        // Multi-dimensional case, part 1: prime the line-buffer slots that a
        // real token hasn't reached yet with zeros, guarded on being at the
        // very first row of the image (the window's first index is 0);
        // otherwise shift the buffer left by one column. The condition
        // doesn't depend on the two new loops below, so it's computed once,
        // outside them.
        LAKSA_DEBUG(llvm::dbgs() << "  This is a multi-dimensional window");
        LAKSA_DEBUG(llvm::dbgs() << "    Priming/shifting line buffer columns");
        rewriter.setInsertionPointToEnd(&info.loops.back().getBody().front());

        Value rowIndex = adaptor.getIndices()[0];
        Value colIndex = adaptor.getIndices()[1];

        Value zeroIndex = constIndex(rewriter, loc, 0);
        Value shiftCond = ExpressionOp::create(
                              rewriter,
                              loc,
                              rewriter.getI1Type(),
                              [&](OpBuilder &b, Location l) {
                                  Value cmp = ArithCmpOp::create(
                                      b,
                                      l,
                                      b.getI1Type(),
                                      CmpPredicate::eq,
                                      rowIndex,
                                      zeroIndex);
                                  YieldOp::create(b, l, cmp);
                              })
                              .getResult();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "    Created index check expression before shift loop");

        auto outerLoop = ForOp::create(rewriter, loc, 0, info.numLine + 1);
        LAKSA_DEBUG(llvm::dbgs() << "    Created loop " << outerLoop);
        rewriter.setInsertionPointToStart(&outerLoop.getBody().front());
        auto innerLoop = ForOp::create(rewriter, loc, 0, info.numLine);
        LAKSA_DEBUG(llvm::dbgs() << "    Created loop " << innerLoop);
        rewriter.setInsertionPointToStart(&innerLoop.getBody().front());

        auto buildShiftIndices = [&] {
            SmallVector<Value> indices;
            for (auto &loop : info.loops)
                indices.push_back(loop.getInductionVariable());
            indices.push_back(outerLoop.getInductionVariable());
            indices.push_back(innerLoop.getInductionVariable());
            return indices;
        };

        IfOp::create(
            rewriter,
            loc,
            shiftCond,
            [&](OpBuilder &b, Location l) {
                // Nothing has arrived yet at the very first row: zero-fill.
                auto elementType =
                    cast<ArrayType>(windowVar.getVariable().getType())
                        .getElementType();
                Value zeroElem = VariableOp::create(
                                     b,
                                     l,
                                     elementType,
                                     buildZeroAttr(b, elementType),
                                     /*is_const=*/true)
                                     .getVariable();
                UpdateOp::create(
                    b,
                    l,
                    windowVar.getVariable(),
                    zeroElem,
                    buildShiftIndices());
            },
            [&](OpBuilder &b, Location l) {
                // Otherwise shift the window left by one column: slot idx4
                // takes what used to be at idx4 + 1.
                Value one = constIndex(b, l, 1);
                Value shiftedCol = ExpressionOp::create(
                                       b,
                                       l,
                                       b.getIndexType(),
                                       [&](OpBuilder &b2, Location l2) {
                                           Value sum = ArithAddOp::create(
                                               b2,
                                               l2,
                                               b2.getIndexType(),
                                               innerLoop.getInductionVariable(),
                                               one);
                                           YieldOp::create(b2, l2, sum);
                                       })
                                       .getResult();

                SmallVector<Value> targetIndices = buildShiftIndices();
                SmallVector<Value> sourceIndices = targetIndices;
                sourceIndices.back() = shiftedCol;

                UpdateOp::create(
                    b,
                    l,
                    windowVar.getVariable(),
                    windowVar.getVariable(),
                    targetIndices,
                    sourceIndices);
            });
        LAKSA_DEBUG(llvm::dbgs() << "      Created window update logic");

        // Multi-dimensional case, part 2: populate the window's newest
        // column for each of its numLine historical rows from the line
        // buffer.
        LAKSA_DEBUG(
            llvm::dbgs() << "    Populating window rows from line buffer");
        rewriter.setInsertionPointAfter(outerLoop);

        SmallVector<Value> chanIndices;
        for (auto &loop : info.loops)
            chanIndices.push_back(loop.getInductionVariable());

        auto elementType =
            cast<ArrayType>(windowVar.getVariable().getType()).getElementType();

        Value numLineConst = constIndex(rewriter, loc, info.numLine);

        for (int32_t line = 0; line < info.numLine; ++line) {
            int64_t threshold = info.numLine - line;
            LAKSA_DEBUG(
                llvm::dbgs()
                << "      Row " << line << " uses threshold " << threshold);
            Value thresholdConst = constIndex(rewriter, loc, threshold);
            Value lineConst = constIndex(rewriter, loc, line);

            Value cond = ExpressionOp::create(
                             rewriter,
                             loc,
                             rewriter.getI1Type(),
                             [&](OpBuilder &b, Location l) {
                                 Value cmp = ArithCmpOp::create(
                                     b,
                                     l,
                                     b.getI1Type(),
                                     CmpPredicate::ge,
                                     rowIndex,
                                     thresholdConst);
                                 YieldOp::create(b, l, cmp);
                             })
                             .getResult();

            SmallVector<Value> writeIndices = chanIndices;
            writeIndices.push_back(lineConst);
            writeIndices.push_back(numLineConst);

            IfOp::create(
                rewriter,
                loc,
                cond,
                [&](OpBuilder &b, Location l) {
                    Value lineSlot = ExpressionOp::create(
                                         b,
                                         l,
                                         b.getIndexType(),
                                         [&](OpBuilder &b2, Location l2) {
                                             Value diff = ArithSubOp::create(
                                                 b2,
                                                 l2,
                                                 b2.getIndexType(),
                                                 rowIndex,
                                                 thresholdConst);
                                             Value rem = ArithRemOp::create(
                                                 b2,
                                                 l2,
                                                 b2.getIndexType(),
                                                 diff,
                                                 numLineConst);
                                             YieldOp::create(b2, l2, rem);
                                         })
                                         .getResult();

                    SmallVector<Value> readIndices = chanIndices;
                    readIndices.push_back(lineSlot);
                    readIndices.push_back(colIndex);

                    UpdateOp::create(
                        b,
                        l,
                        windowVar.getVariable(),
                        info.bufVar,
                        writeIndices,
                        readIndices);
                },
                [&](OpBuilder &b, Location l) {
                    Value zeroElem = VariableOp::create(
                                         b,
                                         l,
                                         elementType,
                                         buildZeroAttr(b, elementType),
                                         /*is_const=*/true)
                                         .getVariable();
                    UpdateOp::create(
                        b,
                        l,
                        windowVar.getVariable(),
                        zeroElem,
                        writeIndices);
                });
        }

        // Multi-dimensional case, part 3: the current row's token, already
        // loaded by ResolveLineBuffer, becomes both the window's newest
        // corner (row numLine, column numLine) and the line buffer's entry
        // for this row's circular slot (idx0 % numLine) at the current
        // column (keep index).
        LAKSA_DEBUG(
            llvm::dbgs()
            << "    Writing current row into window corner and line buffer");
        Value currentSlot = ExpressionOp::create(
                                rewriter,
                                loc,
                                rewriter.getIndexType(),
                                [&](OpBuilder &b, Location l) {
                                    Value rem = ArithRemOp::create(
                                        b,
                                        l,
                                        b.getIndexType(),
                                        rowIndex,
                                        numLineConst);
                                    YieldOp::create(b, l, rem);
                                })
                                .getResult();

        SmallVector<Value> windowCornerIndices = chanIndices;
        windowCornerIndices.push_back(numLineConst);
        windowCornerIndices.push_back(numLineConst);
        UpdateOp::create(
            rewriter,
            loc,
            windowVar.getVariable(),
            info.loadedToken,
            windowCornerIndices);

        SmallVector<Value> bufCurrentIndices = chanIndices;
        bufCurrentIndices.push_back(currentSlot);
        bufCurrentIndices.push_back(info.keepIndex);
        UpdateOp::create(
            rewriter,
            loc,
            info.bufVar,
            info.loadedToken,
            bufCurrentIndices);

        rewriter.replaceOp(op, windowVar);
        return success();
    }
};
struct ResolveAccumulate : OpRewritePattern<HelperAccumulateOp> {
    ResolveAccumulate(MLIRContext* context)
            : OpRewritePattern<HelperAccumulateOp>(context) {};

    LogicalResult matchAndRewrite(
        HelperAccumulateOp op,
        PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting accumulate at " << loc);

        Value accuRef = op.getAccuRef();
        auto allocOp = accuRef.getDefiningOp<memref::AllocOp>();
        if (!allocOp)
            return rewriter.notifyMatchFailure(
                loc,
                "expect accu_ref to be defined by a memref.alloc");
        LAKSA_DEBUG(llvm::dbgs() << "  Found backing alloc " << accuRef);

        // The alloc's accumulated result is later read back out through a
        // memref.load, itself streamed out.
        memref::LoadOp loadOp;
        for (Operation* user : accuRef.getUsers())
            if (auto load = dyn_cast<memref::LoadOp>(user)) {
                loadOp = load;
                break;
            }
        if (!loadOp)
            return rewriter.notifyMatchFailure(
                loc,
                "expect a memref.load reading the accumulate result back out");

        StreamWriteOp streamWriteOp;
        for (Operation* user : loadOp.getResult().getUsers())
            if (auto write = dyn_cast<StreamWriteOp>(user)) {
                streamWriteOp = write;
                break;
            }
        if (!streamWriteOp)
            return rewriter.notifyMatchFailure(
                loc,
                "expect the accumulate result to be streamed out");
        LAKSA_DEBUG(
            llvm::dbgs() << "  Found write-out site "
                         << streamWriteOp.getOperation()->getName());

        Type valueType = op.getValue().getType();
        Type accuElementType =
            cast<MemRefType>(accuRef.getType()).getElementType();
        SmallVector<Value> outputIndices(
            op.getIndices().begin(),
            op.getIndices().end());

        // Use the alloc's fill_with attribute as the accumulator's initial
        // value, otherwise use a zero attribute
        Attribute initAttr;
        if (auto fillAttr = allocOp->getAttrOfType<TypedAttr>("fill_with");
            fillAttr && fillAttr.getType() == valueType) {
            initAttr = fillAttr;
            LAKSA_DEBUG(
                llvm::dbgs() << "  Reusing fill_with initializer " << initAttr);
        } else {
            initAttr = buildZeroAttr(rewriter, valueType);
            LAKSA_DEBUG(llvm::dbgs() << "  Defaulting to zero initializer");
        }

        // Turn the alloc into the scalar accumulator variable, reset every time
        // control reaches its original position.
        rewriter.setInsertionPoint(allocOp);
        auto accVar = VariableOp::create(
            rewriter,
            loc,
            valueType,
            initAttr,
            /*is_const=*/false);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created accumulator variable " << accVar.getVariable());

        // Fold the accumulate into a compound-assignment op (arith.fused)
        rewriter.setInsertionPoint(op);
        ArithFusedOp::create(
            rewriter,
            loc,
            op.getOpCode(),
            accVar.getVariable(),
            op.getValue());
        LAKSA_DEBUG(
            llvm::dbgs() << "  Fused into arith.fused "
                         << stringifyFusedOperator(op.getOpCode()));
        rewriter.eraseOp(op);

        // Replace whatever reads the alloc back out with a single write of the
        // accumulator, using the accumulate's own output indices instead of
        // that loop's induction variables.
        Block* allocBlock = allocOp->getBlock();
        Operation* writeSite = streamWriteOp.getOperation();
        while (writeSite->getBlock() != allocBlock)
            writeSite = writeSite->getParentOp();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Replacing write-out site " << writeSite->getName());
        rewriter.setInsertionPoint(writeSite);

        // Cast back down to the alloc's element type if accumulation ran at a
        // different precision.
        Value result = accVar.getVariable();
        if (accuElementType != valueType) {
            result = ExpressionOp::create(
                         rewriter,
                         loc,
                         accuElementType,
                         [&](OpBuilder &b, Location l) {
                             Value casted = ArithCastOp::create(
                                 b,
                                 l,
                                 accuElementType,
                                 accVar.getVariable());
                             YieldOp::create(b, l, casted);
                         })
                         .getResult();
            LAKSA_DEBUG(
                llvm::dbgs() << "  Cast accumulator down to " << accuElementType
                             << " via expr");
        }
        StreamWriteOp::create(
            rewriter,
            loc,
            result,
            streamWriteOp.getStream(),
            outputIndices);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Write the accumulate variable into output port at index ");

        if (loadOp->getParentOp() != writeSite) rewriter.eraseOp(loadOp);
        rewriter.eraseOp(writeSite);
        rewriter.eraseOp(allocOp);

        return success();
    }
};

// A memref.alloc with a fill_with attribute that's never written to is a
// filler buffer MemRefPadToLAKSALoops created for a memref-shaped port token
// with nowhere else to carry a padding constant. Every load from it yields
// that same constant, so the whole buffer collapses into one shared
// emithls.variable as const.
struct ResolveFillerAlloc : OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocOp op, PatternRewriter &rewriter)
        const override
    {
        auto fillAttr = op->getAttrOfType<TypedAttr>("fill_with");
        if (!fillAttr)
            return rewriter.notifyMatchFailure(op, "no 'fill_with' attribute");
        if (fillAttr.getType() != op.getType().getElementType())
            return rewriter.notifyMatchFailure(
                op,
                "'fill_with' type doesn't match the buffer's element type");

        SmallVector<memref::LoadOp> loads;
        for (Operation* user : op->getUsers()) {
            if (auto load = dyn_cast<memref::LoadOp>(user)) {
                loads.push_back(load);
                continue;
            }
            return rewriter.notifyMatchFailure(
                op,
                "has a non-load user, not a pure filler buffer");
        }
        if (loads.empty()) return rewriter.notifyMatchFailure(op, "never read");

        LAKSA_DEBUG(
            llvm::dbgs() << "Resolving filler alloc " << op.getResult() << " ("
                         << loads.size() << " load(s))");

        rewriter.setInsertionPoint(op);
        auto fillVar = VariableOp::create(
            rewriter,
            op.getLoc(),
            fillAttr.getType(),
            fillAttr,
            /*is_const=*/true);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Created filler variable " << fillVar.getVariable());

        for (memref::LoadOp load : loads) {
            LAKSA_DEBUG(
                llvm::dbgs() << "  Replacing load " << load.getResult());
            rewriter.replaceOp(load, fillVar.getVariable());
        }
        rewriter.eraseOp(op);

        return success();
    }
};
} // namespace

namespace {
struct EmitHLSResolveHelpersPass
        : public emithls::impl::EmitHLSResolveHelpersBase<
              EmitHLSResolveHelpersPass> {
    void runOnOperation() override;

    LineBufferInfoMap lineBufferInfos;
};
} // namespace

void EmitHLSResolveHelpersPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    lineBufferInfos.clear();
    patterns.add<ResolveLineBuffer>(&getContext(), lineBufferInfos);
    patterns.add<ResolveWindow>(&getContext(), lineBufferInfos);
    patterns.add<ResolveAccumulate>(&getContext());

    target.addLegalDialect<EmitHLSDialect>();
    target
        .addIllegalOp<HelperLineBufferOp, HelperWindowOp, HelperAccumulateOp>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
        return;
    }

    // Filler allocs are plain memref.alloc ops that never go through an
    // illegal op, so this runs as its own greedy sweep instead of fighting
    // the conversion target's legality rules above.
    RewritePatternSet fillerPatterns(&getContext());
    fillerPatterns.add<ResolveFillerAlloc>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(fillerPatterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSResolveHelpersPass()
{ return std::make_unique<EmitHLSResolveHelpersPass>(); }
