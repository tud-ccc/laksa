// Implementation of AddIOFunctions transform pass.
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Debug.h>
#include <optional>
#include <utility>

#define DEBUG_TYPE "emithls-add-io-functions"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-add-io-functions] "; X;                      \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSADDIOFUNCTIONS
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

// Computes the bit width behind an IO argument type.
static std::optional<int64_t> getBitWidth(Type type)
{
    if (auto arrayType = dyn_cast<ArrayType>(type)) {
        auto elemBits = getBitWidth(arrayType.getElementType());
        if (!elemBits) return std::nullopt;
        int64_t numElements = 1;
        for (int64_t dim : arrayType.getShape()) numElements *= dim;
        return numElements * *elemBits;
    }
    if (auto streamType = dyn_cast<StreamType>(type))
        return getBitWidth(streamType.getElementType());
    if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
    return std::nullopt;
}

// Unwraps an array-of-stream (or plain stream) type down to the scalar type
// carried by the stream, e.g. `!emithls.array<8x!emithls.stream<i8>>` -> i8.
static Type getStreamElementType(Type type)
{
    if (auto arrayType = dyn_cast<ArrayType>(type))
        return getStreamElementType(arrayType.getElementType());
    if (auto streamType = dyn_cast<StreamType>(type))
        return getStreamElementType(streamType.getElementType());
    return type;
}

// Renders `type` as an identifier-safe name fragment, e.g. `i8`.
static std::string typeToIdentifier(Type type)
{
    if (auto intType = dyn_cast<IntegerType>(type))
        return ("i" + Twine(intType.getWidth())).str();
    std::string s;
    llvm::raw_string_ostream os(s);
    type.print(os);
    return s;
}

// Returns the value of "value" if it is defined by a const VariableOp
static std::optional<int64_t> getConstantInt(Value value)
{
    auto varOp = value.getDefiningOp<VariableOp>();
    if (!varOp || !varOp.getIsConst()) return std::nullopt;
    auto intAttr = dyn_cast_or_null<IntegerAttr>(varOp.getInitNumberAttr());
    if (!intAttr) return std::nullopt;
    return intAttr.getValue().getSExtValue();
}

// Parses "ifOp"'s condition, a single "emithls.arith.cmp", or several combined
// with "emithls.arith.logical_and", into a per-induction-variable [lo, hi]
// range
static void collectIfRangeConstraints(
    IfOp ifOp,
    DenseMap<Value, std::pair<int64_t, int64_t>> &ranges)
{
    auto exprOp = ifOp.getCondition().getDefiningOp<ExpressionOp>();
    if (!exprOp) return;
    auto yieldOp = cast<YieldOp>(exprOp.getBody().front().getTerminator());

    SmallVector<Value> conjuncts;
    if (auto andOp = yieldOp.getValue().getDefiningOp<ArithLogicalAndOp>())
        conjuncts.append(andOp.getValues().begin(), andOp.getValues().end());
    else
        conjuncts.push_back(yieldOp.getValue());

    for (Value v : conjuncts) {
        auto cmpOp = v.getDefiningOp<ArithCmpOp>();
        if (!cmpOp) continue;
        auto rhs = getConstantInt(cmpOp.getRhs());
        if (!rhs) continue;

        auto &range = ranges
                          .try_emplace(
                              cmpOp.getLhs(),
                              std::numeric_limits<int64_t>::min(),
                              std::numeric_limits<int64_t>::max())
                          .first->second;
        switch (cmpOp.getPredicate()) {
        case CmpPredicate::ge: range.first = std::max(range.first, *rhs); break;
        case CmpPredicate::le:
            range.second = std::min(range.second, *rhs);
            break;
        case CmpPredicate::gt:
            range.first = std::max(range.first, *rhs + 1);
            break;
        case CmpPredicate::lt:
            range.second = std::min(range.second, *rhs - 1);
            break;
        default: break;
        }
    }
}

// Whether "streamOp" (a "stream.read"/"stream.write") indexes into an
// array-of-stream, i.e. has a channel dimension whose loop should be
// excluded from the trip count.
static bool hasChannelIndex(Operation* streamOp)
{
    if (auto readOp = dyn_cast<StreamReadOp>(streamOp))
        return !readOp.getIndices().empty();
    if (auto writeOp = dyn_cast<StreamWriteOp>(streamOp))
        return !writeOp.getIndices().empty();
    return false;
}

// Counts how many times "streamOp" (a "stream.read"/"stream.write") executes
static int64_t computeTripCount(Operation* streamOp)
{
    DenseMap<Value, std::pair<int64_t, int64_t>> ranges;
    int64_t rawProduct = 1;
    int64_t restrictedProduct = 1;
    // A plain (non-array-of-stream) stream has no channel loop to exclude.
    bool skippedChannelLoop = !hasChannelIndex(streamOp);
    bool sawElseBranch = false;

    Operation* node = streamOp;
    while (Operation* parent = node->getParentOp()) {
        if (isa<FuncOp>(parent)) break;
        if (auto forOp = dyn_cast<ForOp>(parent)) {
            if (!skippedChannelLoop) {
                skippedChannelLoop = true;
            } else {
                int64_t lb = forOp.getLowerBound().getSExtValue();
                int64_t ub = forOp.getUpperBound().getSExtValue();
                int64_t step = forOp.getStep().getSExtValue();
                rawProduct *= std::max<int64_t>((ub - lb + step - 1) / step, 0);

                auto it = ranges.find(forOp.getInductionVariable());
                if (it != ranges.end()) {
                    lb = std::max(lb, it->second.first);
                    // An unpaired `ge` (no matching `le`) leaves the upper
                    // bound at INT64_MAX; `+ 1` would overflow, so only
                    // tighten `ub` when the constraint is actually lower.
                    if (it->second.second < ub) ub = it->second.second + 1;
                }
                restrictedProduct *=
                    std::max<int64_t>((ub - lb + step - 1) / step, 0);
            }
        } else if (auto ifOp = dyn_cast<IfOp>(parent)) {
            collectIfRangeConstraints(ifOp, ranges);
            if (node->getParentRegion() != &ifOp.getThenRegion())
                sawElseBranch = true;
        }
        node = parent;
    }
    return sawElseBranch ? (rawProduct - restrictedProduct) : restrictedProduct;
}

// Builds “word.range((idx+1)*elemBits-1, idx*elemBits)”
static Value buildChannelRangeExpr(
    OpBuilder &rewriter,
    Location loc,
    Type rawElemTy,
    Value word,
    Value idx1,
    Value one,
    Value elemBitsConst)
{
    auto indexTy = rewriter.getIndexType();
    auto outer = ExpressionOp::create(
        rewriter,
        loc,
        rawElemTy,
        [&](OpBuilder &eb, Location el) {
            auto hi = ExpressionOp::create(
                eb,
                el,
                indexTy,
                [&](OpBuilder &hb, Location hl) {
                    auto add0 = ArithAddOp::create(hb, hl, idx1, one);
                    auto mul0 = ArithMulOp::create(
                        hb,
                        hl,
                        add0.getResult(),
                        elemBitsConst);
                    auto sub0 =
                        ArithSubOp::create(hb, hl, mul0.getResult(), one);
                    YieldOp::create(hb, hl, sub0.getResult());
                });
            LAKSA_DEBUG(
                llvm::dbgs() << "    High bit as (idx + 1) * elemBits - 1");
            auto lo = ExpressionOp::create(
                eb,
                el,
                indexTy,
                [&](OpBuilder &lb, Location ll) {
                    auto mul0 = ArithMulOp::create(lb, ll, idx1, elemBitsConst);
                    YieldOp::create(lb, ll, mul0.getResult());
                });
            LAKSA_DEBUG(llvm::dbgs() << "    Low bit as idx * elemBits");
            auto dataRange = ArithDataRangeOp::create(
                eb,
                el,
                rawElemTy,
                word,
                hi.getResult(),
                lo.getResult());
            LAKSA_DEBUG(llvm::dbgs() << "    Data range using [hi, lo]");
            YieldOp::create(eb, el, dataRange.getResult());
        });
    return outer.getResult();
}

// Creates a new IO function right before "anchor"
static FuncOp createIOFunction(
    ConversionPatternRewriter &rewriter,
    Location loc,
    Operation* anchor,
    StringRef name,
    Type ptrTy,
    Type arrTy,
    bool isInput,
    int64_t tripCount)
{
    LAKSA_DEBUG(
        llvm::dbgs() << "Creating io function \"" << name << "\" at top");
    SmallVector<Type, 2> argTypes = isInput
                                        ? SmallVector<Type, 2>{ptrTy, arrTy}
                                        : SmallVector<Type, 2>{arrTy, ptrTy};
    auto funcType = rewriter.getFunctionType(argTypes, {});

    rewriter.setInsertionPoint(anchor);
    auto funcOp = FuncOp::create(rewriter, loc, name, funcType);
    SmallVector<Location> argLocs(argTypes.size(), loc);
    Block* entry = rewriter.createBlock(
        &funcOp.getBody(),
        funcOp.getBody().end(),
        argTypes,
        argLocs);
    Value ptrArg = isInput ? entry->getArgument(0) : entry->getArgument(1);
    Value arrArg = isInput ? entry->getArgument(1) : entry->getArgument(0);

    int64_t numChannels = 1;
    if (auto arrayType = dyn_cast<ArrayType>(arrTy))
        for (int64_t dim : arrayType.getShape()) numChannels *= dim;
    Type elemTy = getStreamElementType(arrTy);
    int64_t elemBits = *getBitWidth(elemTy);
    Type rawElemTy = rewriter.getIntegerType(elemBits);
    Type wordTy = cast<PointerType>(ptrTy).getElementType();
    auto indexTy = rewriter.getIndexType();

    rewriter.setInsertionPointToStart(entry);
    // Only needed to slice a channel out of/into a multi-channel word; a
    // bare (non-array-of-stream) argument has nothing to slice.
    Value one, elemBitsConst;
    if (numChannels > 1) {
        one = VariableOp::create(
                  rewriter,
                  loc,
                  indexTy,
                  rewriter.getIndexAttr(1),
                  true)
                  .getVariable();
        elemBitsConst = VariableOp::create(
                            rewriter,
                            loc,
                            indexTy,
                            rewriter.getIndexAttr(elemBits),
                            true)
                            .getVariable();
        LAKSA_DEBUG(
            llvm::dbgs() << "  Created index size-" << elemBits
                         << " for data_range indexing");
    }

    auto outer = ForOp::create(rewriter, loc, 0, tripCount, 1);
    rewriter.setInsertionPointToStart(&outer.getBody().front());
    Value outerIdx = outer.getInductionVariable();

    if (isInput) {
        auto word = ArrayPointerReadOp::create(
            rewriter,
            loc,
            wordTy,
            ptrArg,
            ValueRange{outerIdx});

        if (numChannels == 1) {
            // The whole word IS the one channel.
            StreamWriteOp::create(
                rewriter,
                loc,
                word.getResult(),
                arrArg,
                ValueRange{});
        } else {
            auto inner = ForOp::create(rewriter, loc, 0, numChannels, 1);
            rewriter.setInsertionPointToStart(&inner.getBody().front());
            Value idx1 = inner.getInductionVariable();

            Value slice = buildChannelRangeExpr(
                rewriter,
                loc,
                rawElemTy,
                word.getResult(),
                idx1,
                one,
                elemBitsConst);
            LAKSA_DEBUG(
                llvm::dbgs() << "  Built data range slice for " << word);
            StreamWriteOp::create(
                rewriter,
                loc,
                slice,
                arrArg,
                ValueRange{idx1});
        }
    } else if (numChannels == 1) {
        // Nothing to accumulate: the one channel read out directly becomes the
        // word.
        auto chanVal =
            StreamReadOp::create(rewriter, loc, elemTy, arrArg, ValueRange{});
        ArrayPointerWriteOp::create(
            rewriter,
            loc,
            chanVal.getResult(),
            ptrArg,
            ValueRange{outerIdx});
    } else {
        Value wordVar =
            VariableOp::create(
                rewriter,
                loc,
                wordTy,
                rewriter.getIntegerAttr(cast<IntegerType>(wordTy), 0),
                false)
                .getVariable();

        auto inner = ForOp::create(rewriter, loc, 0, numChannels, 1);
        rewriter.setInsertionPointToStart(&inner.getBody().front());
        Value idx1 = inner.getInductionVariable();

        auto chanVal = StreamReadOp::create(
            rewriter,
            loc,
            elemTy,
            arrArg,
            ValueRange{idx1});
        Value rawVal = chanVal.getResult();

        Value slice = buildChannelRangeExpr(
            rewriter,
            loc,
            rawElemTy,
            wordVar,
            idx1,
            one,
            elemBitsConst);
        LAKSA_DEBUG(llvm::dbgs() << "  Built data range slice for " << wordVar);
        UpdateOp::create(rewriter, loc, slice, rawVal);

        rewriter.setInsertionPointAfter(inner);
        ArrayPointerWriteOp::create(
            rewriter,
            loc,
            wordVar,
            ptrArg,
            ValueRange{outerIdx});
    }

    return funcOp;
}

struct AddIOFunctions : public OpConversionPattern<FuncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FuncOp op,
        FuncOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Adding IO functions for top function \""
                         << op.getSymName() << "\" at " << loc);

        auto moduleOp = op->getParentOfType<ModuleOp>();

        // New signature using ptr type
        SmallVector<Type> newArgTypes;
        for (Type argType : op.getArgumentTypes()) {
            auto bits = getBitWidth(argType);
            if (!bits)
                return rewriter.notifyMatchFailure(
                    op,
                    "unsupported argument type for IO pointer signature");
            auto ptrTy = PointerType::get(
                rewriter.getContext(),
                rewriter.getIntegerType(*bits));
            LAKSA_DEBUG(
                llvm::dbgs() << "  " << argType << " mapped to " << ptrTy);
            newArgTypes.push_back(ptrTy);
        }
        auto newFuncType = rewriter.getFunctionType(newArgTypes, {});

        // For each old argument, find the callee it feeds and inspect that
        // callee's body to see whether it's read (input) or written (output),
        // and how many transfers are actually needed.
        struct IOInfo {
            bool isInput = false;
            int64_t tripCount = 0;
            bool found = false;
        };
        SmallVector<IOInfo> ioInfos(op.getNumArguments());
        for (auto [i, blockArg] : llvm::enumerate(op.getArguments())) {
            for (Operation* user : blockArg.getUsers()) {
                auto callOp = dyn_cast<CallOp>(user);
                if (!callOp) continue;
                auto calleeFunc = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
                    callOp,
                    callOp.getCalleeAttr());
                if (!calleeFunc) continue;

                for (auto [argIdx, callArg] :
                     llvm::enumerate(callOp.getArgOperands())) {
                    if (callArg != blockArg) continue;
                    BlockArgument calleeArg = calleeFunc.getArgument(argIdx);
                    for (Operation* calleeUser : calleeArg.getUsers()) {
                        if (auto readOp = dyn_cast<StreamReadOp>(calleeUser)) {
                            ioInfos[i].isInput = true;
                            ioInfos[i].found = true;
                            ioInfos[i].tripCount += computeTripCount(readOp);
                        } else if (
                            auto writeOp =
                                dyn_cast<StreamWriteOp>(calleeUser)) {
                            ioInfos[i].isInput = false;
                            ioInfos[i].found = true;
                            ioInfos[i].tripCount += computeTripCount(writeOp);
                        }
                    }
                }
            }
            if (!ioInfos[i].found)
                return rewriter.notifyMatchFailure(
                    op,
                    "could not determine IO role/trip count for argument");
            LAKSA_DEBUG(
                llvm::dbgs() << "  arg" << i << ": "
                             << (ioInfos[i].isInput ? "input" : "output")
                             << ", " << ioInfos[i].tripCount << " transfers");
        }

        // Create or reuse the IO functions. Two arguments that need the exact
        // same pointer/array types and trip count share one IO function instead
        // of getting duplicate ones.
        struct CreatedIOFunc {
            bool isInput;
            Type ptrTy;
            Type arrTy;
            int64_t tripCount;
            FuncOp func;
        };
        SmallVector<CreatedIOFunc> createdIOFuncs;
        DenseMap<Type, int> nextReadNum, nextWriteNum;
        Operation* moduleAnchor = &moduleOp.getBody()->front();

        SmallVector<FuncOp> ioFuncs;
        for (auto [i, info] : llvm::enumerate(ioInfos)) {
            Type ptrTy = newArgTypes[i];
            Type arrTy = op.getArgumentTypes()[i];

            FuncOp reused;
            for (auto &created : createdIOFuncs) {
                if (created.isInput == info.isInput && created.ptrTy == ptrTy
                    && created.arrTy == arrTy
                    && created.tripCount == info.tripCount) {
                    reused = created.func;
                    break;
                }
            }
            if (reused) {
                LAKSA_DEBUG(
                    llvm::dbgs() << "  arg" << i << ": reusing \""
                                 << reused.getSymName() << "\"");
                ioFuncs.push_back(reused);
                continue;
            }

            Type elemTy = getStreamElementType(arrTy);
            DenseMap<Type, int> &nextNum =
                info.isInput ? nextReadNum : nextWriteNum;
            int num = nextNum[elemTy]++;
            std::string name =
                (op.getSymName() + (info.isInput ? "_read_" : "_write_")
                 + typeToIdentifier(elemTy) + "_" + Twine(num))
                    .str();

            FuncOp ioFunc = createIOFunction(
                rewriter,
                loc,
                moduleAnchor,
                name,
                ptrTy,
                arrTy,
                info.isInput,
                info.tripCount);
            createdIOFuncs.push_back(
                {info.isInput, ptrTy, arrTy, info.tripCount, ioFunc});
            ioFuncs.push_back(ioFunc);
        }

        // Rebuild the top function
        rewriter.setInsertionPoint(op);
        auto newFuncOp =
            FuncOp::create(rewriter, loc, op.getSymName(), newFuncType);
        if (Attribute rootAttr = op->getAttr(laksa::kRootAttrName))
            newFuncOp->setAttr(laksa::kRootAttrName, rootAttr);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Rebuilding top function \"" << op.getSymName() << "\"");
        SmallVector<Location> argLocs(newArgTypes.size(), loc);
        Block* newEntry = rewriter.createBlock(
            &newFuncOp.getBody(),
            newFuncOp.getBody().end(),
            newArgTypes,
            argLocs);
        rewriter.setInsertionPointToStart(newEntry);

        SmallVector<Value> localVars;
        for (Type oldTy : op.getArgumentTypes()) {
            localVars.push_back(
                VariableOp::create(rewriter, loc, oldTy, /*initValue=*/Value{})
                    .getVariable());
            LAKSA_DEBUG(
                llvm::dbgs()
                << "  Added variable to replace old block argument");
        }

        // The input IO calls must land after every variable, right before the
        // first kernel call.
        Operation* firstOldCallOp = nullptr;
        for (Operation &bodyOp : op.getBody().front())
            if (isa<CallOp>(bodyOp)) {
                firstOldCallOp = &bodyOp;
                break;
            }
        rewriter.mergeBlocks(&op.getBody().front(), newEntry, localVars);

        // Input IO calls go right before the first original kernel call, and
        // output IO calls go at the very end.
        if (firstOldCallOp) rewriter.setInsertionPoint(firstOldCallOp);
        for (auto [i, info] : llvm::enumerate(ioInfos)) {
            if (!info.isInput) continue;
            if (!firstOldCallOp) rewriter.setInsertionPointToEnd(newEntry);
            CallOp::create(
                rewriter,
                loc,
                ioFuncs[i],
                ValueRange{newEntry->getArgument(i), localVars[i]});
            LAKSA_DEBUG(
                llvm::dbgs() << "  Added call to input function \""
                             << ioFuncs[i].getSymName() << "\"");
        }
        for (auto [i, info] : llvm::enumerate(ioInfos)) {
            if (info.isInput) continue;
            rewriter.setInsertionPointToEnd(newEntry);
            CallOp::create(
                rewriter,
                loc,
                ioFuncs[i],
                ValueRange{localVars[i], newEntry->getArgument(i)});
            LAKSA_DEBUG(
                llvm::dbgs() << "  Added call to output function \""
                             << ioFuncs[i].getSymName() << "\"");
        }

        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

namespace {
struct EmitHLSAddIOFunctionsPass
        : public emithls::impl::EmitHLSAddIOFunctionsBase<
              EmitHLSAddIOFunctionsPass> {
    void runOnOperation() override;
};
} // namespace

void EmitHLSAddIOFunctionsPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<AddIOFunctions>(&getContext());

    target.addLegalDialect<EmitHLSDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    // Only the root function is the top-level driver whose ports need pointer
    // interfaces.
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
        if (!funcOp->hasAttr(laksa::kRootAttrName)) return true;
        return llvm::all_of(funcOp.getArgumentTypes(), [](Type type) {
            return isa<PointerType>(type);
        });
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSAddIOFunctionsPass()
{ return std::make_unique<EmitHLSAddIOFunctionsPass>(); }
