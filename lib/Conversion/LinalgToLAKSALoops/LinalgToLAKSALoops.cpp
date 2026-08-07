/// Implementation of LinalgToLAKSALoops pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/LinalgToLAKSALoops/LinalgToLAKSALoops.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>

#define DEBUG_TYPE "linalg-to-laksa-loops"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[linalg-to-laksa-loops] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOLAKSALOOPS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::emithls;

namespace {

// Some indexing maps use a constant (e.g. 0) for an axis that is numerically
// equivalent to using the shared loop's own induction variable, which happens
// when the axis is size-1 on both this operand and the loop's range for that
// position.
static AffineMap normalizeDegenerateConstantResults(
    AffineMap map,
    ArrayRef<int64_t> operandShape,
    ArrayRef<unsigned> loopRanges)
{
    if (map.getNumResults() != loopRanges.size()
        || map.getNumResults() != operandShape.size())
        return map;
    SmallVector<AffineExpr> results(map.getResults());
    bool changed = false;
    for (unsigned i = 0; i < results.size(); ++i) {
        if (isa<AffineDimExpr>(results[i])) continue;
        if (operandShape[i] == static_cast<int64_t>(loopRanges[i])) {
            results[i] = getAffineDimExpr(i, map.getContext());
            changed = true;
        }
    }
    if (!changed) return map;
    return AffineMap::get(
        map.getNumDims(),
        map.getNumSymbols(),
        results,
        map.getContext());
}

struct GenericToOptimizedLoops : OpConversionPattern<linalg::GenericOp> {
    GenericToOptimizedLoops(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<linalg::GenericOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        linalg::GenericOp op,
        linalg::GenericOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting generic operation at " << loc);
        auto process = op->getParentOfType<dfg::ProcessOp>();
        if (!process)
            return rewriter.notifyMatchFailure(
                loc,
                "expect operation to be inside a process");

        // 　If this generic is a fill op, save the filling value(s)
        auto isFill = linalg::isaFillOpInterface(op);
        auto isBcast = linalg::isaBroadcastOpInterface(op);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "isaFillOpInterface=" << (isFill.has_value() ? "true" : "false")
            << " isaBroadcastOpInterface="
            << (isBcast.has_value() ? "true" : "false"));
        if (isFill || isBcast) {
            auto fillAttrs = getFillOrBroadcastAttrs(op);
            if (fillAttrs.empty())
                return rewriter.notifyMatchFailure(
                    loc,
                    "Cannot find the numbers to be filled with.");
            for (auto [filled, fillAttr] :
                 llvm::zip(op.getDpsInits(), fillAttrs)) {
                auto filledDefOp = filled.getDefiningOp();
                filledDefOp->setAttr("fill_with", fillAttr);
            }
            LAKSA_DEBUG(
                llvm::dbgs() << "Found fill/broadcast generic operation, "
                                "saved the attribute to output");
            rewriter.eraseOp(op);
            return success();
        }

        auto iteratorTypes = op.getIteratorTypesArray();

        auto [hasSlidingWindow, stride, dilation] = analyzeSlidingWindow(op);
        auto isPureParallel =
            llvm::all_of(iteratorTypes, [](auto iteratorType) {
                return iteratorType == utils::IteratorType::parallel;
            });

        if (hasSlidingWindow) {
            // If the operation contains a sliding window, need to analyze the
            // loop invariants, create the affine loops, and insert the LAKSA
            // optimizations
            LAKSA_DEBUG(
                llvm::dbgs()
                << "This is a sliding window operation with stride=" << stride
                << " and dilation=" << dilation);

            // Get information of to be generated loops.
            // Dim positions in the linalg loop space (d0..dn-1) are stored
            // parallel to the range vectors so that after loop creation:
            //   writeLoopIterators[i]  ↔ writeDimPositions[i]
            //   reduceLoopIterators[i] ↔ reduceDimPositions[i]
            SmallVector<unsigned> readLoopRanges, writeLoopRanges,
                reduceLoopRanges, windowRanges;
            SmallVector<unsigned> writeDimPositions, reduceDimPositions,
                windowDimPositions;
            // All plain (non-compound) AffineDimExpr dims from the sliding
            // input, in map-result order.  Includes both parallel dims (N, C
            // for pooling) and reduction dims (IC for conv2d).  These set the
            // linebuf/compWindow channel shape and the window access map —
            // deliberately excluding output-only dims like OC.
            SmallVector<unsigned> slidingInputPlainDimPositions,
                slidingInputPlainRanges;
            Value slidingInput, compWindow;
            {
                auto inputs = op.getDpsInputs();
                auto outputs = op.getDpsInits();
                auto affineMaps = op.getIndexingMapsArray();
                unsigned numInputs = op.getNumDpsInputs();

                // Pre-scan: find the sliding input (the first input with at
                // least one compound affine expression).
                int slidingInputMapIdx = -1;
                for (unsigned mapIdx = 0; mapIdx < numInputs; ++mapIdx) {
                    auto inputMap = affineMaps[mapIdx];
                    for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
                        if (!isa<AffineDimExpr>(inputMap.getResult(i))) {
                            slidingInputMapIdx = static_cast<int>(mapIdx);
                            break;
                        }
                    }
                    if (slidingInputMapIdx >= 0) break;
                }
                if (slidingInputMapIdx >= 0)
                    slidingInput = inputs[slidingInputMapIdx];

                // Parallel dims that appear as part of a compound expression
                // (e.g. OH in `OH*stride + KH`).  These are the output-spatial
                // dims that must NOT become write loops.
                SmallVector<unsigned> compoundParallelDims;
                // Collect (dim_pos, size) for reduction dims, sorted by pos.
                SmallVector<std::pair<unsigned, unsigned>> reduceDimSizes;
                SmallVector<unsigned> seenReduceDims;

                for (unsigned mapIdx = 0; mapIdx < numInputs; ++mapIdx) {
                    auto inputMap = affineMaps[mapIdx];
                    auto memrefShape =
                        cast<MemRefType>(inputs[mapIdx].getType()).getShape();
                    for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
                        auto expr = inputMap.getResult(i);
                        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                            unsigned pos = dimExpr.getPosition();
                            // Record every plain dim from the sliding input
                            // (both parallel dims like N,C and reduction dims
                            // like IC) for linebuf/compWindow channel shaping.
                            if (static_cast<int>(mapIdx) == slidingInputMapIdx
                                && !llvm::is_contained(
                                    slidingInputPlainDimPositions,
                                    pos)) {
                                slidingInputPlainDimPositions.push_back(pos);
                                slidingInputPlainRanges.push_back(
                                    static_cast<unsigned>(memrefShape[i]));
                            }
                            if (iteratorTypes[pos]
                                == utils::IteratorType::reduction) {
                                if (!llvm::is_contained(seenReduceDims, pos)) {
                                    seenReduceDims.push_back(pos);
                                    reduceDimSizes.push_back(
                                        {pos,
                                         static_cast<unsigned>(
                                             memrefShape[i])});
                                }
                            }
                        } else {
                            // Compound expression (stride/dilation pattern).
                            readLoopRanges.push_back(
                                static_cast<unsigned>(memrefShape[i]));
                            expr.walk([&](AffineExpr e) {
                                if (auto d = dyn_cast<AffineDimExpr>(e)) {
                                    unsigned p = d.getPosition();
                                    if (iteratorTypes[p]
                                            == utils::IteratorType::parallel
                                        && !llvm::is_contained(
                                            compoundParallelDims,
                                            p))
                                        compoundParallelDims.push_back(p);
                                    if (iteratorTypes[p]
                                            == utils::IteratorType::reduction
                                        && !llvm::is_contained(
                                            windowDimPositions,
                                            p))
                                        windowDimPositions.push_back(p);
                                }
                            });
                        }
                    }
                }

                llvm::sort(reduceDimSizes, llvm::less_first());
                for (auto [pos, size] : reduceDimSizes) {
                    reduceLoopRanges.push_back(size);
                    reduceDimPositions.push_back(pos);
                    if (llvm::is_contained(windowDimPositions, pos))
                        windowRanges.push_back(size);
                }

                // Write dims: all parallel output dims EXCEPT the output-
                // spatial dims (OH, OW) that appear in compound expressions.
                // This correctly captures dims like OC that only appear in
                // the output and not in any input map.
                SmallVector<std::pair<unsigned, unsigned>> writeDimSizes;
                for (unsigned mapIdx = 0; mapIdx < op.getNumDpsInits();
                     ++mapIdx) {
                    auto outputMap = affineMaps[numInputs + mapIdx];
                    auto memrefShape =
                        cast<MemRefType>(outputs[mapIdx].getType()).getShape();
                    for (unsigned i = 0; i < outputMap.getNumResults(); ++i) {
                        auto expr = outputMap.getResult(i);
                        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                            unsigned pos = dimExpr.getPosition();
                            if (iteratorTypes[pos]
                                    == utils::IteratorType::parallel
                                && !llvm::is_contained(
                                    compoundParallelDims,
                                    pos))
                                writeDimSizes.push_back(
                                    {pos,
                                     static_cast<unsigned>(memrefShape[i])});
                        }
                    }
                }
                llvm::sort(writeDimSizes, llvm::less_first());
                for (auto [pos, size] : writeDimSizes) {
                    writeLoopRanges.push_back(size);
                    writeDimPositions.push_back(pos);
                }
            }
            LAKSA_DEBUG(
                llvm::dbgs() << "Read loop ranges (input spatial): ";
                for (auto r : readLoopRanges) llvm::dbgs() << r << " ");
            LAKSA_DEBUG(
                llvm::dbgs() << "Write loop ranges: ";
                for (auto [pos, r] : llvm::zip(
                         writeDimPositions,
                         writeLoopRanges)) llvm::dbgs()
                << "d" << pos << "=" << r << " ");
            LAKSA_DEBUG(
                llvm::dbgs() << "Reduce loop ranges: ";
                for (auto [pos, r] : llvm::zip(
                         reduceDimPositions,
                         reduceLoopRanges)) llvm::dbgs()
                << "d" << pos << "=" << r << " ");
            LAKSA_DEBUG(
                llvm::dbgs() << "Window ranges: ";
                for (auto r : windowRanges) llvm::dbgs() << r << " ");

            // Create loops with LAKSA optimization.
            // readLoopIterators scan raw INPUT spatial positions (not a single
            // affine map dim) and are only used for linebuf/window/AffineIf.
            // writeLoopIterators[i] ↔ writeDimPositions[i]
            // reduceLoopIterators[i] ↔ reduceDimPositions[i]
            // dimToIterator[d] = IV for affine map dim d, enabling:
            //   affine.load %operand[indexingMap](dimToIterator[0..n-1])
            SmallVector<Value> readLoopIterators, writeLoopIterators,
                reduceLoopIterators;
            SmallVector<Value> dimToIterator(op.getNumLoops());
            {
                for (auto range : readLoopRanges) {
                    auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                    readLoopIterators.push_back(forOp.getInductionVar());
                    rewriter.setInsertionPointToStart(
                        &forOp.getRegion().front());
                }
                // Insert line buffer operation.
                // With kernel height KH and dilation d, (KH-1)*d rows must be
                // retained to cover the full dilated receptive field.
                // The channel dimensions come from the sliding input's own
                // plain parallel dims (N, C for pooling; N only for conv2d
                // where OC is output-only and not buffered per channel).
                int32_t numLines =
                    static_cast<int32_t>((windowRanges[0] - 1) * dilation);
                SmallVector<int32_t> numChan(
                    slidingInputPlainRanges.begin(),
                    slidingInputPlainRanges.end());
                SmallVector<int64_t> linebufShape;
                for (auto range : slidingInputPlainRanges)
                    linebufShape.push_back(static_cast<int64_t>(range));
                linebufShape.push_back(static_cast<int64_t>(numLines));
                // For image-like (2D+) sliding windows, e.g. conv2d, each
                // buffered line spans the full inner (column) extent, which
                // must be kept as a trailing dimension of the line buffer;
                // the column IV (the "keep" index) then selects where along
                // that dimension the newly streamed element is written. A
                // purely 1D sliding window (e.g. a FIR filter) has no such
                // kept dimension: the buffer is just a numLines-deep shift
                // register of scalars (per channel), so no index needs to be
                // kept.
                Value scanIndex;
                if (readLoopRanges.size() > 1) {
                    linebufShape.push_back(
                        static_cast<int64_t>(readLoopRanges[1]));
                    scanIndex = readLoopIterators[1];
                }
                Value lineBuffer = HelperLineBufferOp::create(
                    rewriter,
                    loc,
                    slidingInput,
                    scanIndex,
                    MemRefType::get(
                        linebufShape,
                        cast<ShapedType>(slidingInput.getType())
                            .getElementType()),
                    numChan,
                    numLines);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "Constructed line buffer at " << lineBuffer.getLoc());
                // Construct compute window.
                // The window covers the full dilated receptive field, so each
                // spatial dimension has effective size (K-1)*dilation + 1.
                SmallVector<int64_t> compwinShape;
                for (auto range : slidingInputPlainRanges)
                    compwinShape.push_back(static_cast<int64_t>(range));
                for (auto range : windowRanges)
                    compwinShape.push_back(static_cast<int64_t>(range));
                compWindow = HelperWindowOp::create(
                    rewriter,
                    loc,
                    lineBuffer,
                    readLoopIterators,
                    MemRefType::get(
                        compwinShape,
                        cast<ShapedType>(lineBuffer.getType())
                            .getElementType()));
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "Constructed compute window at " << compWindow.getLoc());
                // Create affine if with affine set to start computation under
                // condition, where the dimensions meet the stride and dilation.
                // For each read dim i with window size W_i, stride s, dilation
                // d, window elements are at d_i, d_i-d, ..., d_i-(W_i-1)*d,
                // so we need:
                //   d_i >= (W_i-1)*d           [warmup: enough data collected]
                //   (d_i - (W_i-1)*d) % s == 0  [stride: valid output position]
                unsigned numReadDims = readLoopIterators.size();
                SmallVector<AffineExpr> constraints;
                SmallVector<bool> eqFlags;
                for (unsigned i = 0; i < numReadDims; ++i) {
                    AffineExpr d = getAffineDimExpr(i, ctx);
                    int64_t warmup =
                        static_cast<int64_t>((windowRanges[i] - 1) * dilation);
                    // d_i - warmup >= 0
                    constraints.push_back(d - warmup);
                    eqFlags.push_back(false);
                    // (d_i - warmup) % stride == 0
                    if (stride > 1) {
                        constraints.push_back(
                            (d - warmup) % static_cast<int64_t>(stride));
                        eqFlags.push_back(true);
                    }
                }
                auto condSet =
                    IntegerSet::get(numReadDims, 0, constraints, eqFlags);
                auto ifOp = AffineIfOp::create(
                    rewriter,
                    loc,
                    condSet,
                    readLoopIterators,
                    /*withElseRegion=*/false);
                rewriter.setInsertionPointToStart(ifOp.getThenBlock());
            }

            // Create write loops (batch, oc) then reduce loops (for instance
            // kh, kw, ic) inside the AffineIfOp then-block, all in dim-position
            // order. Each IV is registered in dimToIterator so that any
            // operand's affine.load can be built as:
            //   affine.load %operand[indexingMap](dimToIterator[0..n-1])
            for (auto [range, dimPos] :
                 llvm::zip(writeLoopRanges, writeDimPositions)) {
                auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                Value iv = forOp.getInductionVar();
                writeLoopIterators.push_back(iv);
                dimToIterator[dimPos] = iv;
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }
            // Placeholder for dims not in dimToIterator (d1/d2, output spatial
            // dims).  They don't appear in any inner indexing map; index 0
            // keeps AffineLoadOp operands valid.  Created here so it is
            // outside the reduce loops and not recreated each iteration.
            Value indexZero = arith::ConstantOp::create(
                rewriter,
                loc,
                rewriter.getIndexAttr(0));
            for (auto [range, dimPos] :
                 llvm::zip(reduceLoopRanges, reduceDimPositions)) {
                auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                Value iv = forOp.getInductionVar();
                reduceLoopIterators.push_back(iv);
                dimToIterator[dimPos] = iv;
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }

            // Copy the computation body from the generic op into the innermost
            // loop, replacing block args with affine.loads and converting the
            // yield's integer-add-with-output-arg reduction pattern to
            // helper.accumulate.
            {
                auto &body = *op.getBody();
                auto blockArgs = body.getArguments();
                unsigned numInputs = op.getNumDpsInputs();
                unsigned numOutputs = op.getNumDpsInits();
                auto inputOperands = op.getDpsInputs();
                auto outputOperands = op.getDpsInits();
                auto maps = op.getIndexingMapsArray();
                auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

                // Build the flat operand list for AffineLoadOp: one Value per
                // loop dim, with indexZero for dims not covered by any loop.
                auto dimIterOperands = llvm::map_to_vector(
                    dimToIterator,
                    [&](Value iv) -> Value { return iv ? iv : indexZero; });

                // Detect accumulate pattern.
                DenseSet<Operation*> accumulateDefOps;
                SmallVector<Operation*> accuDefOps(numOutputs, nullptr);
                SmallVector<unsigned> accuIncrIdx(numOutputs, 0);
                SmallVector<FusedOperator> accuFuseOps(
                    numOutputs,
                    FusedOperator::add);
                detectAccumulatePattern(
                    blockArgs,
                    numInputs,
                    numOutputs,
                    yieldOp,
                    accuDefOps,
                    accuIncrIdx,
                    accuFuseOps,
                    accumulateDefOps);

                // Map block args to loaded values.
                IRMapping bodyMapping;
                for (unsigned i = 0; i < numInputs; ++i) {
                    auto blkArg = blockArgs[i];
                    if (blkArg.use_empty()) continue;

                    auto inputMap = maps[i];
                    // A sliding-window input has at least one compound result.
                    bool isWindow = llvm::any_of(
                        inputMap.getResults(),
                        [](AffineExpr e) { return !isa<AffineDimExpr>(e); });

                    if (isWindow) {
                        // compWindow shape is [slidingInputPlainRanges...,
                        // windowRanges...], i.e. slidingInputPlainDimPositions
                        // then windowDimPositions.  Use those to build the
                        // index map directly.  writeDimPositions may include
                        // output-only dims like OC that are NOT axes of
                        // compWindow.
                        SmallVector<AffineExpr> winExprs;
                        for (unsigned dimPos : slidingInputPlainDimPositions)
                            winExprs.push_back(getAffineDimExpr(dimPos, ctx));
                        for (unsigned dimPos : windowDimPositions)
                            winExprs.push_back(getAffineDimExpr(dimPos, ctx));
                        auto winMap =
                            AffineMap::get(op.getNumLoops(), 0, winExprs, ctx);
                        bodyMapping.map(
                            blkArg,
                            AffineLoadOp::create(
                                rewriter,
                                loc,
                                compWindow,
                                winMap,
                                dimIterOperands));
                    } else {
                        // Regular input: apply original indexing map.
                        bodyMapping.map(
                            blkArg,
                            AffineLoadOp::create(
                                rewriter,
                                loc,
                                inputOperands[i],
                                inputMap,
                                dimIterOperands));
                    }
                }

                // Clone body ops, skipping accumulate-pattern ops.
                for (auto &bodyOp : body.without_terminator()) {
                    if (accumulateDefOps.contains(&bodyOp)) continue;
                    rewriter.clone(bodyOp, bodyMapping);
                }

                // Generate helper.accumulate for each output.
                // Directly use defOp->getOperand(incrIdx) — the non-output
                // operand from the detected binary op — then look it up in
                // bodyMapping to get the cloned version.
                for (auto [defOp, incrIdx, fuseOp, outOperand] : llvm::zip(
                         accuDefOps,
                         accuIncrIdx,
                         accuFuseOps,
                         outputOperands)) {
                    if (!defOp) continue;
                    HelperAccumulateOp::create(
                        rewriter,
                        loc,
                        outOperand,
                        writeLoopIterators,
                        fuseOp,
                        bodyMapping.lookup(defOp->getOperand(incrIdx)));
                    // Attach attribute to the outOperand to mark which
                    // dimensions are used for potentially parallel writing, as
                    // well as helping the channel construction later.
                    // writeDimPositions[i] ↔ writeLoopIterators[i] by
                    // construction.
                    outOperand.getDefiningOp()->setAttr(
                        "accu_at",
                        rewriter.getI32ArrayAttr(
                            SmallVector<int32_t>(
                                writeDimPositions.begin(),
                                writeDimPositions.end())));
                }
            }
            rewriter.eraseOp(op);
            LAKSA_DEBUG(llvm::dbgs() << "Erase original generic operation.");
        } else if (isPureParallel) {
            LAKSA_DEBUG(llvm::dbgs() << "This is a pure parallel operation.");

            auto inputs = op.getDpsInputs();
            auto outputs = op.getDpsInits();
            unsigned numInputs = op.getNumDpsInputs();
            unsigned numOutputs = op.getNumDpsInits();
            auto maps = op.getIndexingMapsArray();
            unsigned numLoops = op.getNumLoops();

            // Collect loop bound for each dim position from all operands.
            SmallVector<unsigned> loopRanges(numLoops, 0);
            for (unsigned mapIdx = 0; mapIdx < maps.size(); ++mapIdx) {
                Value operand = mapIdx < numInputs
                                    ? inputs[mapIdx]
                                    : outputs[mapIdx - numInputs];
                auto shape = cast<MemRefType>(operand.getType()).getShape();
                for (unsigned i = 0; i < maps[mapIdx].getNumResults(); ++i) {
                    if (auto dimExpr = dyn_cast<AffineDimExpr>(
                            maps[mapIdx].getResult(i))) {
                        unsigned pos = dimExpr.getPosition();
                        if (loopRanges[pos] == 0)
                            loopRanges[pos] = static_cast<unsigned>(shape[i]);
                    }
                }
            }

            // Attach parallel mark to output memrefs for downstream passes.
            for (auto outputValue : outputs)
                outputValue.getDefiningOp()->setAttr(
                    "parallel",
                    rewriter.getUnitAttr());

            // Create one AffineForOp per dim position, outermost-first.
            SmallVector<Value> dimToIterator(numLoops);
            for (unsigned pos = 0; pos < numLoops; ++pos) {
                auto forOp =
                    AffineForOp::create(rewriter, loc, 0, loopRanges[pos]);
                dimToIterator[pos] = forOp.getInductionVar();
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }

            Value indexZero = arith::ConstantOp::create(
                rewriter,
                loc,
                rewriter.getIndexAttr(0));
            auto dimIterOperands = llvm::map_to_vector(
                dimToIterator,
                [&](Value iv) -> Value { return iv ? iv : indexZero; });

            auto &body = *op.getBody();
            auto blockArgs = body.getArguments();
            auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

            // Map input block args to affine.load results.
            IRMapping bodyMapping;
            for (unsigned i = 0; i < numInputs; ++i) {
                auto blkArg = blockArgs[i];
                if (blkArg.use_empty()) continue;
                bodyMapping.map(
                    blkArg,
                    AffineLoadOp::create(
                        rewriter,
                        loc,
                        inputs[i],
                        normalizeDegenerateConstantResults(
                            maps[i],
                            cast<MemRefType>(inputs[i].getType()).getShape(),
                            loopRanges),
                        dimIterOperands));
            }

            // Clone body ops, excluding the terminator.
            for (auto &bodyOp : body.without_terminator())
                rewriter.clone(bodyOp, bodyMapping);

            // Store each yielded value into the corresponding output.
            for (unsigned i = 0; i < numOutputs; ++i) {
                Value stored =
                    bodyMapping.lookupOrDefault(yieldOp.getValues()[i]);
                AffineStoreOp::create(
                    rewriter,
                    loc,
                    stored,
                    outputs[i],
                    normalizeDegenerateConstantResults(
                        maps[numInputs + i],
                        cast<MemRefType>(outputs[i].getType()).getShape(),
                        loopRanges),
                    dimIterOperands);
            }

            rewriter.eraseOp(op);
            LAKSA_DEBUG(
                llvm::dbgs()
                << "Erase original pure parallel generic operation.");
        } else {
            LAKSA_DEBUG(llvm::dbgs() << "This is a reduction operation.");

            auto inputs = op.getDpsInputs();
            auto affineMaps = op.getIndexingMapsArray();
            unsigned numInputs = op.getNumDpsInputs();

            SmallVector<unsigned> reduceLoopRanges, readLoopRanges;
            SmallVector<unsigned> seenReduceDims, seenReadDims;
            // The first non-constant input: used as the line buffer token and
            // to derive the line buffer memref shape.
            Value nonConstInput;
            AffineMap nonConstInputMap;
            SmallVector<int64_t> nonConstInputShape;

            for (unsigned mapIdx = 0; mapIdx < numInputs; ++mapIdx) {
                auto inputMap = affineMaps[mapIdx];
                auto memrefShape =
                    cast<MemRefType>(inputs[mapIdx].getType()).getShape();
                auto* defOp = inputs[mapIdx].getDefiningOp();
                // Weight/constant tensors are backed by a global memref.
                bool isConstantOrWeight =
                    defOp && isa<memref::GetGlobalOp>(defOp);

                if (!isConstantOrWeight && !nonConstInput) {
                    nonConstInput = inputs[mapIdx];
                    nonConstInputMap = inputMap;
                    nonConstInputShape.assign(
                        memrefShape.begin(),
                        memrefShape.end());
                }

                for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
                    auto expr = inputMap.getResult(i);
                    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                        unsigned pos = dimExpr.getPosition();
                        if (iteratorTypes[pos] == utils::IteratorType::reduction
                            && !llvm::is_contained(seenReduceDims, pos)) {
                            seenReduceDims.push_back(pos);
                            reduceLoopRanges.push_back(
                                static_cast<unsigned>(memrefShape[i]));
                        }
                        if (!isConstantOrWeight
                            && iteratorTypes[pos]
                                   == utils::IteratorType::parallel
                            && !llvm::is_contained(seenReadDims, pos)) {
                            seenReadDims.push_back(pos);
                            readLoopRanges.push_back(
                                static_cast<unsigned>(memrefShape[i]));
                        }
                    }
                }
            }

            // Collect write dims: parallel dims in the output map that were
            // not claimed as read loop dims (i.e. output-only dims like OC).
            SmallVector<unsigned> writeLoopRanges;
            SmallVector<unsigned> seenWriteDims;
            auto outputs = op.getDpsInits();
            unsigned numOutputs = op.getNumDpsInits();
            for (unsigned mapIdx = 0; mapIdx < numOutputs; ++mapIdx) {
                auto outputMap = affineMaps[numInputs + mapIdx];
                auto memrefShape =
                    cast<MemRefType>(outputs[mapIdx].getType()).getShape();
                for (unsigned i = 0; i < outputMap.getNumResults(); ++i) {
                    auto expr = outputMap.getResult(i);
                    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                        unsigned pos = dimExpr.getPosition();
                        if (iteratorTypes[pos] == utils::IteratorType::parallel
                            && !llvm::is_contained(seenReadDims, pos)
                            && !llvm::is_contained(seenWriteDims, pos)) {
                            seenWriteDims.push_back(pos);
                            writeLoopRanges.push_back(
                                static_cast<unsigned>(memrefShape[i]));
                        }
                    }
                }
            }

            LAKSA_DEBUG(
                llvm::dbgs() << "Read loop ranges: ";
                for (auto [pos, r] : llvm::zip(seenReadDims, readLoopRanges))
                    llvm::dbgs()
                << "d" << pos << "=" << r << " ");
            LAKSA_DEBUG(
                llvm::dbgs() << "Write loop ranges: ";
                for (auto [pos, r] : llvm::zip(seenWriteDims, writeLoopRanges))
                    llvm::dbgs()
                << "d" << pos << "=" << r << " ");
            LAKSA_DEBUG(
                llvm::dbgs() << "Reduce loop ranges: ";
                for (auto [pos, r] : llvm::zip(
                         seenReduceDims,
                         reduceLoopRanges)) llvm::dbgs()
                << "d" << pos << "=" << r << " ");

            // Create AffineForOps for each read (parallel) dimension, nesting
            // them outermost-first, then set the insertion point inside the
            // innermost loop body.
            SmallVector<Value> readLoopIterators;
            for (auto range : readLoopRanges) {
                auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                readLoopIterators.push_back(forOp.getInductionVar());
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }

            // Build the line buffer memref shape from the non-constant input:
            // parallel dims → 1, reduction dims → keep original size.
            Value linebuf;
            if (nonConstInput) {
                SmallVector<int64_t> linebufShape;
                for (unsigned i = 0; i < nonConstInputMap.getNumResults();
                     ++i) {
                    auto expr = nonConstInputMap.getResult(i);
                    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                        unsigned pos = dimExpr.getPosition();
                        if (iteratorTypes[pos] == utils::IteratorType::parallel)
                            linebufShape.push_back(1);
                        else
                            linebufShape.push_back(nonConstInputShape[i]);
                    }
                }
                auto linebufType = MemRefType::get(
                    linebufShape,
                    cast<ShapedType>(nonConstInput.getType()).getElementType());
                Value scanIndex = readLoopIterators.empty()
                                      ? Value{}
                                      : readLoopIterators.back();
                linebuf = HelperLineBufferOp::create(
                    rewriter,
                    loc,
                    nonConstInput,
                    scanIndex,
                    linebufType,
                    ArrayRef<int32_t>{1},
                    /*numLine=*/0);
                LAKSA_DEBUG(
                    llvm::dbgs()
                        << "Constructed line buffer for reduction, shape=[";
                    llvm::interleaveComma(linebufShape, llvm::dbgs());
                    llvm::dbgs() << "]");
            } else {
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "No non-constant input; skipping line buffer.");
            }

            // A [0,1) loop wraps the compute body and represents the single
            // buffered row.  Its IV is used to index the parallel dimensions
            // of the line buffer, keeping the load address inside its shape.
            Value linebufRowIV;
            if (linebuf) {
                auto rowOp = AffineForOp::create(rewriter, loc, 0, 1);
                linebufRowIV = rowOp.getInductionVar();
                rewriter.setInsertionPointToStart(&rowOp.getRegion().front());
            }

            // Build dimToIterator[pos] = IV for all loop dim positions so
            // that affine.load can be generated as:
            //   affine.load %operand[map](dimToIterator[0..n-1])
            SmallVector<Value> writeLoopIterators, reduceLoopIterators;
            SmallVector<Value> dimToIterator(op.getNumLoops());
            for (auto [pos, iv] : llvm::zip(seenReadDims, readLoopIterators))
                dimToIterator[pos] = iv;

            // index 0 placeholder for dims not covered by any loop
            Value indexZero = arith::ConstantOp::create(
                rewriter,
                loc,
                rewriter.getIndexAttr(0));

            for (auto [range, pos] :
                 llvm::zip(writeLoopRanges, seenWriteDims)) {
                auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                Value iv = forOp.getInductionVar();
                writeLoopIterators.push_back(iv);
                dimToIterator[pos] = iv;
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }
            for (auto [range, pos] :
                 llvm::zip(reduceLoopRanges, seenReduceDims)) {
                auto forOp = AffineForOp::create(rewriter, loc, 0, range);
                Value iv = forOp.getInductionVar();
                reduceLoopIterators.push_back(iv);
                dimToIterator[pos] = iv;
                rewriter.setInsertionPointToStart(&forOp.getRegion().front());
            }

            // Clone the generic body into the innermost loop, replacing block
            // args with affine.loads and the reduction binary op with
            // helper.accumulate (same pattern as the sliding window case).
            {
                auto &body = *op.getBody();
                auto blockArgs = body.getArguments();
                auto inputOperands = op.getDpsInputs();
                auto outputOperands = op.getDpsInits();
                auto maps = op.getIndexingMapsArray();
                auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

                auto dimIterOperands = llvm::map_to_vector(
                    dimToIterator,
                    [&](Value iv) -> Value { return iv ? iv : indexZero; });

                // Detect accumulate pattern.
                DenseSet<Operation*> accumulateDefOps;
                SmallVector<Operation*> accuDefOps(numOutputs, nullptr);
                SmallVector<unsigned> accuIncrIdx(numOutputs, 0);
                SmallVector<FusedOperator> accuFuseOps(
                    numOutputs,
                    FusedOperator::add);
                detectAccumulatePattern(
                    blockArgs,
                    numInputs,
                    numOutputs,
                    yieldOp,
                    accuDefOps,
                    accuIncrIdx,
                    accuFuseOps,
                    accumulateDefOps);

                // Map block args to affine.load results.
                // For the non-constant input, load from the line buffer:
                //   parallel dims → linebufRowIV (the [0,1) loop IV)
                //   reduction dims → corresponding reduce loop IV
                // All other inputs load from their original memref.
                IRMapping bodyMapping;
                for (unsigned i = 0; i < numInputs; ++i) {
                    auto blkArg = blockArgs[i];
                    if (blkArg.use_empty()) continue;

                    if (linebuf && inputOperands[i] == nonConstInput) {
                        SmallVector<Value> linebufIndices;
                        for (unsigned j = 0;
                             j < nonConstInputMap.getNumResults();
                             ++j) {
                            auto expr = nonConstInputMap.getResult(j);
                            if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                                unsigned pos = dimExpr.getPosition();
                                if (iteratorTypes[pos]
                                    == utils::IteratorType::parallel)
                                    linebufIndices.push_back(linebufRowIV);
                                else
                                    linebufIndices.push_back(
                                        dimToIterator[pos] ? dimToIterator[pos]
                                                           : indexZero);
                            }
                        }
                        bodyMapping.map(
                            blkArg,
                            AffineLoadOp::create(
                                rewriter,
                                loc,
                                linebuf,
                                AffineMap::getMultiDimIdentityMap(
                                    linebufIndices.size(),
                                    ctx),
                                linebufIndices));
                    } else {
                        bodyMapping.map(
                            blkArg,
                            AffineLoadOp::create(
                                rewriter,
                                loc,
                                inputOperands[i],
                                maps[i],
                                dimIterOperands));
                    }
                }

                // Clone body ops, skipping accumulate-pattern ops.
                for (auto &bodyOp : body.without_terminator()) {
                    if (accumulateDefOps.contains(&bodyOp)) continue;
                    rewriter.clone(bodyOp, bodyMapping);
                }

                // Generate helper.accumulate for each output and attach
                // accu_at attribute marking the write dim positions.
                for (auto [defOp, incrIdx, fuseOp, outOperand] : llvm::zip(
                         accuDefOps,
                         accuIncrIdx,
                         accuFuseOps,
                         outputOperands)) {
                    if (!defOp) continue;
                    HelperAccumulateOp::create(
                        rewriter,
                        loc,
                        outOperand,
                        writeLoopIterators,
                        fuseOp,
                        bodyMapping.lookup(defOp->getOperand(incrIdx)));
                    // seenWriteDims[i] ↔ writeLoopIterators[i] by construction.
                    outOperand.getDefiningOp()->setAttr(
                        "accu_at",
                        rewriter.getI32ArrayAttr(
                            SmallVector<int32_t>(
                                seenWriteDims.begin(),
                                seenWriteDims.end())));
                }
            }
            rewriter.eraseOp(op);
            LAKSA_DEBUG(llvm::dbgs() << "Erase original generic operation.");
        }

        return success();
    }

private:
    // Detects `yield(binOp(out_arg, X))` per output.
    void detectAccumulatePattern(
        ArrayRef<BlockArgument> blockArgs,
        unsigned numInputs,
        unsigned numOutputs,
        linalg::YieldOp yieldOp,
        SmallVectorImpl<Operation*> &accuDefOps,
        SmallVectorImpl<unsigned> &accuIncrIdx,
        SmallVectorImpl<FusedOperator> &accuFuseOps,
        DenseSet<Operation*> &opsToSkip) const
    {
        for (unsigned outIdx = 0; outIdx < numOutputs; ++outIdx) {
            Value traced = yieldOp.getValues()[outIdx];
            auto outArg = blockArgs[numInputs + outIdx];
            auto* defOp = traced.getDefiningOp();
            if (!defOp || defOp->getNumOperands() != 2) continue;
            for (unsigned k : {0u, 1u}) {
                if (defOp->getOperand(k) != outArg) continue;
                accuDefOps[outIdx] = defOp;
                accuIncrIdx[outIdx] = 1 - k;
                if (isa<arith::MulIOp>(defOp))
                    accuFuseOps[outIdx] = FusedOperator::mul;
                else if (isa<arith::SubIOp>(defOp))
                    accuFuseOps[outIdx] = FusedOperator::sub;
                opsToSkip.insert(defOp);
                LAKSA_DEBUG(
                    llvm::dbgs() << "Detected accumulate for output " << outIdx
                                 << ": op=" << defOp->getName());
                break;
            }
        }
    }

    SmallVector<Attribute> getFillOrBroadcastAttrs(linalg::GenericOp op) const
    {
        auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
        if (!yieldOp) return SmallVector<Attribute>{};
        if (op.getDpsInputs().empty()) {
            // If the input is empty, it means that the filled value is
            // constant and it's folded. For all yielded values, save the
            // attribute produced by folding their ConstantLike defining op,
            // e.g. arith.constant.
            SmallVector<Attribute> yieldedAttrs;
            for (auto value : yieldOp.getValues()) {
                Attribute yieldedAttr;
                if (!matchPattern(value, m_Constant(&yieldedAttr)))
                    return SmallVector<Attribute>{};
                yieldedAttrs.push_back(yieldedAttr);
                LAKSA_DEBUG(
                    llvm::dbgs() << "Found fill number " << yieldedAttr);
            }
            return yieldedAttrs;
        } else {
            // If there is input memref, it means that this is a broadcast
            // operation with a dense attribute input
            // For all yielded value, save its name attribute defined in the
            // memref.global
            SmallVector<Attribute> yieldedAttrs;
            auto inputValues = op.getDpsInputs();
            for (auto [yieldValue, outputValue] :
                 llvm::zip(yieldOp.getValues(), op.getDpsInits())) {
                auto yieldBlkArg = cast<BlockArgument>(yieldValue);
                auto yieldSrcInput = inputValues[yieldBlkArg.getArgNumber()];
                Attribute yieldedAttr;
                if (auto yieldInputDefOp = dyn_cast<memref::GetGlobalOp>(
                        yieldSrcInput.getDefiningOp())) {
                    auto yieldedMemrefNameAttr = yieldInputDefOp.getNameAttr();
                    yieldedAttrs.push_back(yieldedMemrefNameAttr);
                    LAKSA_DEBUG(
                        llvm::dbgs() << "Found broadcast memref name "
                                     << yieldedMemrefNameAttr);
                } else if (
                    matchPattern(yieldSrcInput, m_Constant(&yieldedAttr))) {
                    yieldedAttrs.push_back(yieldedAttr);
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "Found scalar fill number " << yieldedAttr);
                } else {
                    return SmallVector<Attribute>{};
                }
            }
            return yieldedAttrs;
        }
    }
    std::tuple<bool, unsigned, unsigned>
    analyzeSlidingWindow(linalg::GenericOp op) const
    {
        // Default return values
        unsigned stride = 0, dilation = 0;

        auto iterTypes = op.getIteratorTypesArray();
        // If all the iterator types are parallel, the generic op is legal
        if (llvm::all_of(iterTypes, [](utils::IteratorType iterTy) {
                return iterTy == utils::IteratorType::parallel;
            })) {
            LAKSA_DEBUG(
                llvm::dbgs() << "Found pure parallel generic operation.");
            return std::make_tuple(false, stride, dilation);
        }

        // Check if two dims form a sliding window pair (one parallel, one
        // reduction — order does not matter).
        auto isSlidingWindowPair = [&](unsigned leftPos,
                                       unsigned rightPos) -> bool {
            auto isParallel = [&](unsigned pos) {
                return iterTypes[pos] == utils::IteratorType::parallel;
            };
            return isParallel(leftPos) != isParallel(rightPos);
        };

        auto affineMaps = op.getIndexingMapsArray();
        for (unsigned mapIdx = 0; mapIdx < op.getNumDpsInputs(); ++mapIdx) {
            auto inputMap = affineMaps[mapIdx];
            for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
                auto expr = inputMap.getResult(i);
                bool found = false;
                expr.walk([&](AffineExpr e) {
                    if (auto binOp = dyn_cast<AffineBinaryOpExpr>(e)) {
                        if (binOp.getKind() == AffineExprKind::Add) {
                            auto lhs = binOp.getLHS();
                            auto rhs = binOp.getRHS();

                            // Pattern 1: parallel_dim + reduction_dim
                            if (auto leftDim = dyn_cast<AffineDimExpr>(lhs)) {
                                if (auto rightDim =
                                        dyn_cast<AffineDimExpr>(rhs)) {
                                    if (isSlidingWindowPair(
                                            leftDim.getPosition(),
                                            rightDim.getPosition())) {
                                        found = true;
                                        stride = 1;
                                        dilation = 1;
                                    }
                                }
                            }

                            // Pattern 2: parallel_dim * stride +
                            // reduction_dim
                            if (auto mulExpr =
                                    dyn_cast<AffineBinaryOpExpr>(lhs)) {
                                if (mulExpr.getKind() == AffineExprKind::Mul) {
                                    if (auto dimExpr = dyn_cast<AffineDimExpr>(
                                            mulExpr.getLHS())) {
                                        if (auto rightDim =
                                                dyn_cast<AffineDimExpr>(rhs)) {
                                            if (isSlidingWindowPair(
                                                    dimExpr.getPosition(),
                                                    rightDim.getPosition())) {
                                                found = true;
                                            }
                                            // Store stride
                                            if (auto constExpr = dyn_cast<
                                                    AffineConstantExpr>(
                                                    mulExpr.getRHS())) {
                                                stride = constExpr.getValue();
                                                dilation = 1;
                                            }
                                        }
                                    }
                                }
                            }

                            // Pattern 3: parallel_dim + reduction_dim *
                            // dilation
                            if (auto mulExpr =
                                    dyn_cast<AffineBinaryOpExpr>(rhs)) {
                                if (mulExpr.getKind() == AffineExprKind::Mul) {
                                    if (auto dimExpr = dyn_cast<AffineDimExpr>(
                                            mulExpr.getLHS())) {
                                        if (auto leftDim =
                                                dyn_cast<AffineDimExpr>(lhs)) {
                                            if (isSlidingWindowPair(
                                                    leftDim.getPosition(),
                                                    dimExpr.getPosition())) {
                                                found = true;
                                            }
                                            // Store dilation
                                            if (auto constExpr = dyn_cast<
                                                    AffineConstantExpr>(
                                                    mulExpr.getRHS())) {
                                                stride = 1;
                                                dilation = constExpr.getValue();
                                            }
                                        }
                                    }
                                }
                            }

                            // Pattern 4: parallel_dim * stride +
                            // reduction_dim * dilation
                            if (auto leftMul =
                                    dyn_cast<AffineBinaryOpExpr>(lhs)) {
                                if (auto rightMul =
                                        dyn_cast<AffineBinaryOpExpr>(rhs)) {
                                    if (leftMul.getKind() == AffineExprKind::Mul
                                        && rightMul.getKind()
                                               == AffineExprKind::Mul) {
                                        if (auto leftDim =
                                                dyn_cast<AffineDimExpr>(
                                                    leftMul.getLHS())) {
                                            if (auto rightDim =
                                                    dyn_cast<AffineDimExpr>(
                                                        rightMul.getLHS())) {
                                                if (isSlidingWindowPair(
                                                        leftDim.getPosition(),
                                                        rightDim
                                                            .getPosition())) {
                                                    found = true;
                                                }
                                                // Store stride and dilation
                                                if (auto constExpr = dyn_cast<
                                                        AffineConstantExpr>(
                                                        leftMul.getRHS())) {
                                                    stride =
                                                        constExpr.getValue();
                                                }
                                                if (auto constExpr = dyn_cast<
                                                        AffineConstantExpr>(
                                                        rightMul.getRHS())) {
                                                    dilation =
                                                        constExpr.getValue();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

                if (found) return std::make_tuple(true, stride, dilation);
            }
        }

        return std::make_tuple(false, stride, dilation);
    }
};
} // namespace

void mlir::populateLinalgToLAKSALoopsConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{ patterns.add<GenericToOptimizedLoops>(typeConverter, patterns.getContext()); }

namespace {
struct ConvertLinalgToLAKSALoopsPass
        : public impl::ConvertLinalgToLAKSALoopsBase<
              ConvertLinalgToLAKSALoopsPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertLinalgToLAKSALoopsPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateLinalgToLAKSALoopsConversionPatterns(converter, patterns);

    target.addLegalDialect<AffineDialect, ArithDialect, EmitHLSDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertLinalgToLAKSALoopsPass()
{ return std::make_unique<ConvertLinalgToLAKSALoopsPass>(); }
