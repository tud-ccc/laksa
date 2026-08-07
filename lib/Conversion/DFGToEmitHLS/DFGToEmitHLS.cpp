/// Implementation of DFGToEmitHLS pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/DFGToEmitHLS/DFGToEmitHLS.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "dfg-to-emithls"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(llvm::dbgs() << "[dfg-to-emithls] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTDFGTOEMITHLS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::emithls;

namespace {

// Prints a list of values as `[%0, %1, ...]` for debug logging.
static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, ArrayRef<Value> values)
{
    os << "[";
    for (auto [i, v] : llvm::enumerate(values)) {
        if (i) os << ", ";
        os << v;
    }
    os << "]";
    return os;
}

static Type buildPortType(Type baseType)
{
    if (auto shapedType = dyn_cast<ShapedType>(baseType)) {
        auto streamType = StreamType::get(
            shapedType.getContext(),
            shapedType.getElementType());
        return ArrayType::get(shapedType.getShape(), streamType);
    }
    return StreamType::get(baseType.getContext(), baseType);
}
// process and region share the same NodeInterface-shaped port accessors, so
// this and NodeToFunc below are templated over both.
template<typename NodeOpT>
static FunctionType buildFuncTypeFromNode(NodeOpT op, PatternRewriter &rewriter)
{
    SmallVector<Type> inTypes;
    for (auto type : op.getInputPortTypes())
        inTypes.push_back(buildPortType(type));
    for (auto type : op.getOutputPortTypes())
        inTypes.push_back(buildPortType(type));
    return rewriter.getFunctionType(inTypes, {});
}

// Looks up the unrealized cast a dfg port was mapped to
static FailureOr<Value> getConvertedPort(Value dfgPort)
{
    auto castOp = dfgPort.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getInputs().size() != 1) return failure();
    return castOp.getInputs().front();
}
// Builds one emithls.for nested loop per dimension of the port shape
static void buildStreamReadLoopNest(
    OpBuilder &builder,
    Location loc,
    ArrayRef<int64_t> shape,
    SmallVector<Value> &indices,
    function_ref<void(OpBuilder &, Location, ValueRange)> innermost)
{
    if (indices.size() == shape.size()) {
        innermost(builder, loc, indices);
        return;
    }
    ForOp::create(
        builder,
        loc,
        0,
        shape[indices.size()],
        1,
        [&](OpBuilder &forBuilder, Location forLoc, ValueRange forArgs) {
            indices.push_back(forArgs.front());
            buildStreamReadLoopNest(
                forBuilder,
                forLoc,
                shape,
                indices,
                innermost);
            indices.pop_back();
        });
}

template<typename NodeOpT>
struct NodeToFunc : OpConversionPattern<NodeOpT> {
    using OpConversionPattern<NodeOpT>::OpConversionPattern;

    NodeToFunc(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<NodeOpT>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        NodeOpT op,
        typename NodeOpT::Adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto nodeName = op.getNodeName();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Rewriting " << (isa<dfg::ProcessOp>(op) ? "process" : "region")
            << " \"" << nodeName << "\" at " << loc);

        auto newFuncType = buildFuncTypeFromNode(op, rewriter);
        LAKSA_DEBUG(llvm::dbgs() << "  New func signature is " << newFuncType);
        auto newFuncOp = FuncOp::create(rewriter, loc, nodeName, newFuncType);

        // Build block with new types.
        auto* entryBlock = rewriter.createBlock(
            &newFuncOp.getBody(),
            newFuncOp.getBody().end(),
            newFuncType.getInputs(),
            SmallVector<Location>(newFuncType.getNumInputs(), loc));
        rewriter.setInsertionPointToStart(entryBlock);

        // Cast each new block arg back to the dfg port type it replaces
        auto oldInputTypes = op.getFunctionType().getInputs();
        auto oldOutputTypes = op.getFunctionType().getResults();
        IRMapping mapper;
        for (auto [idx, oldPort] :
             llvm::enumerate(op.getBody().front().getArguments())) {
            Type oldPortType = idx < oldInputTypes.size()
                                   ? oldInputTypes[idx]
                                   : oldOutputTypes[idx - oldInputTypes.size()];
            auto castOp = UnrealizedConversionCastOp::create(
                rewriter,
                loc,
                oldPortType,
                entryBlock->getArgument(idx));
            mapper.map(oldPort, castOp.getResult(0));
        }
        // Copy original content
        for (auto &innerOp : op.getBody().getOps())
            rewriter.clone(innerOp, mapper);
        LAKSA_DEBUG(llvm::dbgs() << "  Cloned body into new func");

        rewriter.eraseOp(op);
        return success();
    }
};
struct EraseLoop : OpConversionPattern<dfg::LoopOp> {
    using OpConversionPattern<dfg::LoopOp>::OpConversionPattern;

    EraseLoop(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::LoopOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        dfg::LoopOp op,
        dfg::LoopOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Erasing loop at " << loc);

        rewriter.inlineBlockBefore(&op.getBody().front(), op);
        LAKSA_DEBUG(llvm::dbgs() << "  Inlined the body");

        rewriter.eraseOp(op);
        return success();
    }
};
struct PullToStreamRead : OpConversionPattern<dfg::PullOp> {
    using OpConversionPattern<dfg::PullOp>::OpConversionPattern;

    PullToStreamRead(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::PullOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        dfg::PullOp op,
        dfg::PullOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting pull at " << loc);

        auto stream = getConvertedPort(op.getReadPort());
        if (failed(stream))
            return rewriter.notifyMatchFailure(
                op,
                "read port not converted yet");
        LAKSA_DEBUG(llvm::dbgs() << "  Found cast port: " << *stream);
        auto readOp = rewriter.replaceOpWithNewOp<StreamReadOp>(
            op,
            op.getToken().getType(),
            *stream,
            op.getIndices());
        LAKSA_DEBUG(llvm::dbgs() << "  Replace pull with " << readOp);
        return success();
    }
};
struct PullMemRefToForLoop : OpConversionPattern<dfg::PullAsMemRefOp> {
    using OpConversionPattern<dfg::PullAsMemRefOp>::OpConversionPattern;

    PullMemRefToForLoop(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::PullAsMemRefOp>(typeConverter, context) {
              };

    LogicalResult matchAndRewrite(
        dfg::PullAsMemRefOp op,
        dfg::PullAsMemRefOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting pull_as_memref at " << loc);

        auto array = getConvertedPort(op.getReadPort());
        if (failed(array))
            return rewriter.notifyMatchFailure(
                op,
                "read port not converted yet");
        LAKSA_DEBUG(llvm::dbgs() << "  Found cast port: " << *array);
        auto memrefType = op.getTokenMemref().getType();
        Value alloc = memref::AllocOp::create(rewriter, loc, memrefType);
        LAKSA_DEBUG(llvm::dbgs() << "  Created buffer to store read values");

        SmallVector<Value> indices;
        LAKSA_DEBUG(llvm::dbgs() << "  Building nested loop");
        buildStreamReadLoopNest(
            rewriter,
            loc,
            memrefType.getShape(),
            indices,
            [&](OpBuilder &loopBuilder,
                Location loopLoc,
                SmallVector<Value> indices) {
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Rewrite pull_as_memref with indices " << indices);
                auto elem = StreamReadOp::create(
                    loopBuilder,
                    loopLoc,
                    memrefType.getElementType(),
                    *array,
                    indices);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Read from the input port using these indices");
                memref::StoreOp::create(
                    loopBuilder,
                    loopLoc,
                    elem,
                    alloc,
                    indices);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Store the temporary value into the buffer");
            });

        rewriter.replaceOp(op, alloc);
        return success();
    }
};
struct PushToStreamWrite : OpConversionPattern<dfg::PushOp> {
    using OpConversionPattern<dfg::PushOp>::OpConversionPattern;

    PushToStreamWrite(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::PushOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        dfg::PushOp op,
        dfg::PushOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting push at " << loc);

        auto stream = getConvertedPort(op.getWritePort());
        if (failed(stream))
            return rewriter.notifyMatchFailure(
                op,
                "write port not converted yet");
        LAKSA_DEBUG(llvm::dbgs() << "  Found cast port: " << *stream);
        auto writeOp = rewriter.replaceOpWithNewOp<StreamWriteOp>(
            op,
            op.getToken(),
            *stream,
            op.getIndices());
        LAKSA_DEBUG(llvm::dbgs() << "  Replace push with " << writeOp);
        return success();
    }
};
struct PushMemRefToForLoop : OpConversionPattern<dfg::PushMemRefOp> {
    using OpConversionPattern<dfg::PushMemRefOp>::OpConversionPattern;

    PushMemRefToForLoop(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::PushMemRefOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        dfg::PushMemRefOp op,
        dfg::PushMemRefOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting push_memref at " << loc);

        auto array = getConvertedPort(op.getWritePort());
        if (failed(array))
            return rewriter.notifyMatchFailure(
                op,
                "write port not converted yet");
        LAKSA_DEBUG(llvm::dbgs() << "  Found cast port: " << *array);
        Value tokenMemref = op.getTokenMemref();
        auto memrefType = op.getTokenMemref().getType();

        SmallVector<Value> indices;
        LAKSA_DEBUG(llvm::dbgs() << "  Building nested loop");
        buildStreamReadLoopNest(
            rewriter,
            loc,
            memrefType.getShape(),
            indices,
            [&](OpBuilder &loopBuilder,
                Location loopLoc,
                SmallVector<Value> indices) {
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Rewrite push_memref with indices " << indices);
                auto elem = memref::LoadOp::create(
                    loopBuilder,
                    loopLoc,
                    tokenMemref,
                    indices);
                LAKSA_DEBUG(llvm::dbgs() << "    Load the value from the buffer");
                StreamWriteOp::create(
                    loopBuilder,
                    loopLoc,
                    elem,
                    *array,
                    indices);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    Write into the output port using these indices");
            });

        rewriter.eraseOp(op);
        return success();
    }
};
struct ChannelToVariable : OpConversionPattern<dfg::ChannelOp> {
    using OpConversionPattern<dfg::ChannelOp>::OpConversionPattern;

    ChannelToVariable(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::ChannelOp>(typeConverter, context) {};

    LogicalResult matchAndRewrite(
        dfg::ChannelOp op,
        dfg::ChannelOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting channel at " << loc);

        auto baseType =
            cast<dfg::InputType>(op.getInputPort().getType()).getBaseType();
        auto portType = buildPortType(baseType);
        auto varOp = VariableOp::create(
            rewriter,
            loc,
            portType,
            /*initNumber=*/Attribute{});
        LAKSA_DEBUG(llvm::dbgs() << "  Replaced channel with variable " << varOp);

        // Cast the emithls.variable to dfg port types
        auto inCast = UnrealizedConversionCastOp::create(
            rewriter,
            loc,
            op.getInputPort().getType(),
            varOp.getResult());
        auto outCast = UnrealizedConversionCastOp::create(
            rewriter,
            loc,
            op.getOutputPort().getType(),
            varOp.getResult());
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  Added unrealized cast to deal with the port types");

        op.getInputPort().replaceAllUsesWith(inCast.getResult(0));
        op.getOutputPort().replaceAllUsesWith(outCast.getResult(0));
        rewriter.eraseOp(op);
        return success();
    }
};
struct InstantiateToCall : OpConversionPattern<dfg::InstantiateOp> {
    using OpConversionPattern<dfg::InstantiateOp>::OpConversionPattern;

    InstantiateToCall(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<dfg::InstantiateOp>(typeConverter, context) {
              };

    LogicalResult matchAndRewrite(
        dfg::InstantiateOp op,
        dfg::InstantiateOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        LAKSA_DEBUG(llvm::dbgs() << "Rewriting instantiate at " << loc);

        // Loop up callee's parameters
        SmallVector<Value> operands;
        for (Value input : op.getInputs()) {
            auto converted = getConvertedPort(input);
            if (failed(converted))
                return rewriter.notifyMatchFailure(
                    op,
                    "input port not converted yet (likely a channel whose "
                    "ChannelToVariable conversion hasn't run yet)");
            operands.push_back(*converted);
        }
        for (Value output : op.getOutputs()) {
            auto converted = getConvertedPort(output);
            if (failed(converted))
                return rewriter.notifyMatchFailure(
                    op,
                    "output port not converted yet (likely a channel whose "
                    "ChannelToVariable conversion hasn't run yet)");
            operands.push_back(*converted);
        }

        auto calleeFunc =
            SymbolTable::lookupNearestSymbolFrom<FuncOp>(op, op.getActorAttr());
        if (!calleeFunc)
            return rewriter.notifyMatchFailure(
                op,
                "callee has not been converted to an emithls.func yet");

        rewriter.replaceOpWithNewOp<CallOp>(op, calleeFunc, operands);
        LAKSA_DEBUG(llvm::dbgs() << "  Replace it with a call to function");
        return success();
    }
};
} // namespace

void mlir::populateDFGToEmitHLSConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns)
{
    patterns.add<NodeToFunc<dfg::ProcessOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<NodeToFunc<dfg::RegionOp>>(
        typeConverter,
        patterns.getContext());
    patterns.add<EraseLoop>(typeConverter, patterns.getContext());
    patterns.add<PullToStreamRead>(typeConverter, patterns.getContext());
    patterns.add<PullMemRefToForLoop>(typeConverter, patterns.getContext());
    patterns.add<PushToStreamWrite>(typeConverter, patterns.getContext());
    patterns.add<PushMemRefToForLoop>(typeConverter, patterns.getContext());
    patterns.add<ChannelToVariable>(typeConverter, patterns.getContext());
    patterns.add<InstantiateToCall>(typeConverter, patterns.getContext());
}

namespace {
struct ConvertDFGToEmitHLSPass
        : public impl::ConvertDFGToEmitHLSBase<ConvertDFGToEmitHLSPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertDFGToEmitHLSPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    populateDFGToEmitHLSConversionPatterns(converter, patterns);

    target.addLegalDialect<EmitHLSDialect>();
    target.addIllegalDialect<dfg::DFGDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertDFGToEmitHLSPass()
{ return std::make_unique<ConvertDFGToEmitHLSPass>(); }
