/// Implementation of FuncToDFG pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/FuncToDFG/FuncToDFG.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#define DEBUG_TYPE "func-to-dfg"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(llvm::dbgs() << "[func-to-dfg] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTODFG
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dfg;

namespace {
struct NodeFuncToOperator : OpConversionPattern<func::FuncOp> {
    NodeFuncToOperator(TypeConverter &typeConverter, MLIRContext* context)
            : OpConversionPattern<func::FuncOp>(
                  typeConverter,
                  context,
                  /*benefit=*/1) {};

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (op->hasAttr(laksa::kRootAttrName)) return failure();
        LAKSA_DEBUG(
            llvm::dbgs() << "Found node function @" << op.getSymName() << " at "
                         << op.getLoc());
        rewriter.replaceOpWithNewOp<OperatorOp>(
            op,
            op.getSymName(),
            op.getFunctionType(),
            [&](OpBuilder &operatorBuilder,
                Location operatorLoc,
                ValueRange operatorBlkArgs) {
                IRMapping mapper;
                for (auto [oldArg, newArg] :
                     llvm::zip(op.getBody().getArguments(), operatorBlkArgs))
                    mapper.map(oldArg, newArg);
                for (auto &opFunc : op.getBody().getOps()) {
                    if (auto retOp = dyn_cast<func::ReturnOp>(&opFunc)) {
                        SmallVector<Value> outputs;
                        for (auto operand : retOp.getOperands())
                            outputs.push_back(mapper.lookup(operand));
                        OutputOp::create(operatorBuilder, operatorLoc, outputs);
                        LAKSA_DEBUG(
                            llvm::dbgs()
                            << "Replace return function to operator's output.");
                    } else {
                        operatorBuilder.clone(opFunc, mapper);
                    }
                }
            });
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Replace the node function with operator operation.");
        return success();
    }
};

struct GraphFuncToRegion : OpConversionPattern<func::FuncOp> {
    GraphFuncToRegion(
        TypeConverter &typeConverter,
        MLIRContext* context,
        DenseMap<Value, Value> &returnValueToOutputPortMap)
            : OpConversionPattern<func::FuncOp>(
                  typeConverter,
                  context,
                  /*benefit=*/2),
              returnValueToOutputPortMap(returnValueToOutputPortMap) {};

    LogicalResult matchAndRewrite(
        func::FuncOp op,
        func::FuncOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        if (!op->hasAttr(laksa::kRootAttrName)) return failure();
        LAKSA_DEBUG(
            llvm::dbgs() << "Found graph function @" << op.getSymName()
                         << " at " << op.getLoc());
        auto regionOp = rewriter.replaceOpWithNewOp<RegionOp>(
            op,
            op.getSymName(),
            op.getFunctionType(),
            [&](OpBuilder &operatorBuilder,
                Location,
                ValueRange operatorBlkArgs) {
                IRMapping mapper;
                for (auto [oldArg, newArg] :
                     llvm::zip(op.getBody().getArguments(), operatorBlkArgs))
                    mapper.map(oldArg, newArg);
                unsigned numInputs = op.getFunctionType().getNumInputs();
                for (auto &opFunc : op.getBody().getOps()) {
                    if (auto retOp = dyn_cast<func::ReturnOp>(&opFunc)) {
                        for (auto [idx, operand] :
                             llvm::enumerate(retOp.getOperands()))
                            returnValueToOutputPortMap.insert(
                                {mapper.lookup(operand),
                                 operatorBlkArgs[numInputs + idx]});
                        LAKSA_DEBUG(
                            llvm::dbgs() << "Saved return values and erase "
                                            "return operation.");
                    } else {
                        operatorBuilder.clone(opFunc, mapper);
                    }
                }
            });
        regionOp->setAttr(
            laksa::kRootAttrName,
            UnitAttr::get(rewriter.getContext()));
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Replace the graph function with region operation.");
        return success();
    }

private:
    DenseMap<Value, Value> &returnValueToOutputPortMap;
};

struct CallToInstantiate : OpConversionPattern<func::CallOp> {
    CallToInstantiate(
        TypeConverter &typeConverter,
        MLIRContext* context,
        DenseMap<Value, Value> &returnValueToOutputPortMap,
        DenseMap<Value, Value> &callResultChannelMap)
            : OpConversionPattern<func::CallOp>(typeConverter, context),
              returnValueToOutputPortMap(returnValueToOutputPortMap),
              callResultChannelMap(callResultChannelMap) {};

    LogicalResult matchAndRewrite(
        func::CallOp op,
        func::CallOpAdaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto nodeName = op.getCallee();

        SmallVector<Value> inputs, outputs;
        for (auto operand : op.getOperands()) {
            if (isa<BlockArgument>(operand)) {
                inputs.push_back(operand);
                LAKSA_DEBUG(
                    llvm::dbgs() << "Found input value as block argument.");
            } else {
                auto it = callResultChannelMap.find(operand);
                if (it == callResultChannelMap.end())
                    return rewriter.notifyMatchFailure(
                        loc,
                        "Cannot find channel port map.");
                inputs.push_back(it->second);
                LAKSA_DEBUG(llvm::dbgs() << "Use channel's input port as input.");
            }
        }
        for (auto result : op.getResults()) {
            auto it = returnValueToOutputPortMap.find(result);
            if (it != returnValueToOutputPortMap.end()) {
                outputs.push_back(it->second);
                LAKSA_DEBUG(llvm::dbgs() << "Found return value to output.");
            } else {
                auto newChannel =
                    ChannelOp::create(rewriter, loc, result.getType());
                callResultChannelMap.insert(
                    {result, newChannel.getOutputPort()});
                outputs.push_back(newChannel.getInputPort());
                LAKSA_DEBUG(llvm::dbgs() << "Created ChannelOp for call result.");
            }
        }

        InstantiateOp::create(rewriter, loc, nodeName, inputs, outputs);
        LAKSA_DEBUG(llvm::dbgs() << "Created InstantiateOp to replace CallOp.");
        rewriter.eraseOp(op);
        return success();
    }

private:
    DenseMap<Value, Value> &returnValueToOutputPortMap;
    DenseMap<Value, Value> &callResultChannelMap;
};
} // namespace

void mlir::populateFuncToDFGConversionPatterns(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    DenseMap<Value, Value> &returnValueToOutputPortMap,
    DenseMap<Value, Value> &callResultChannelMap)
{
    patterns.add<NodeFuncToOperator>(typeConverter, patterns.getContext());
    patterns.add<GraphFuncToRegion>(
        typeConverter,
        patterns.getContext(),
        returnValueToOutputPortMap);
    patterns.add<CallToInstantiate>(
        typeConverter,
        patterns.getContext(),
        returnValueToOutputPortMap,
        callResultChannelMap);
}

namespace {
struct ConvertFuncToDFGPass
        : public impl::ConvertFuncToDFGBase<ConvertFuncToDFGPass> {
    void runOnOperation() final;
};
} // namespace

void ConvertFuncToDFGPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter converter;
    converter.addConversion([&](Type type) { return type; });

    DenseMap<Value, Value> returnValueToOutputPortMap;
    DenseMap<Value, Value> callResultChannelMap;
    populateFuncToDFGConversionPatterns(
        converter,
        patterns,
        returnValueToOutputPortMap,
        callResultChannelMap);

    target.addLegalDialect<DFGDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::createConvertFuncToDFGPass()
{ return std::make_unique<ConvertFuncToDFGPass>(); }
