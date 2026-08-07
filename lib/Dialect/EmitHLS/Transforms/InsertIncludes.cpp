/// Implementation of InsertIncludes transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "emithls-insert-includes"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-insert-includes] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSINSERTINCLUDES
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {
struct EmitHLSInsertIncludesPass
        : public emithls::impl::EmitHLSInsertIncludesBase<
              EmitHLSInsertIncludesPass> {
    void runOnOperation() override;
};
} // namespace

void EmitHLSInsertIncludesPass::runOnOperation()
{
    ModuleOp module = cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&module.getBodyRegion().front());
    auto loc = module.getLoc();

    bool hasSizeT = false, hasStream = false, hasInteger = false,
         hasAlgo = false;

    auto checkLeafType = [&](Type type) {
        if (isa<IntegerType>(type)) hasInteger = true;
        if (isa<IndexType>(type)) hasSizeT = true;
    };
    auto checkType = [&](Type type) {
        checkLeafType(type);
        if (auto streamTy = dyn_cast<StreamType>(type)) {
            hasStream = true;
            checkLeafType(streamTy.getElementType());
        } else if (auto arrTy = dyn_cast<ArrayType>(type)) {
            if (auto streamTy = dyn_cast<StreamType>(arrTy.getElementType())) {
                hasStream = true;
                checkLeafType(streamTy.getElementType());
            }
        }
    };

    module.walk([&](Operation* op) {
        for (auto type : op->getOperandTypes()) checkType(type);
        for (auto type : op->getResultTypes()) checkType(type);
        if (auto funcOp = dyn_cast<FuncOp>(op))
            for (auto type : funcOp.getFunctionType().getInputs())
                checkType(type);
        if (isa<ArithMinOp, ArithMaxOp>(op)) hasAlgo = true;
    });

    if (hasSizeT) {
        IncludeOp::create(builder, loc, builder.getStringAttr("cstddef"));
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Found index type usage, insert \"cstddef\" header.");
    }
    if (hasAlgo) {
        IncludeOp::create(builder, loc, builder.getStringAttr("algorithm"));
        LAKSA_DEBUG(
            llvm::dbgs() << "Found arith min/max operation usage, insert "
                            "\"algorithm\" header.");
    }
    if (hasInteger) {
        IncludeOp::create(builder, loc, builder.getStringAttr("ap_int.h"));
        LAKSA_DEBUG(
            llvm::dbgs() << "Found integer usage, insert \"ap_int\" header.");
    }
    if (hasStream) {
        IncludeOp::create(builder, loc, builder.getStringAttr("hls_stream.h"));
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Found stream type usage, insert \"hls_stream\" header.");
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSInsertIncludesPass()
{ return std::make_unique<EmitHLSInsertIncludesPass>(); }
