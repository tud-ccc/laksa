/// Implementation of FuncRemoveOutlinedFunctionWrappers transform pass.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Dialect/Func/Transforms/Passes.h"

#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace func {
#define GEN_PASS_DEF_FUNCREMOVEOUTLINEDFUNCTIONWRAPPERS
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h.inc"
} // namespace func
} // namespace mlir

using namespace mlir;

namespace {

struct FuncRemoveOutlinedFunctionWrappersPass
        : public func::impl::FuncRemoveOutlinedFunctionWrappersBase<
              FuncRemoveOutlinedFunctionWrappersPass> {
    void runOnOperation() final
    {
        SmallVector<func::FuncOp> wrappers;
        for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>())
            if (funcOp->hasAttr(laksa::kRootAttrName))
                wrappers.push_back(funcOp);

        for (func::FuncOp wrapper : wrappers) wrapper.erase();
    }
};

} // namespace

std::unique_ptr<Pass>
mlir::func::createFuncRemoveOutlinedFunctionWrappersPass()
{
    return std::make_unique<FuncRemoveOutlinedFunctionWrappersPass>();
}
