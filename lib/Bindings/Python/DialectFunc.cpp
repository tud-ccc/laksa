//===- DialectFunc.cpp - Nanobind module for Func dialect passes ----------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/Func.h"
#include "mlir-c/Pass.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

static void populateDialectFuncSubmodule(nb::module_ m)
{
    auto func_ext = m.def_submodule("func_ext");
    func_ext.def("register_passes", []() { mlirRegisterFuncExtPasses(); });

    func_ext.def("add_outline_computation_leaf_pass", [](MlirPassManager pm) {
        // FuncOutlineComputationLeaf is anchored to `func::FuncOp` (not a
        // generic "any op" pass), so it must run in a nested pass manager.
        MlirOpPassManager nested = mlirPassManagerGetNestedUnder(
            pm,
            mlirStringRefCreateFromCString("func.func"));
        mlirOpPassManagerAddOwnedPass(
            nested,
            mlirCreateFuncExtFuncOutlineComputationLeaf());
    });
}

NB_MODULE(_mlirDialectsFuncExt, m)
{
    m.doc() = "Func dialect passes.";

    mlirRegisterFuncExtPasses();
    populateDialectFuncSubmodule(m);
}
