//===- DialectLinalg.cpp - Nanobind module for extended Linalg passes -----===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/Linalg.h"
#include "mlir-c/Pass.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

static void populateDialectLinalgSubmodule(nb::module_ m)
{
    auto linalg_ext = m.def_submodule("linalg_ext");
    linalg_ext.def("register_passes", []() { mlirRegisterLinalgExtPasses(); });

    linalg_ext.def("add_map_to_generic_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLinalgExtLinalgMapToGeneric());
    });
    linalg_ext.def("add_scalarize_splat_dense_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLinalgExtLinalgScalarizeSplatDense());
    });
    linalg_ext.def("add_soft_transpose_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLinalgExtLinalgSoftTranspose());
    });
}

NB_MODULE(_mlirDialectsLinalgExt, m)
{
    m.doc() = "Extended Linalg dialect passes.";

    mlirRegisterLinalgExtPasses();
    populateDialectLinalgSubmodule(m);
}
