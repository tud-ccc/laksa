//===- Conversion.cpp - Nanobind module for conversion passes -------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Conversion.h"

#include "mlir-c/Pass.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_mlirConversion, m)
{
    m.doc() = "LAKSA conversion passes.";

    mlirRegisterLAKSAConversionPasses();
    mlirRegisterLAKSAConvertToEmitHLSPipelines();

    m.def("register_passes", []() { mlirRegisterLAKSAConversionPasses(); });
    m.def("register_pipelines", []() {
        mlirRegisterLAKSAConvertToEmitHLSPipelines();
    });

    m.def("add_affine_to_emithls_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertAffineToEmitHLS());
    });
    m.def("add_arith_to_emithls_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertArithToEmitHLS());
    });
    m.def("add_dfg_to_emithls_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertDFGToEmitHLS());
    });
    m.def("add_func_to_dfg_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertFuncToDFG());
    });
    m.def("add_index_to_emithls_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertIndexToEmitHLS());
    });
    m.def("add_linalg_to_laksa_loops_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertLinalgToLAKSALoops());
    });
    m.def("add_memref_pad_to_laksa_loops_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateLAKSAConversionConvertMemRefPadToLAKSALoops());
    });

    m.def(
        "add_convert_to_emithls_pipeline",
        [](MlirPassManager pm, int64_t availableBRAM, int64_t availableDSP) {
            mlirConversionAddConvertToEmitHLSPasses(
                mlirPassManagerGetAsOpPassManager(pm),
                availableBRAM,
                availableDSP);
        },
        nb::arg("pm"),
        nb::arg("available_bram") = 288,
        nb::arg("available_dsp") = 1248);
}
