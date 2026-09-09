//===- DialectEmitHLS.cpp - Nanobind module for EmitHLS dialect -----------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

/// Collects what `translator` streams out into a string, raising a Python
/// ValueError carrying `errorMessage` if it fails.
template<typename TranslateFn>
static std::string translateToString(
    MlirOperation op,
    TranslateFn translator,
    const char* errorMessage)
{
    std::string result;
    MlirLogicalResult res = translator(
        op,
        [](MlirStringRef str, void* userData) {
            static_cast<std::string*>(userData)->append(str.data, str.length);
        },
        &result);
    if (mlirLogicalResultIsFailure(res)) throw nb::value_error(errorMessage);
    return result;
}

static void populateDialectEmitHLSSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // EmitHLS dialect registration
    //===--------------------------------------------------------------------===//
    auto emithls = m.def_submodule("emithls");

    emithls.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__emithls__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    emithls.def("register_passes", []() { mlirRegisterEmitHLSPasses(); });

    emithls.def("add_add_io_functions_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSAddIOFunctions());
    });
    emithls.def("add_fold_expression_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSFoldExpression());
    });
    emithls.def("add_fuse_operator_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateEmitHLSEmitHLSFuseOperator());
    });
    emithls.def("add_insert_includes_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSInsertIncludes());
    });
    emithls.def("add_loop_fusion_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateEmitHLSEmitHLSLoopFusion());
    });
    emithls.def("add_merge_cast_chain_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSMergeCastChain());
    });
    emithls.def(
        "add_pragma_dse_pass",
        [](MlirPassManager pm, int64_t availableBRAM, int64_t availableDSP) {
            mlirPassManagerAddOwnedPass(
                pm,
                mlirCreateEmitHLSEmitHLSPragmaDSEWithOptions(
                    availableBRAM,
                    availableDSP));
        },
        nb::arg("pm"),
        nb::arg("available_bram") = 288,
        nb::arg("available_dsp") = 1248);
    emithls.def("add_pragma_insertion_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSPragmaInsertion());
    });
    emithls.def("add_resolve_helpers_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateEmitHLSEmitHLSResolveHelpers());
    });

    emithls.def(
        "translate_to_cpp",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToCpp,
                "Translation to HLS C++ failed.");
        },
        nb::arg("op"));

    emithls.def(
        "translate_to_hls_tcl",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToHLSTcl,
                "Translation to a Vitis HLS run_hls.tcl script failed.");
        },
        nb::arg("op"));

    emithls.def(
        "translate_to_kria_dtsi",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToKriaDtsi,
                "Translation to a Kria device tree overlay source failed.");
        },
        nb::arg("op"));

    emithls.def(
        "translate_to_laksa_app",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToLaksaApp,
                "Translation to a laksa-hls-kria-driver application failed.");
        },
        nb::arg("op"));

    emithls.def(
        "translate_to_laksa_header",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToLaksaHeader,
                "Translation to a laksa-hls-kria-driver header failed.");
        },
        nb::arg("op"));

    emithls.def(
        "translate_to_vivado_tcl",
        [](MlirOperation op) -> std::string {
            return translateToString(
                op,
                mlirTranslateEmitHLSToVivadoTcl,
                "Translation to a Vivado run_vivado.tcl script failed.");
        },
        nb::arg("op"));

    //===--------------------------------------------------------------------===//
    // StreamType
    //===--------------------------------------------------------------------===//
    auto streamType =
        mlir_type_subclass(m, "StreamType", mlirTypeIsAEmitHLSStreamType);

    streamType.def_classmethod(
        "get",
        [](nb::object cls, MlirContext context, MlirType elementType) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirEmitHLSStreamTypeGet(elementType);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("element_type"));

    streamType.def_property_readonly("element_type", [](MlirType self) {
        return mlirEmitHLSStreamTypeGetElementType(self);
    });

    //===--------------------------------------------------------------------===//
    // PointerType
    //===--------------------------------------------------------------------===//
    auto pointerType =
        mlir_type_subclass(m, "PointerType", mlirTypeIsAEmitHLSPointerType);

    pointerType.def_classmethod(
        "get",
        [](nb::object cls, MlirContext context, MlirType elementType) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirEmitHLSPointerTypeGet(elementType);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("element_type"));

    pointerType.def_property_readonly("element_type", [](MlirType self) {
        return mlirEmitHLSPointerTypeGetElementType(self);
    });

    //===--------------------------------------------------------------------===//
    // ArrayType
    //===--------------------------------------------------------------------===//
    auto arrayType =
        mlir_type_subclass(m, "ArrayType", mlirTypeIsAEmitHLSArrayType);

    arrayType.def_classmethod(
        "get",
        [](nb::object cls,
           MlirContext context,
           MlirType elementType,
           std::vector<int64_t> shape) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = mlirEmitHLSArrayTypeGet(
                static_cast<intptr_t>(shape.size()),
                shape.data(),
                elementType);
            if (mlirTypeIsNull(type))
                throw nb::value_error(scope.takeMessage().c_str());
            return cls(type);
        },
        nb::arg("cls"),
        nb::arg("context").none() = nb::none(),
        nb::arg("element_type"),
        nb::arg("shape"));

    arrayType.def_property_readonly("shape", [](MlirType self) {
        intptr_t rank = mlirEmitHLSArrayTypeGetRank(self);
        std::vector<int64_t> shape(rank);
        for (intptr_t i = 0; i < rank; ++i)
            shape[i] = mlirEmitHLSArrayTypeGetDimSize(self, i);
        return shape;
    });

    arrayType.def_property_readonly("element_type", [](MlirType self) {
        return mlirEmitHLSArrayTypeGetElementType(self);
    });
}

NB_MODULE(_mlirDialectsEmitHLS, m)
{
    m.doc() = "EmitHLS dialect.";

    mlirRegisterEmitHLSPasses();
    populateDialectEmitHLSSubmodule(m);
}
