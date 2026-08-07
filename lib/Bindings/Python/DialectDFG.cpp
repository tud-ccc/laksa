//===- DialectDFG.cpp - Pybind module for DFG dialect API support ---------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/DFG.h"
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

static void populateDialectDFGSubmodule(nb::module_ m)
{
    //===--------------------------------------------------------------------===//
    // DFG dialect registration
    //===--------------------------------------------------------------------===//
    auto dfg = m.def_submodule("dfg");

    dfg.def(
        "register_dialect",
        [](MlirContext context, bool load) {
            MlirDialectHandle handle = mlirGetDialectHandle__dfg__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) mlirDialectHandleLoadDialect(handle, context);
        },
        nb::arg("context").none() = nb::none(),
        nb::arg("load") = true);

    dfg.def("register_passes", []() { mlirRegisterDFGPasses(); });

    dfg.def(
        "register_bufferizable_op_interface_external_models",
        [](MlirContext context) {
            mlirDFGRegisterBufferizableOpInterfaceExternalModels(context);
        },
        nb::arg("context").none() = nb::none());

    dfg.def("add_channel_fanout_expansion_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(
            pm,
            mlirCreateDFGDFGChannelFanoutExpansion());
    });

    dfg.def("add_collapse_unit_dims_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateDFGDFGCollapseUnitDims());
    });

    dfg.def("add_inline_embed_region_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateDFGDFGInlineEmbedRegion());
    });

    dfg.def("add_io_normalization_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateDFGDFGIONormalization());
    });

    dfg.def("add_operator_to_process_pass", [](MlirPassManager pm) {
        mlirPassManagerAddOwnedPass(pm, mlirCreateDFGDFGOperatorToProcess());
    });

    dfg.def(
        "translate_to_dot",
        [](MlirOperation op) -> std::string {
            std::string result;
            MlirLogicalResult res = mlirTranslateDFGToDot(
                op,
                [](MlirStringRef str, void* userData) {
                    static_cast<std::string*>(userData)->append(
                        str.data,
                        str.length);
                },
                &result);
            if (mlirLogicalResultIsFailure(res))
                throw nb::value_error("Translation to Dot failed.");
            return result;
        },
        nb::arg("op"));

    //===--------------------------------------------------------------------===//
    // InputType
    //===--------------------------------------------------------------------===//
    auto inputType = mlir_type_subclass(m, "InputType", mlirTypeIsAInputType);

    inputType.def_classmethod(
        "get",
        [](nb::object cls,
           MlirContext context,
           MlirType elementType,
           std::vector<int64_t> shape) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = shape.empty()
                                ? mlirInputTypeGet(elementType)
                                : mlirInputTypeGetWithShape(
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
        nb::arg("shape") = std::vector<int64_t>{});

    inputType.def_property_readonly("shape", [](MlirType self) {
        intptr_t rank = mlirInputTypeGetRank(self);
        std::vector<int64_t> shape(rank);
        for (intptr_t i = 0; i < rank; ++i)
            shape[i] = mlirInputTypeGetDimSize(self, i);
        return shape;
    });

    inputType.def_property_readonly("element_type", [](MlirType self) {
        return mlirInputTypeGetElementType(self);
    });

    //===--------------------------------------------------------------------===//
    // OutputType
    //===--------------------------------------------------------------------===//
    auto outputType =
        mlir_type_subclass(m, "OutputType", mlirTypeIsAOutputType);

    outputType.def_classmethod(
        "get",
        [](nb::object cls,
           MlirContext context,
           MlirType elementType,
           std::vector<int64_t> shape) {
            CollectDiagnosticsToStringScope scope(context);
            MlirType type = shape.empty()
                                ? mlirOutputTypeGet(elementType)
                                : mlirOutputTypeGetWithShape(
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
        nb::arg("shape") = std::vector<int64_t>{});

    outputType.def_property_readonly("shape", [](MlirType self) {
        intptr_t rank = mlirOutputTypeGetRank(self);
        std::vector<int64_t> shape(rank);
        for (intptr_t i = 0; i < rank; ++i)
            shape[i] = mlirOutputTypeGetDimSize(self, i);
        return shape;
    });

    outputType.def_property_readonly("element_type", [](MlirType self) {
        return mlirOutputTypeGetElementType(self);
    });
}

NB_MODULE(_mlirDialectsDFG, m)
{
    m.doc() = "DFG dialect.";

    mlirRegisterDFGPasses();
    populateDialectDFGSubmodule(m);
}
