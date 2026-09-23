include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `MLIR_PYTHON_PACKAGE_PREFIX.`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(LAKSAPythonSources)

declare_mlir_python_sources(LAKSAPythonSources.SiteInit
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    SOURCES
        _mlir_libs/_site_initialize_0.py
)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    TD_FILE dialects/DFGOps.td
    SOURCES
        dialects/dfg.py
    DIALECT_NAME dfg
    GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    TD_FILE dialects/EmitHLSOps.td
    SOURCES
        dialects/emithls.py
    DIALECT_NAME emithls
    GEN_ENUM_BINDINGS
)

declare_mlir_python_sources(LAKSAPythonSources.FuncDialect
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    SOURCES
        dialects/func_ext.py
)

declare_mlir_python_sources(LAKSAPythonSources.LinalgDialect
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    SOURCES
        dialects/linalg_ext.py
)

declare_mlir_python_sources(LAKSAPythonSources.Conversion
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_laksa"
    SOURCES
        conversion.py
)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

set(PYTHON_SOURCE_DIR "${CMAKE_SOURCE_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(LAKSAPythonSources.NanobindExtension
    MODULE_NAME _mlirDialectsDFG
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
        DialectDFG.cpp
    PRIVATE_LINK_LIBS
        LLVMSupport
    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPIDFG
)

declare_mlir_python_extension(LAKSAPythonSources.EmitHLSNanobindExtension
    MODULE_NAME _mlirDialectsEmitHLS
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
        DialectEmitHLS.cpp
    PRIVATE_LINK_LIBS
        LLVMSupport
    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPIEmitHLS
)

declare_mlir_python_extension(LAKSAPythonSources.ConversionNanobindExtension
    MODULE_NAME _mlirConversion
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
        Conversion.cpp
    PRIVATE_LINK_LIBS
        LLVMSupport
    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPILAKSAConversion
)

declare_mlir_python_extension(LAKSAPythonSources.FuncNanobindExtension
    MODULE_NAME _mlirDialectsFuncExt
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
        DialectFunc.cpp
    PRIVATE_LINK_LIBS
        LLVMSupport
    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPIFuncExt
)

declare_mlir_python_extension(LAKSAPythonSources.LinalgNanobindExtension
    MODULE_NAME _mlirDialectsLinalgExt
    ADD_TO_PARENT LAKSAPythonSources
    ROOT_DIR "${PYTHON_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
        DialectLinalg.cpp
    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPILinalgExt
)

################################################################################
# Common CAPI library
################################################################################

add_mlir_python_common_capi_library(LAKSAPythonCAPI
    INSTALL_COMPONENT LAKSAPythonModules
    INSTALL_DESTINATION ${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs
    OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
    RELATIVE_INSTALL_ROOT "../../.."
    DECLARED_SOURCES
        LAKSAPythonSources
        MLIRPythonExtension.RegisterEverything
        MLIRPythonSources
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(StandalonePythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

add_mlir_python_modules(LAKSAPythonModules
    ROOT_PREFIX "${StandalonePythonModules_ROOT_PREFIX}"
    INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
    DECLARED_SOURCES
        LAKSAPythonSources
        MLIRPythonExtension.RegisterEverything
        MLIRPythonSources
    COMMON_CAPI_LINK_LIBS
        LAKSAPythonCAPI
)

find_program(UV_EXECUTABLE uv)
if(UV_EXECUTABLE)
    set(_pip_cmd ${UV_EXECUTABLE} pip install -p ${Python3_EXECUTABLE})
else()
    set(_pip_cmd ${Python3_EXECUTABLE} -m pip install)
endif()

add_custom_target(install-python-package
    COMMAND ${_pip_cmd} "scikit-build-core>=0.9"
    COMMAND ${_pip_cmd}
        --no-build-isolation
        -e ${CMAKE_SOURCE_DIR}
        -C cmake.define.MLIR_DIR=${MLIR_DIR}
        -C cmake.define.LLVM_DIR=${LLVM_DIR}
        -C cmake.define.GUROBI_DIR=${GUROBI_DIR}
    DEPENDS LAKSAPythonModules
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Installing mlir-laksa Python package"
    USES_TERMINAL
    VERBATIM
)

add_custom_target(check-laksa-python
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_examples.py
        --laksa-opt $<TARGET_FILE:laksa-opt>
        --python ${Python3_EXECUTABLE}
    DEPENDS laksa-opt install-python-package
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking python/examples/*.py against their test/*.mlir reference output"
    USES_TERMINAL
)
