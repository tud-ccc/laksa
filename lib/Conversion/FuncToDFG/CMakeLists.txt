add_mlir_conversion_library(FuncToDFG
    FuncToDFG.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRFuncDialect
        DFGIR
)
