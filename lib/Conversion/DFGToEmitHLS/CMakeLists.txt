add_mlir_conversion_library(DFGToEmitHLS
    DFGToEmitHLS.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        DFGIR
        EmitHLSIR
)
