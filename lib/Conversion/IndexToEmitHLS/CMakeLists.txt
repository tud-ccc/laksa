add_mlir_conversion_library(IndexToEmitHLS
    IndexToEmitHLS.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRIndexDialect
        EmitHLSIR
)
