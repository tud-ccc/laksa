add_mlir_conversion_library(ArithToEmitHLS
    ArithToEmitHLS.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRArithDialect
        EmitHLSIR
)
