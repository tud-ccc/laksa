add_mlir_conversion_library(AffineToEmitHLS
    AffineToEmitHLS.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRAffineDialect
        EmitHLSIR
)
