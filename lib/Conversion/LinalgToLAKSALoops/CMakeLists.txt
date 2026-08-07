add_mlir_conversion_library(LinalgToLAKSALoops
    LinalgToLAKSALoops.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRAffineDialect
        MLIRArithDialect
        EmitHLSIR
)
