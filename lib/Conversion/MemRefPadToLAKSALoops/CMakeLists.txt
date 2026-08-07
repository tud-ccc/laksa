add_mlir_conversion_library(MemRefPadToLAKSALoops
    MemRefPadToLAKSALoops.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRAffineDialect
        MLIRMemRefDialect
        DFGIR
        EmitHLSIR
)
