add_mlir_conversion_library(ReshapedCopyToLoops
    ReshapedCopyToLoops.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRTransformUtils
)
