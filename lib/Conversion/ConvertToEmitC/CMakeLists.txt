add_mlir_conversion_library(ConvertToEmitC
    ConvertToEmitC.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRPass
        ReshapedCopyToLoops
)
