add_mlir_conversion_library(ConvertToEmitHLS
    ConvertToEmitHLS.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRPass
)
