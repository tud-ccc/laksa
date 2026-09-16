add_mlir_conversion_library(ConvertToDFG
    ConvertToDFG.cpp

    DEPENDS
        LAKSAConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRPass
)
