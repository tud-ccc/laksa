add_mlir_library(MLIRTargetLaksaCommon
    BufferInfo.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        EmitHLSIR
)
