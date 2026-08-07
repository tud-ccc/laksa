add_mlir_upstream_c_api_library(MLIRCAPIEmitHLS
        EmitHLS.cpp
        EmitHLSPasses.cpp

    DEPENDS
        EmitHLSPassesIncGen

    LINK_LIBS PUBLIC
        MLIRCAPIIR
        EmitHLSIR
        EmitHLSTransforms
        MLIRTargetEmitHLSToCpp
)