add_mlir_upstream_c_api_library(MLIRCAPIDFG
        DFG.cpp
        DFGPasses.cpp

    DEPENDS
        DFGPassesIncGen

    LINK_LIBS PUBLIC
        MLIRCAPIIR
        DFGIR
        DFGTransforms
        MLIRTargetDFGToDot
)
