add_mlir_upstream_c_api_library(MLIRCAPILinalgExt
        Linalg.cpp

    DEPENDS
        LinalgExtPassesIncGen

    LINK_LIBS PUBLIC
        MLIRCAPIIR
        LinalgExtTransforms
)
