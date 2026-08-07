add_mlir_upstream_c_api_library(MLIRCAPIFuncExt
        Func.cpp

    DEPENDS
        FuncExtPassesIncGen

    LINK_LIBS PUBLIC
        MLIRCAPIIR
        FuncExtTransforms
)
