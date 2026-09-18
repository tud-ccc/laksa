add_mlir_dialect_library(FuncExtTransforms
        OutlineComputationLeaf.cpp
        RemoveOutlinedFunctionWrappers.cpp

    ENABLE_AGGREGATION

    DEPENDS
        FuncExtPassesIncGen

    LINK_LIBS PUBLIC
        MLIRFuncDialect
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
