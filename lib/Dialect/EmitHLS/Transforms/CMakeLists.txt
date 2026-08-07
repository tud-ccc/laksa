add_mlir_dialect_library(EmitHLSTransforms
        AddIOFunctions.cpp
        FoldExpression.cpp
        FuseOperator.cpp
        InsertIncludes.cpp
        LoopFusion.cpp
        MergeCastChain.cpp
        PragmaDSE.cpp
        PragmaInsertion.cpp
        ResolveHelpers.cpp

    ENABLE_AGGREGATION

    DEPENDS
        EmitHLSPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        Gurobi::cxx
)
