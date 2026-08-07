add_mlir_dialect_library(DFGTransforms
        BufferizableOpInterfaceImpl.cpp
        OperatorToProcess.cpp
        InlineEmbedRegion.cpp
        IONormalization.cpp
        ChannelFanoutExpansion.cpp
        CollapseUnitDims.cpp

    ENABLE_AGGREGATION

    DEPENDS
        DFGPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRBufferizationDialect
        MLIRMemRefDialect
)
