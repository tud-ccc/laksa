add_mlir_dialect_library(LinalgExtTransforms
        MapToGeneric.cpp
        SoftTranspose.cpp
        ScalarizeSplatDense.cpp

    ENABLE_AGGREGATION

    DEPENDS
        LinalgExtPassesIncGen

    LINK_LIBS PUBLIC
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRFuncDialect
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
