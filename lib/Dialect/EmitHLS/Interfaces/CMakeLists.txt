add_mlir_library(EmitHLSInterfaces
    PragmaInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        EmitHLSIncGen
)