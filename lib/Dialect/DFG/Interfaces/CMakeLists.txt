add_mlir_library(DFGInterfaces
    EdgeInterface.cpp
    GraphInterface.cpp
    NodeInterface.cpp
    TypeInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        DFGIncGen
)