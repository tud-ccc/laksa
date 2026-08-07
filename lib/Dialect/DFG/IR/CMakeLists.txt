add_mlir_dialect_library(DFGIR
        DFGAttributes.cpp
        DFGBase.cpp
        DFGOps.cpp
        DFGTypes.cpp

    DEPENDS
        DFGIncGen

    LINK_LIBS PUBLIC
        MLIRCallInterfaces
        MLIRFunctionInterfaces
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRControlFlowInterfaces
        DFGEnums
        DFGInterfaces
)
