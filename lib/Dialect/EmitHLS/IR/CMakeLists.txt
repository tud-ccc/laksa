add_mlir_dialect_library(EmitHLSIR
        EmitHLSAttributes.cpp
        EmitHLSBase.cpp
        EmitHLSOps.cpp
        EmitHLSTypes.cpp

    DEPENDS
        EmitHLSIncGen

    LINK_LIBS PUBLIC
        MLIRAffineDialect
        MLIRCallInterfaces
        MLIRFunctionInterfaces
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRControlFlowInterfaces
        EmitHLSEnums
        EmitHLSInterfaces
)
