add_mlir_translation_library(MLIRTargetEmitHLSToCpp
        TranslationRegistration.cpp
        EmitHLSToCpp.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
