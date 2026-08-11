add_mlir_translation_library(MLIRTargetEmitHLSToVivadoTcl
        TranslationRegistration.cpp
        EmitHLSToVivadoTcl.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
