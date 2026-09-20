add_mlir_translation_library(MLIRTargetEmitHLSToHLSProfileScript
        TranslationRegistration.cpp
        EmitHLSToHLSProfileScript.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
