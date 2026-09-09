add_mlir_translation_library(MLIRTargetEmitHLSToLaksaApp
        TranslationRegistration.cpp
        EmitHLSToLaksaApp.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
        MLIRTargetLaksaCommon
)
