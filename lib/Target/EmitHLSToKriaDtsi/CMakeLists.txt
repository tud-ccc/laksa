add_mlir_translation_library(MLIRTargetEmitHLSToKriaDtsi
        TranslationRegistration.cpp
        EmitHLSToKriaDtsi.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
