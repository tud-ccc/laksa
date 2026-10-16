add_mlir_translation_library(MLIRTargetEmitHLSToDot
        TranslationRegistration.cpp
        EmitHLSToDot.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
        MLIRTargetDotCommon
        MLIRTargetLaksaCommon
)
