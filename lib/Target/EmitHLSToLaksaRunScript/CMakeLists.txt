add_mlir_translation_library(MLIRTargetEmitHLSToLaksaRunScript
        TranslationRegistration.cpp
        EmitHLSToLaksaRunScript.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
        MLIRTargetLaksaCommon
)
