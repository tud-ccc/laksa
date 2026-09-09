add_mlir_translation_library(MLIRTargetEmitHLSToHLSBuildScript
        TranslationRegistration.cpp
        EmitHLSToHLSBuildScript.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
        MLIRTargetLaksaCommon
)
