add_mlir_translation_library(MLIRTargetEmitHLSToFPGAModelProfile
        TranslationRegistration.cpp
        EmitHLSToFPGAModelProfile.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
