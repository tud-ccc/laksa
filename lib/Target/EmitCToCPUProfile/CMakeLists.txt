add_mlir_translation_library(MLIRTargetEmitCToCPUProfile
        TranslationRegistration.cpp
        EmitCToCPUProfile.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTargetCpp
        MLIRTranslateLib
        MLIREmitCDialect
)
