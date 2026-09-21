add_mlir_translation_library(MLIRTargetEmitCToCPUProfileRunScript
        TranslationRegistration.cpp
        EmitCToCPUProfileRunScript.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        MLIREmitCDialect
)
