add_mlir_translation_library(MLIRTargetEmitHLSToHLSTcl
        TranslationRegistration.cpp
        EmitHLSToHLSTcl.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)
