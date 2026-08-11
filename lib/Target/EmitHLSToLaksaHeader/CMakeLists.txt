add_mlir_translation_library(MLIRTargetEmitHLSToLaksaHeader
        TranslationRegistration.cpp
        EmitHLSToLaksaHeader.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        EmitHLSIR
)