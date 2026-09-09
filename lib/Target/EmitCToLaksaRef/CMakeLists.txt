add_mlir_translation_library(MLIRTargetEmitCToLaksaRef
        TranslationRegistration.cpp
        EmitCToLaksaRef.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        MLIREmitCDialect
)
