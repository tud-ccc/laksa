add_mlir_translation_library(MLIRTargetDFGToDot
        TranslateRegistration.cpp
        DFGToDot.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRTranslateLib
        DFGIR
)
