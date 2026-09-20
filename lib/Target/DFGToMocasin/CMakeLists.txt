add_mlir_translation_library(MLIRTargetDFGToMocasin
        TranslateRegistration.cpp
        DFGToMocasin.cpp
        MocasinMergeScript.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRRegisterAllDialects
        MLIRSupport
        MLIRTranslateLib
        DFGIR
)
