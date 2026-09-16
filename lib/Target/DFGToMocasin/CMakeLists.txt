add_mlir_translation_library(MLIRTargetDFGToMocasin
        TranslateRegistration.cpp
        DFGToMocasin.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRRegisterAllDialects
        MLIRSupport
        MLIRTranslateLib
        DFGIR
)
