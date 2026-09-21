add_mlir_translation_library(MLIRTargetDFGToDot
        TranslateRegistration.cpp
        DFGToDot.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRRegisterAllDialects
        MLIRSupport
        MLIRTranslateLib
        DFGIR
)
