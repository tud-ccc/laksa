/// Register all translations in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

namespace mlir {

void registerDFGToDotTranslation();
void registerDFGToMocasinTranslation();
void registerEmitCToCPUProfileTranslation();
void registerEmitCToCPUProfileRunScriptTranslation();
void registerEmitCToLaksaRefTranslation();
void registerEmitHLSToCppTranslation();
void registerEmitHLSToHLSBuildScriptTranslation();
void registerEmitHLSToFPGAModelProfileTranslation();
void registerEmitHLSToHLSTclTranslation();
void registerEmitHLSToKriaDtsiTranslation();
void registerEmitHLSToLaksaAppTranslation();
void registerEmitHLSToLaksaHeaderTranslation();
void registerEmitHLSToLaksaRunScriptTranslation();
void registerEmitHLSToVivadoTclTranslation();

inline void registerAllLAKSAMLIRTranslations()
{
    static bool initOnce = []() {
        registerDFGToDotTranslation();
        registerDFGToMocasinTranslation();
        registerEmitCToCPUProfileTranslation();
        registerEmitCToCPUProfileRunScriptTranslation();
        registerEmitCToLaksaRefTranslation();
        registerEmitHLSToCppTranslation();
        registerEmitHLSToHLSBuildScriptTranslation();
        registerEmitHLSToFPGAModelProfileTranslation();
        registerEmitHLSToHLSTclTranslation();
        registerEmitHLSToKriaDtsiTranslation();
        registerEmitHLSToLaksaAppTranslation();
        registerEmitHLSToLaksaHeaderTranslation();
        registerEmitHLSToLaksaRunScriptTranslation();
        registerEmitHLSToVivadoTclTranslation();
        return true;
    }();
    (void)initOnce;
}

} // namespace mlir
