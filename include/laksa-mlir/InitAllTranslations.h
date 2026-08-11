/// Register all translations in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

namespace mlir {

void registerDFGToDotTranslation();
void registerEmitHLSToCppTranslation();
void registerEmitHLSToHLSTclTranslation();
void registerEmitHLSToVivadoTclTranslation();

inline void registerAllLAKSAMLIRTranslations()
{
    static bool initOnce = []() {
        registerDFGToDotTranslation();
        registerEmitHLSToCppTranslation();
        registerEmitHLSToHLSTclTranslation();
        registerEmitHLSToVivadoTclTranslation();
        return true;
    }();
    (void)initOnce;
}

} // namespace mlir
