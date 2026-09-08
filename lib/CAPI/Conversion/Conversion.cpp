//===- Conversion.cpp - C Interface for conversion passes -----------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Conversion.h"

#include "laksa-mlir/Conversion/ConvertToEmitC/ConvertToEmitC.h"
#include "laksa-mlir/Conversion/ConvertToEmitHLS/ConvertToEmitHLS.h"
#include "laksa-mlir/Conversion/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

// Must include the declarations as they carry important visibility attributes.
#include "laksa-mlir/Conversion/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::laksa;

void mlirRegisterLAKSAConvertToEmitCPipelines()
{ registerConvertToEmitCPipelines(); }

void mlirConversionAddConvertToEmitCPasses(
    MlirOpPassManager passManager,
    uint32_t maxAllocSizeInBytes)
{ addConvertToEmitCPasses(*unwrap(passManager), maxAllocSizeInBytes); }

void mlirRegisterLAKSAConvertToEmitHLSPipelines()
{ registerConvertToEmitHLSPipelines(); }

void mlirConversionAddConvertToEmitHLSPasses(
    MlirOpPassManager passManager,
    int64_t availableBRAM,
    int64_t availableDSP)
{
    addConvertToEmitHLSPasses(
        *unwrap(passManager),
        availableBRAM,
        availableDSP);
}

MlirLogicalResult mlirTranslateEmitCToCpp(
    MlirOperation op,
    MlirStringCallback callback,
    void* userData,
    bool declareVariablesAtTop,
    MlirStringRef fileId)
{
    std::string buf;
    llvm::raw_string_ostream os(buf);
    if (failed(
            emitc::translateToCpp(
                unwrap(op),
                os,
                declareVariablesAtTop,
                unwrap(fileId))))
        return mlirLogicalResultFailure();
    callback(mlirStringRefCreate(buf.data(), buf.size()), userData);
    return mlirLogicalResultSuccess();
}

#ifdef __cplusplus
extern "C" {
#endif

#include "laksa-mlir/Conversion/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
