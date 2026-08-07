//===- EmitHLS.cpp - C Interface for EmitHLS dialect ----------------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/EmitHLS.h"

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"
#include "laksa-mlir/Target/EmitHLSToCpp/HLSCppEmitter.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::emithls;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(EmitHLS, emithls, EmitHLSDialect)

//===---------------------------------------------------------------------===//
// StreamType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitHLSStreamType(MlirType type)
{ return isa<StreamType>(unwrap(type)); }

MlirType mlirEmitHLSStreamTypeGet(MlirType elementType)
{
    auto unwrapped = unwrap(elementType);
    return wrap(StreamType::get(unwrapped.getContext(), unwrapped));
}

MlirType mlirEmitHLSStreamTypeGetElementType(MlirType type)
{ return wrap(cast<StreamType>(unwrap(type)).getElementType()); }

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitHLSPointerType(MlirType type)
{ return isa<PointerType>(unwrap(type)); }

MlirType mlirEmitHLSPointerTypeGet(MlirType elementType)
{
    auto unwrapped = unwrap(elementType);
    return wrap(PointerType::get(unwrapped.getContext(), unwrapped));
}

MlirType mlirEmitHLSPointerTypeGetElementType(MlirType type)
{ return wrap(cast<PointerType>(unwrap(type)).getElementType()); }

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitHLSArrayType(MlirType type)
{ return isa<ArrayType>(unwrap(type)); }

MlirType mlirEmitHLSArrayTypeGet(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType)
{
    return wrap(
        ArrayType::get(ArrayRef<int64_t>(shape, rank), unwrap(elementType)));
}

intptr_t mlirEmitHLSArrayTypeGetRank(MlirType type)
{
    return static_cast<intptr_t>(
        cast<ArrayType>(unwrap(type)).getShape().size());
}

int64_t mlirEmitHLSArrayTypeGetDimSize(MlirType type, intptr_t dim)
{ return cast<ArrayType>(unwrap(type)).getShape()[dim]; }

MlirType mlirEmitHLSArrayTypeGetElementType(MlirType type)
{ return wrap(cast<ArrayType>(unwrap(type)).getElementType()); }

//===---------------------------------------------------------------------===//
// Translation
//===---------------------------------------------------------------------===//

MlirLogicalResult mlirTranslateEmitHLSToCpp(
    MlirOperation op,
    MlirStringCallback callback,
    void* userData)
{
    std::string buf;
    llvm::raw_string_ostream os(buf);
    if (failed(translateEmitHLSToCpp(unwrap(op), os)))
        return mlirLogicalResultFailure();
    callback(mlirStringRefCreate(buf.data(), buf.size()), userData);
    return mlirLogicalResultSuccess();
}
