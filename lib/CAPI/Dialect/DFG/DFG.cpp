//===- DFG.cpp - C Interface for DFG dialect --------------------------===//
//
// @author  Jiahong Bi (jiahong.bi@tu-dresden.de)
//===----------------------------------------------------------------------===//

#include "laksa-mlir-c/Dialect/DFG.h"

#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.h"
#include "laksa-mlir/Dialect/DFG/Transforms/BufferizableOpInterfaceImpl.h"
#include "laksa-mlir/Target/DFGToDot/DotEmitter.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dfg;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DFG, dfg, DFGDialect)

//===---------------------------------------------------------------------===//
// InputType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAInputType(MlirType type)
{ return isa<InputType>(unwrap(type)); }

MlirType mlirInputTypeGet(MlirType elementType)
{ return wrap(InputType::get(unwrap(elementType))); }

MlirType mlirInputTypeGetWithShape(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType)
{
    return wrap(
        InputType::get(ArrayRef<int64_t>(shape, rank), unwrap(elementType)));
}

intptr_t mlirInputTypeGetRank(MlirType type)
{
    return static_cast<intptr_t>(
        llvm::cast<InputType>(unwrap(type)).getShape().size());
}

int64_t mlirInputTypeGetDimSize(MlirType type, intptr_t dim)
{ return llvm::cast<InputType>(unwrap(type)).getShape()[dim]; }

MlirType mlirInputTypeGetElementType(MlirType type)
{ return wrap(llvm::cast<InputType>(unwrap(type)).getElementType()); }

//===---------------------------------------------------------------------===//
// OutputType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAOutputType(MlirType type)
{ return isa<OutputType>(unwrap(type)); }

MlirType mlirOutputTypeGet(MlirType elementType)
{ return wrap(OutputType::get(unwrap(elementType))); }

MlirType mlirOutputTypeGetWithShape(
    intptr_t rank,
    const int64_t* shape,
    MlirType elementType)
{
    return wrap(
        OutputType::get(ArrayRef<int64_t>(shape, rank), unwrap(elementType)));
}

intptr_t mlirOutputTypeGetRank(MlirType type)
{
    return static_cast<intptr_t>(
        llvm::cast<OutputType>(unwrap(type)).getShape().size());
}

int64_t mlirOutputTypeGetDimSize(MlirType type, intptr_t dim)
{ return llvm::cast<OutputType>(unwrap(type)).getShape()[dim]; }

MlirType mlirOutputTypeGetElementType(MlirType type)
{ return wrap(llvm::cast<OutputType>(unwrap(type)).getElementType()); }

//===---------------------------------------------------------------------===//
// BufferizableOpInterface registration
//===---------------------------------------------------------------------===//

void mlirDFGRegisterBufferizableOpInterfaceExternalModels(MlirContext context)
{
    DialectRegistry registry;
    registerBufferizableOpInterfaceExternalModels(registry);
    unwrap(context)->appendDialectRegistry(registry);
}

//===---------------------------------------------------------------------===//
// Translation registration
//===---------------------------------------------------------------------===//

MlirLogicalResult mlirTranslateDFGToDot(
    MlirOperation op,
    MlirStringCallback callback,
    void* userData)
{
    std::string buf;
    llvm::raw_string_ostream os(buf);
    if (failed(dfg::translateDFGToDot(unwrap(op), os)))
        return mlirLogicalResultFailure();
    callback(mlirStringRefCreate(buf.data(), buf.size()), userData);
    return mlirLogicalResultSuccess();
}
