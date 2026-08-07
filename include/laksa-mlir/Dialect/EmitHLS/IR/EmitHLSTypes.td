//===- EmitHLSTypes.td -------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_TYPES
#define EMITHLS_TYPES

include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSAttributes.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Scalar types
//===----------------------------------------------------------------------===//

def EmitHLS_StreamType : EmitHLS_Type<"Stream", []>
{
    let mnemonic = "stream";
    let summary = "Stream type in HLS C++";
    let description = [{
        The `stream` type represents a HLS C++ hls::stream type, which
        constructs an FIFO stream with element type flowing in it,
        e.g. `!emithls.stream<i32>`.
    }];

    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` qualified($elementType) `>`";
    let genVerifyDecl = 1;
}

def EmitHLS_PointerType : EmitHLS_Type<"Pointer", []>
{
    let mnemonic = "ptr";
    let summary = "Pointer type in HLS C++";
    let description = [{
        The `pointer` type represents a C++ pointer type, which points
        to the address of a value, e.g. `!emithls.ptr<i32>`.
    }];

    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` qualified($elementType) `>`";
    let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Shaped types
//===----------------------------------------------------------------------===//

def EmitHLS_ArrayType : EmitHLS_Type<"Array", [ShapedTypeInterface]>
{
    let mnemonic = "array";
    let summary = "Array type in C++";
    let description = [{
        The `array` type represents a C++ style array, which contains
        a shaped amount of variables of one scalar type, such as:
        `!emithls.array<2xi32>`, `!emithls.array<2x2xi32>`
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType
    );

    let builders = [
        TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                            "Type":$elementType)>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // ShapedTypeInterface Methods
        //===------------------------------------------------------------------===//

        // For HLS, an array should always have rank
        bool hasRank() const;

        ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                              Type elementType) const;
    }];
}

#endif
