//===- DFGTypes.td ------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_TYPES
#define DFG_TYPES

include "laksa-mlir/Dialect/DFG/IR/DFGAttributes.td"
include "laksa-mlir/Dialect/DFG/Interfaces/TypeInterface.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def DFG_InputType : DFG_Type<"Input", [DFG_TypeInterface]>
{
    let mnemonic = "input";
    let summary = "Sending channel end type";
    let description = [{
        The `input` type represents the sending end of a dataflow edge
        in a dataflow graph.
        This type takes an optional shape and a scalar MLIR type as
        element type, such as:
        `!dfg.input<i32>`, `!dfg.input<2xi32>`, `!dfg.input<2x2xi32>`
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType
    );

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                            "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "ShapedType":$shapedType)>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def DFG_OutputType : DFG_Type<"Output", [DFG_TypeInterface]>
{
    let mnemonic = "output";
    let summary = "Receiving channel end type";
    let description = [{
        The `output` type represents the receiving end of a dataflow edge
        in a dataflow graph.
        Details are similar to dfg::InputType.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType
    );

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                            "Type":$elementType)>,
        TypeBuilderWithInferredContext<(ins "ShapedType":$shapedType)>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif
