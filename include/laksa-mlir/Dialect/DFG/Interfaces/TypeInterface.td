//===- TypeInterface.td ---------------------------------*- tablegen -*-===//
//
// This is the definitions file for the TypeInterface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_TYPEINTERFACE
#define DFG_TYPEINTERFACE

include "mlir/IR/OpBase.td"

def DFG_TypeInterface : TypeInterface<"TypeInterface"> {
    let cppNamespace = "::mlir::dfg";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Gets the base data type of the channel.
                For scalar channels (no shape), returns the element type directly.
                For shaped channels, returns a MemRefType with the channel's shape
                and element type.
            }],
            /*retTy=*/"::mlir::Type",
            /*methodName=*/"getBaseType",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto shape = $_type.getShape();
                if (shape.empty())
                    return $_type.getElementType();
                return ::mlir::MemRefType::get(shape, $_type.getElementType());
            }]
        >
    ];
}

#endif // DFG_TYPEINTERFACE
