//===- EdgeInterface.td - DFG edge interface -------------*- tablegen -*-===//
//
// This is the definitions file for the DFG edge interface.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_EDGEINTERFACE
#define DFG_EDGEINTERFACE

include "mlir/IR/OpBase.td"

def DFG_EdgeInterface : OpInterface<"EdgeInterface"> {
    let cppNamespace = "::mlir::dfg";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the node the input port connects to.}],
            /*retTy=*/"::mlir::Operation*",
            /*methodName=*/"getInputConnectedOp",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the node the output port connects to.}],
            /*retTy=*/"::mlir::Operation*",
            /*methodName=*/"getOutputConnectedOp",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >
    ];
}

#endif // DFG_EDGEINTERFACE
