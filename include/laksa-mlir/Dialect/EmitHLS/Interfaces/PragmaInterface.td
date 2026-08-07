
//===- PragmaInterface.td - EmitHLS pragma interface --*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS pragma interface.
//
//===------------------------------------------------------------------------===//

#ifndef EMITHLS_PRAGMAINTERFACE
#define EMITHLS_PRAGMAINTERFACE

include "mlir/IR/OpBase.td"

def EmitHLS_PragmaInterface : OpInterface<"PragmaInterface"> {
    let cppNamespace = "::mlir::emithls";

    let methods = [];
}

#endif // EMITHLS_PRAGMAINTERFACE
