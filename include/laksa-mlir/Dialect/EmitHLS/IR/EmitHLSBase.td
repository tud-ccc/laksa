//===- EmitHLSBase.td - EmitHLS dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_BASE
#define EMITHLS_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EmitHLS_Dialect : Dialect {
    let name = "emithls";
    let cppNamespace = "::mlir::emithls";

    let description = [{
        The `emithls` dialect groups together a set of types, operations and
        transformations that are useful to represent a High-Level Synthese
        style C++ code.
        These abstractions works as a.
    }];

    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
        void registerAttributes();
    }];
}

// Template for ops.
class EmitHLS_Op<string mnemonic, list<Trait> traits = []>
        : Op<EmitHLS_Dialect, mnemonic, traits>;
        
// Template for types.
class EmitHLS_Type<string name, list<Trait> traits = []>
        : TypeDef<EmitHLS_Dialect, name, traits>;

#endif
