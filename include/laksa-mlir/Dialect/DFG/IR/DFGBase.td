//===- DFGBase.td - DFG dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_BASE
#define DFG_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DFG_Dialect : Dialect {
    let name = "dfg";
    let cppNamespace = "::mlir::dfg";

    let description = [{
        The `dfg` dialect groups together a set of types, operations and
        transformations that are useful to implement a structured abstraction
        on dataflow graphs.
        These abstractions are useful for constructing KPN/SDF-like dataflow
        graphs, which include definitions of nodes, edges and their connections.
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
class DFG_Op<string mnemonic, list<Trait> traits = []>
        : Op<DFG_Dialect, mnemonic, traits>;
        
// Template for types.
class DFG_Type<string name, list<Trait> traits = []>
        : TypeDef<DFG_Dialect, name, traits>;

#endif
