//===- EmitHLSAttributes.td - EmitHLS dialect attributes ---*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef EMITHLS_ATTRIBUTES
#define EMITHLS_ATTRIBUTES

include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"

class EmitHLS_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EmitHLS_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
