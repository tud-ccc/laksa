//===- DFGAttributes.td - DFG dialect attributes --------------*- tablegen -*-===//
//
// This is the definitions file for the DFG dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_ATTRIBUTES
#define DFG_ATTRIBUTES

include "laksa-mlir/Dialect/DFG/IR/DFGBase.td"

class DFG_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<DFG_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
