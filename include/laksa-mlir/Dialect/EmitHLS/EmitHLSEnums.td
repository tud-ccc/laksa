//===- EmitHLSEnums.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the EmitHLS enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef EmitHLS_ENUMS
#define EmitHLS_ENUMS

include "mlir/IR/EnumAttr.td"

def EmitHLS_CmpPredicateAttr : I32EnumAttr<
    "CmpPredicate",
    "predicate for comparison operation",
    [
      I32EnumAttrCase<"eq", 0>,
      I32EnumAttrCase<"ne", 1>,
      I32EnumAttrCase<"lt", 2>,
      I32EnumAttrCase<"le", 3>,
      I32EnumAttrCase<"gt", 4>,
      I32EnumAttrCase<"ge", 5>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

def EmitHLS_FusedOperatorAttr : I32EnumAttr<
    "FusedOperator",
    "fused operator such as `+=`",
    [
      I32EnumAttrCase<"add", 0>,
      I32EnumAttrCase<"sub", 1>,
      I32EnumAttrCase<"mul", 2>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

def EmitHLS_ArrayPartitionTypeAttr : I32EnumAttr<
    "ArrayPartitionType",
    "#pragma HLS array_partition type=?",
    [
      I32EnumAttrCase<"cyclic", 0>,
      I32EnumAttrCase<"block", 1>,
      I32EnumAttrCase<"complete", 2>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

def EmitHLS_BindStorageTypeAttr : I32EnumAttr<
    "BindStorageType",
    "#pragma HLS bind_storage type=?",
    [
      I32EnumAttrCase<"fifo", 0>,
      I32EnumAttrCase<"ram_1p", 1>,
      I32EnumAttrCase<"ram_2p", 2>,
      I32EnumAttrCase<"rom_1p", 3>,
      I32EnumAttrCase<"rom_2p", 4>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

def EmitHLS_BindStorageImplAttr : I32EnumAttr<
    "BindStorageImpl",
    "#pragma HLS bind_storage impl=?",
    [
      I32EnumAttrCase<"bram", 0>,
      I32EnumAttrCase<"lutram", 1>,
      I32EnumAttrCase<"srl", 2>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

def EmitHLS_PipelineStyleAttr : I32EnumAttr<
    "PipelineStyle",
    "##pragma HLS HLS pipeline style=?",
    [
      I32EnumAttrCase<"stp", 0>,
      I32EnumAttrCase<"flp", 1>,
      I32EnumAttrCase<"frp", 2>,
    ]> {
  let cppNamespace = "::mlir::emithls";
}

#endif // EmitHLS_ENUMS