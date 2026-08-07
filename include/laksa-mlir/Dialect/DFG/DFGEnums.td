//===- DFGEnums.td -----------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the DFG enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef DFG_ENUMS
#define DFG_ENUMS

include "mlir/IR/EnumAttr.td"

def DFG_OffloadHardwareAttr : I32EnumAttr<
    "OffloadHardware",
    "The hardware back to be offloaded to",
    [
        I32EnumAttrCase<"EmitHLS", 0b00, "emithls">,
        I32EnumAttrCase<"MDC", 0b01, "mdc">,
        I32EnumAttrCase<"CGRA", 0b10, "cgra">
    ]>
{
    let cppNamespace = "::mlir::dfg";
}

def DFG_OffloadHardwareDefaultAttr
    : DefaultValuedAttr<DFG_OffloadHardwareAttr, "OffloadHardware::EmitHLS">;

#endif // DFG_ENUMS