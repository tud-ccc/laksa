// Implementation of translation EmitHLSToKriaDtsi
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToKriaDtsi/KriaDtsiEmitter.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::emithls;

namespace {

// 0xa0000000 is the base address Vivado's address editor assigns by default
// to the first custom IP's AXI-Lite control interface (s_axi_control) when
// it is auto-connected to the Zynq MPSoC's M_AXI_HPM0_FPD master port, as
// done by the EmitHLSToVivadoTcl translation.
llvm::cl::opt<std::string> clKriaAxiLiteBaseAddress(
    "kria-axi-lite-base-address",
    llvm::cl::desc(
        "Physical base address of the HLS IP's AXI-Lite control interface, "
        "as assigned by Vivado's address editor"),
    llvm::cl::init("0xa0000000"));

llvm::cl::opt<std::string> clKriaAxiLiteSize(
    "kria-axi-lite-size",
    llvm::cl::desc(
        "Size of the AXI-Lite control register window, as assigned by "
        "Vivado's address editor"),
    llvm::cl::init("0x10000"));

} // namespace

// Finds the function marked as the design's top function, i.e. the one whose
// body contains a PragmaTopInterfaceOp. Its symbol name doubles as the
// device tree node's label and unit name.
static FailureOr<FuncOp> findTopFunc(ModuleOp moduleOp)
{
    FuncOp topFunc = nullptr;
    auto result = moduleOp.walk([&](PragmaTopInterfaceOp topInterfaceOp) {
        FuncOp funcOp = topInterfaceOp->getParentOfType<FuncOp>();
        if (!funcOp || topFunc) return WalkResult::interrupt();
        topFunc = funcOp;
        return WalkResult::advance();
    });
    if (result.wasInterrupted() || !topFunc)
        return moduleOp->emitError(
            "expected exactly one function containing a top_interface "
            "pragma");
    return topFunc;
}

// Strips a leading "0x"/"0X", if present, so the address can be reused as a
// device tree unit-address suffix (`node@<address>`).
static StringRef stripHexPrefix(StringRef hex)
{
    if (hex.starts_with_insensitive("0x")) return hex.drop_front(2);
    return hex;
}

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult emithls::translateEmitHLSToKriaDtsi(ModuleOp op, raw_ostream &os)
{
    FailureOr<FuncOp> topFunc = findTopFunc(op);
    if (failed(topFunc)) return failure();
    StringRef nodeName = topFunc->getSymName();
    StringRef unitAddress = stripHexPrefix(clKriaAxiLiteBaseAddress);

    os << "/dts-v1/;\n"
       << "/plugin/;\n"
          "\n"
          "/ {\n"
          "    fragment@0 {\n"
          "        target-path = \"/\";\n"
          "        __overlay__ {\n"
          "            #address-cells = <2>;\n"
          "            #size-cells = <2>;\n"
          "\n"
       << "            " << nodeName << ": " << nodeName << "@" << unitAddress
       << " {\n"
          "                compatible = \"tud,laksa\";\n"
          "                reg = <0x0 "
       << clKriaAxiLiteBaseAddress << " 0x0 " << clKriaAxiLiteSize
       << ">;\n"
          "            };\n"
          "        };\n"
          "    };\n"
          "};\n";

    return success();
}
