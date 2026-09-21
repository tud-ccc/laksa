// Implementation of the EmitHLS-to-FPGA-model-profile translation.
//
// @file
// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToFPGAModelProfile/FPGAModelProfileEmitter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::emithls;

namespace {

constexpr StringLiteral kProcessorType = "K26_PL";

void emitQuoted(raw_ostream &os, StringRef value)
{
    os << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') os << '\\';
        os << c;
    }
    os << '"';
}

} // namespace

LogicalResult emithls::translateEmitHLSToFPGAModelProfile(
    ModuleOp module,
    raw_ostream &os)
{
    SmallVector<std::pair<StringRef, int64_t>> profiles;
    for (FuncOp func : module.getOps<FuncOp>()) {
        auto cycles = dyn_cast_or_null<IntegerAttr>(
            func->getDiscardableAttr(emithls::kModelCyclesAttrName));
        if (!cycles) continue;
        if (cycles.getInt() < 0)
            return func.emitError(
                "expected a non-negative FPGA model cycle count");
        profiles.emplace_back(func.getSymName(), cycles.getInt());
    }

    if (profiles.empty())
        return module.emitError(
            "expected at least one function with a '")
               << emithls::kModelCyclesAttrName << "' attribute";

    os << "metadata: {source: laksa-model}\n"
          "execution:\n"
          "  processes:\n"
          "    profiles:\n";
    for (auto [name, cycles] : profiles) {
        os << "      ";
        emitQuoted(os, name);
        os << ":\n        ";
        emitQuoted(os, kProcessorType);
        os << ": {cycles: " << cycles << "}\n";
    }
    return success();
}
