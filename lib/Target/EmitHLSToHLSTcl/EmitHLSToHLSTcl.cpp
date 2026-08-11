// Implementation of translation EmitHLSToHLSTcl
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToHLSTcl/HLSTclEmitter.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::emithls;

namespace {

llvm::cl::opt<std::string> clHLSTargetDevice(
    "hls-target-device",
    llvm::cl::desc(
        "Target part passed to Vitis HLS's set_part in the generated "
        "run_hls.tcl"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

llvm::cl::opt<std::string> clHLSSourceFile(
    "hls-source-file",
    llvm::cl::desc("HLS C++ source file added to the generated run_hls.tcl"),
    llvm::cl::init("main.cpp"));

} // namespace

// Finds the function marked as the design's top function, i.e. the one whose
// body contains a PragmaTopInterfaceOp. Its symbol name doubles as the HLS
// solution name and the `set_top` target.
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

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult emithls::translateEmitHLSToHLSTcl(ModuleOp op, raw_ostream &os)
{
    FailureOr<FuncOp> topFunc = findTopFunc(op);
    if (failed(topFunc)) return failure();
    StringRef projectName = topFunc->getSymName();

    os << "set TARGET_DEVICE \"" << clHLSTargetDevice << "\"\n"
       << "set CLOCK_PERIOD 10\n"
       << "set PROJECT_DIR \"./hls_project\"\n"
       << "set SOURCE_FILE \"" << clHLSSourceFile << "\"\n"
       << "if {![file exists $SOURCE_FILE]} {\n"
       << "    puts \"ERROR: Source file $SOURCE_FILE does not exist!\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: Using source file: $SOURCE_FILE\"\n"
       << "open_project $PROJECT_DIR\n"
       << "add_files $SOURCE_FILE\n"
       << "puts \"INFO: Creating HLS solution\"\n"
       << "open_solution \"solution_" << projectName << "\"\n"
       << "set_part $TARGET_DEVICE\n"
       << "create_clock -period $CLOCK_PERIOD -name default\n"
       << "set_top " << projectName << "\n"
       << "puts \"INFO: Set top function to '" << projectName << "'\"\n"
       << "puts \"INFO: Running C synthesis...\"\n"
       << "if {[catch {csynth_design} result]} {\n"
       << "    puts \"ERROR: C synthesis failed: $result\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: C synthesis completed successfully\"\n"
       << "puts \"INFO: Exporting RTL design...\"\n"
       << "if {[catch {export_design -rtl verilog} result]} {\n"
       << "    puts \"ERROR: Failed to export design: $result\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: Design exported successfully\"\n"
       << "close_solution\n"
       << "close_project\n"
       << "puts \"INFO: HLS completed successfully\"\n"
       << "exit\n";

    return success();
}
