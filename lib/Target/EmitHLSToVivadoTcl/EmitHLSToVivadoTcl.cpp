// Implementation of translation EmitHLSToVivadoTcl
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToVivadoTcl/VivadoTclEmitter.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::emithls;

namespace {

llvm::cl::opt<std::string> clVivadoTargetDevice(
    "vivado-target-device",
    llvm::cl::desc(
        "Target part passed to Vivado's create_project in the generated "
        "run_vivado.tcl"),
    llvm::cl::init("xck26-sfvc784-2LV-c"));

} // namespace

// Finds the function marked as the design's top function, i.e. the one whose
// body contains a PragmaTopInterfaceOp. Its symbol name doubles as the
// Vivado project name and the HLS IP's VLNV/cell name.
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

LogicalResult emithls::translateEmitHLSToVivadoTcl(ModuleOp op, raw_ostream &os)
{
    FailureOr<FuncOp> topFunc = findTopFunc(op);
    if (failed(topFunc)) return failure();
    StringRef projectName = topFunc->getSymName();
    unsigned numArgs = topFunc->getNumArguments();

    os << "set project_name \"" << projectName << "\"\n"
       << "set project_dir \"./vivado_project/\"\n"
       << "set project_path \"$project_dir/$project_name.xpr\"\n"
       << "set target_device \"" << clVivadoTargetDevice << "\"\n"
       << "set ip_repo_dir \"./hls_project\"\n"
       << "if {[file exists $project_dir]} {\n"
       << "  puts \"INFO: Project directory exists.\"\n"
       << "  exit 1\n"
       << "}\n"
       << "puts \"INFO: Creating a new project...\"\n"
       << "create_project $project_name $project_dir -part $target_device\n"
       << "set_property part $target_device [current_project]\n"
       << "set_property default_lib xil_defaultlib [current_project]\n"
       << "set_property target_language Verilog [current_project]\n"
       << "if {[file exists $ip_repo_dir]} {\n"
       << "  puts \"INFO: Adding IP repository: $ip_repo_dir\"\n"
       << "  set_property ip_repo_paths $ip_repo_dir [current_project]\n"
       << "} else {\n"
       << "  puts \"WARNING: IP repository directory $ip_repo_dir does not "
          "exist!\"\n"
       << "  exit 1\n"
       << "}\n"
       << "set bd_name \"${project_name}_bd\"\n"
       << "puts \"INFO: Creating block design: $bd_name\"\n"
       << "create_bd_design $bd_name\n"
       << "puts \"INFO: Adding Zynq MPSoC to the block design\"\n"
       << "set zynq_mpsoc [create_bd_cell -type ip -vlnv "
          "xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_mpsoc]\n"
       << "puts \"INFO: Applying board preset to Zynq MPSoC\"\n"
       << "apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e "
          "-config {apply_board_preset \"1\"} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__FPGA_PL0_ENABLE {1} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__USE__M_AXI_GP0 {1} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__USE__M_AXI_GP1 {0} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__USE__M_AXI_GP2 {0} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__USE__S_AXI_GP0 {1} $zynq_mpsoc\n"
       << "set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} "
          "$zynq_mpsoc\n"
       << "puts \"INFO: Adding HLS IP kernel to the block design\"\n"
       << "create_bd_cell -type ip -vlnv xilinx.com:hls:" << projectName
       << ":1.0 " << projectName << "\n"
       << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { "
          "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master "
          "{/zynq_mpsoc/M_AXI_HPM0_FPD} Slave {/"
       << projectName
       << "/s_axi_control} ddr_seg {Auto} intc_ip {New AXI SmartConnect} "
          "master_apm {0}}  [get_bd_intf_pins "
       << projectName << "/s_axi_control]\n";

    // Each top-function argument becomes an m_axi_gmem_argN master port on the
    // HLS IP, which needs to be wired to the Zynq's HPC0 slave port. Automating
    // the first one enables Zynq's S_AXI_HPC0_FPD slave port and creates a new
    // AXI SmartConnect. Every subsequent one reuses that same SmartConnect,
    // automating the HLS IP's own port instead since the Zynq side is already
    // wired up.
    for (unsigned i = 0; i < numArgs; ++i) {
        os << "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { "
              "Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/"
           << projectName << "/m_axi_gmem_arg" << i
           << "} Slave {/zynq_mpsoc/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {";
        if (i == 0)
            os << "New AXI SmartConnect";
        else
            os << "/axi_smc_1";
        os << "} master_apm {0}} [get_bd_intf_pins ";
        if (i == 0)
            os << "zynq_mpsoc/S_AXI_HPC0_FPD";
        else
            os << projectName << "/m_axi_gmem_arg" << i;
        os << "]\n";
    }

    os << "puts \"INFO: Validating block design: $bd_name\"\n"
       << "if {[catch {validate_bd_design} result]} {\n"
       << "    puts \"Block design validation failed: $result\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: Validating succeeded\"\n"
       << "regenerate_bd_layout\n"
       << "save_bd_design\n"
       << "puts \"INFO: Closing block design: $bd_name\"\n"
       << "make_wrapper -files [get_files "
          "\"$project_dir/$project_name.srcs/sources_1/bd/$bd_name/"
          "$bd_name.bd\"] -top\n"
       << "set wrapper_path "
          "\"$project_dir/$project_name.srcs/sources_1/bd/$bd_name/hdl/"
          "${bd_name}_wrapper.v\"\n"
       << "add_files $wrapper_path\n"
       << "set_property top \"${bd_name}_wrapper\" [get_filesets sources_1]\n"
       << "set max_cores [get_param general.maxThreads]\n"
       << "puts \"INFO: Using up to $max_cores threads for synthesis and "
          "implementation.\"\n"
       << "puts \"INFO: Running Synthesis...\"\n"
       << "launch_runs synth_1 -jobs $max_cores\n"
       << "wait_on_run synth_1\n"
       << "set synth_status [get_property STATUS [get_runs synth_1]]\n"
       << "if {![string match \"*Complete!*\" $synth_status]} {\n"
       << "    puts \"ERROR: Synthesis failed with status: $synth_status\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: Synthesis completed successfully\"\n"
       << "puts \"INFO: Running Implementation...\"\n"
       << "reset_run impl_1\n"
       << "launch_runs impl_1 -to_step write_bitstream -jobs $max_cores\n"
       << "wait_on_run impl_1\n"
       << "set impl_status [get_property STATUS [get_runs impl_1]]\n"
       << "if {![string match \"*Complete!*\" $impl_status]} {\n"
       << "    puts \"ERROR: Implementation failed with status: "
          "$impl_status\"\n"
       << "    exit 1\n"
       << "}\n"
       << "puts \"INFO: Implementation completed successfully\"\n"
       << "puts \"INFO: Exporting hardware with bitstream\"\n"
       << "write_hw_platform -fixed -include_bit -force -file "
          "$project_dir/${project_name}_bd.xsa\n"
       << "exit\n";

    return success();
}
