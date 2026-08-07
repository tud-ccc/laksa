/// Register all dialects in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/Transforms/BufferizableOpInterfaceImpl.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir {

inline void registerAllLAKSAMLIRDialects(DialectRegistry &registry)
{
    // DFG
    registry.insert<dfg::DFGDialect>();
    dfg::registerBufferizableOpInterfaceExternalModels(registry);
    // EmitHLS
    registry.insert<emithls::EmitHLSDialect>();
}

inline void registerAllLAKSAMLIRDialects(MLIRContext &context)
{
    DialectRegistry registry;
    registerAllLAKSAMLIRDialects(registry);
    context.appendDialectRegistry(registry);
}

} // namespace mlir
