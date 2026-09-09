/// Convenience include for the EmitHLSToLaksaApp translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a C application that runs the design
/// on a Kria board through the `laksa-hls-kria-driver`'s `/dev/laksa`. The
/// application allocates one DMA buffer per top-level `m_axi` argument, fills
/// every argument the kernel reads from `input<n>.bin`, launches the kernel,
/// and dumps every argument the kernel writes to `output<n>.bin`.
///
/// Buffer sizes and register offsets are not emitted here; the application
/// takes them from the header produced by the EmitHLSToLaksaHeader
/// translation, which has to be generated from the same input.
LogicalResult translateEmitHLSToLaksaApp(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
