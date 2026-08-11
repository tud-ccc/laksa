/// Convenience include for the EmitHLSToLaksaHeader translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a C header declaring the AXI-Lite
/// register offsets and DMA buffer sizes of the design's top-level `m_axi`
/// arguments, derived from the amount of data each argument's read/write IO
/// function moves. The macros are meant to be used together with the
/// `laksa-hls-kria-driver`'s `laksa.h` when writing the userspace program
/// that drives the design through `/dev/laksa`.
LogicalResult translateEmitHLSToLaksaHeader(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
