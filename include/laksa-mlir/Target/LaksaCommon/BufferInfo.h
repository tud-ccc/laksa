/// Top function lookup and buffer analysis shared by the translations that
/// target the `laksa-hls-kria-driver`.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::emithls {

/// Facts about one top-level `m_axi` argument, gathered by following it
/// through every `array.ptr_read`/`array.ptr_write` that reads or writes
/// through it -- directly, or through `emithls.call` operands forwarded into
/// dataflow node functions.
struct LaksaBufferInfo {
    /// Whether the host has to fill the buffer before launching the kernel.
    bool isReadByKernel = false;
    /// Whether the host may read the buffer back after the kernel finished.
    bool isWrittenByKernel = false;
    int64_t numElements = 0;
    unsigned elementBits = 0;

    /// Obtains the size of the DMA buffer the host has to allocate.
    int64_t getSizeInBytes() const
    { return numElements * ((elementBits + 7) / 8); }
};

/// Finds the function marked as the design's top function, i.e. the one whose
/// body contains a PragmaTopInterfaceOp. Its symbol name doubles as the
/// design's name in the generated artifacts.
FailureOr<FuncOp> findLaksaTopFunc(ModuleOp moduleOp);

/// Analyzes every argument of @p topFunc into one LaksaBufferInfo, in
/// argument order. Fails if an argument's size cannot be determined.
LogicalResult collectLaksaBufferInfos(
    FuncOp topFunc,
    SmallVectorImpl<LaksaBufferInfo> &buffers);

/// Obtains the macro prefix that both the generated header and the generated
/// application derive from @p designName .
std::string getLaksaMacroPrefix(StringRef designName);

} // namespace mlir::emithls
