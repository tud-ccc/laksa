/// Grid layout shared by the Dot translations.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::laksa {

struct GridDotNode {
    std::string name;
    std::string label;
    std::string shape;
};

struct GridDotEdge {
    std::string src;
    std::string dst;
    std::string label;
};

/// Lays out a graph on a grid with straight edges and emits its Dot body
std::string emitGridDot(
    StringRef title,
    StringRef bendPrefix,
    ArrayRef<GridDotNode> nodes,
    ArrayRef<GridDotEdge> edges);

} // namespace mlir::laksa
