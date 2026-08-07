/// BufferizableOpInterface include
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

namespace mlir {

class DialectRegistry;

namespace dfg {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace dfg
} // namespace mlir
