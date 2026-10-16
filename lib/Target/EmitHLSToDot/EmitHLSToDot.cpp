// Implementation of translation EmitHLSToDot
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/DotCommon/GridDot.h"
#include "laksa-mlir/Target/EmitHLSToDot/HLSDotEmitter.h"
#include "laksa-mlir/Target/LaksaCommon/BufferInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::emithls;

// Whether the callee writes its argument at idx.
static bool writesArgument(FuncOp callee, unsigned idx)
{
    return llvm::any_of(
        callee.getArgument(idx).getUsers(),
        [](Operation* user) {
            return isa<StreamWriteOp, ArrayWriteOp, ArrayPointerWriteOp>(user);
        });
}

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult emithls::translateEmitHLSToDot(ModuleOp op, raw_ostream &os)
{
    FailureOr<FuncOp> topFunc = findLaksaTopFunc(op);
    if (failed(topFunc)) return failure();
    StringRef topName = topFunc->getSymName();

    // Calls moving data from or to memory become the in/out ports.
    auto isTopArg = [&](Value value) {
        auto arg = dyn_cast<BlockArgument>(value);
        return arg && arg.getOwner()->getParentOp() == *topFunc;
    };
    SmallVector<laksa::GridDotNode> inputs, outputs, nodes;
    DenseMap<Operation*, std::string> nodeNames;
    llvm::StringMap<unsigned> numCalls;
    topFunc->walk([&](CallOp callOp) {
        auto callee = op.lookupSymbol<FuncOp>(callOp.getCalleeAttr());
        if (!callee) return;
        StringRef calleeName = callOp.getCallee();
        laksa::GridDotNode node;
        if (llvm::any_of(callOp.getArgOperands(), isTopArg)) {
            bool isInput = false;
            for (auto [idx, operand] : llvm::enumerate(callOp.getArgOperands()))
                if (!isTopArg(operand) && writesArgument(callee, idx))
                    isInput = true;
            auto &ports = isInput ? inputs : outputs;
            std::string label =
                (Twine(isInput ? "in_" : "out_") + Twine(ports.size())).str();
            node = {(topName + "_" + label).str(), label, "ellipse"};
            ports.push_back(node);
        } else {
            unsigned number = numCalls[calleeName]++;
            node = {
                (topName + "_" + calleeName + "_" + Twine(number)).str(),
                calleeName.str(),
                "box"};
            nodes.push_back(node);
        }
        nodeNames[callOp] = node.name;
    });

    // Streams are named as in the emitted C++, in declaration order.
    SmallVector<laksa::GridDotEdge> edges;
    unsigned numStreams = 0;
    for (auto varOp : topFunc->getBody().front().getOps<VariableOp>()) {
        Type type = varOp.getType();
        if (auto arrayType = dyn_cast<ArrayType>(type))
            type = arrayType.getElementType();
        if (varOp.getIsConst() || !isa<StreamType>(type)) continue;
        Value stream = varOp.getVariable();
        std::string label = "stream" + std::to_string(numStreams++);
        std::string producer;
        SmallVector<std::string> consumers;
        for (Operation* user : stream.getUsers()) {
            if (auto streamOp = dyn_cast<PragmaStreamOp>(user))
                label += "\\ndepth=" + std::to_string(streamOp.getDepth());
            auto callOp = dyn_cast<CallOp>(user);
            if (!callOp || !nodeNames.count(callOp)) continue;
            auto callee = op.lookupSymbol<FuncOp>(callOp.getCalleeAttr());
            for (auto [idx, operand] :
                 llvm::enumerate(callOp.getArgOperands())) {
                if (operand != stream) continue;
                if (writesArgument(callee, idx))
                    producer = nodeNames[callOp];
                else
                    consumers.push_back(nodeNames[callOp]);
            }
        }
        if (producer.empty()) continue;
        for (auto &consumer : consumers)
            edges.push_back({producer, consumer, label});
    }

    SmallVector<laksa::GridDotNode> allNodes(inputs);
    allNodes.append(outputs);
    allNodes.append(nodes);
    os << "digraph G_0 {\n"
       << laksa::emitGridDot(topName, topName, allNodes, edges) << "}\n";
    return success();
}
