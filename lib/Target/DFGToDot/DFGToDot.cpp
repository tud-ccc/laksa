/// Implementation of translation DFGToDot
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "laksa-mlir/Target/DFGToDot/DotEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::dfg;

namespace {

constexpr const char* kDotNodeName =
    R"({{GraphName}}_{{GraphNumber}}_{{NodeName}}_{{NodeNumber}})";
constexpr const char* kDotNode =
    R"({{GraphName}}_{{GraphNumber}}_{{NodeName}}_{{NodeNumber}} [label="{{NodeName}}", shape=box];
)";
constexpr const char* kDotSubgraph =
    R"(subgraph cluster_{{GraphName}}_{{GraphNumber}} {
  label="{{GraphName}}";
{{Content}}
}
)";

class DotEmitter {
public:
    LogicalResult emitGraph(ModuleOp module, raw_ostream &os);

private:
    void getGraph(ModuleOp module);

    static std::string
    replaceAll(std::string str, const std::string &from, const std::string &to)
    {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return str;
    }

    static std::string indentContent(const std::string &content)
    {
        std::string indented;
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) indented += "  " + line + "\n";
        return indented;
    }

    static std::string removeBlankLines(const std::string &input)
    {
        std::stringstream ss(input);
        std::string line;
        std::string result;
        while (std::getline(ss, line)) {
            bool isBlank = std::all_of(line.begin(), line.end(), [](char c) {
                return std::isspace(c);
            });
            if (!isBlank) result += line + '\n';
        }
        return result;
    }

    std::map<std::string, std::string> dotNodeNameMap;
    std::map<std::string, std::string> dotNodeMap;
    std::map<std::string, unsigned> dotNodeNumMap;
    llvm::DenseMap<Operation*, std::string> dotNodeOpMap;
    std::map<std::string, std::string> dotGraphMap;
    std::map<std::string, unsigned> dotGraphNumMap;
    // Maps subgraph name → output port dot node names (data flows OUT of
    // subgraph)
    std::map<std::string, SmallVector<std::string>> dotSubgraphOutPortMap;
    // Maps subgraph name → input port dot node names (data flows INTO subgraph)
    std::map<std::string, SmallVector<std::string>> dotSubgraphInPortMap;
    llvm::DenseMap<Operation*, std::string> dotSubgraphNameMap;
    llvm::DenseMap<Value, std::string> graphPortMap;
    SmallVector<std::string> topGraphs;
};

} // namespace

void DotEmitter::getGraph(ModuleOp module)
{
    for (auto &opi : module.getBodyRegion().front()) {
        if (auto graphOp = dyn_cast<GraphInterface>(opi)) {
            auto graphOpAsNode = cast<NodeInterface>(opi);
            auto graphName = graphOp.getGraphName();
            auto isSubGraph = graphOp.isSubGraph();
            std::string graphStr(kDotSubgraph);
            graphStr = replaceAll(graphStr, "{{GraphName}}", graphName);
            std::string graphContentStr;

            // Input ports: data flows INTO the graph from outside.
            SmallVector<std::string> outputPortsStr;
            for (unsigned i = 0; i < graphOpAsNode.getNumInputPorts(); ++i) {
                std::string inPortStr = "in_" + std::to_string(i);
                std::string inGraphPortStr =
                    graphName + "_{{GraphNumber}}_" + inPortStr;
                std::string inPortNodeStr = inGraphPortStr + " [label=\""
                                            + inPortStr
                                            + "\", shape=ellipse];\n";
                auto inPort = graphOpAsNode.getInputPort(i);
                graphContentStr += indentContent(inPortNodeStr);
                graphPortMap.insert({inPort, inGraphPortStr});
                outputPortsStr.push_back(inGraphPortStr);
            }
            if (isSubGraph) {
                auto subGraphNameStr = graphName + "_{{GraphNumber}}";
                dotSubgraphInPortMap.insert({subGraphNameStr, outputPortsStr});
            }

            // Source rank grouping for input ports
            std::string graphSourceNodes = "{rank=source;{{SourceNodes}}}\n";
            std::string sourceNodesStr;
            for (auto inPort : graphOpAsNode.getInputPorts())
                sourceNodesStr += (" " + graphPortMap[inPort] + ";");
            graphContentStr += indentContent(replaceAll(
                graphSourceNodes,
                "{{SourceNodes}}",
                sourceNodesStr));

            // Output ports: data flows OUT of the graph to the parent.
            SmallVector<std::string> inputPortsStr;
            for (unsigned i = 0; i < graphOpAsNode.getNumOutputPorts(); ++i) {
                std::string outPortStr = "out_" + std::to_string(i);
                std::string outGraphPortStr =
                    graphName + "_{{GraphNumber}}_" + outPortStr;
                std::string outPortNodeStr = outGraphPortStr + " [label=\""
                                             + outPortStr
                                             + "\", shape=ellipse];\n";
                auto outPort = graphOpAsNode.getOutputPort(i);
                graphContentStr += indentContent(outPortNodeStr);
                graphPortMap.insert({outPort, outGraphPortStr});
                inputPortsStr.push_back(outGraphPortStr);
            }
            if (isSubGraph) {
                auto subGraphNameStr = graphName + "_{{GraphNumber}}";
                dotSubgraphOutPortMap.insert({subGraphNameStr, inputPortsStr});
            }

            // Sink rank grouping for output ports
            std::string graphSinkNodes = "{rank=sink;{{SinkNodes}}}\n";
            std::string sinkNodesStr;
            for (auto outPort : graphOpAsNode.getOutputPorts())
                sinkNodesStr += (" " + graphPortMap[outPort] + ";");
            graphContentStr += indentContent(
                replaceAll(graphSinkNodes, "{{SinkNodes}}", sinkNodesStr));

            // Internal nodes (instantiate ops)
            for (auto graphContent : graphOp.getGraphNodes()) {
                auto graphContentNode = cast<NodeInterface>(graphContent);
                auto graphContentNodeName = graphContentNode.getNodeName();
                auto graphContentNodeStr = dotNodeMap[graphContentNodeName];
                auto graphContentNodeNameStr =
                    dotNodeNameMap[graphContentNodeName];
                auto graphContentNodeNumber =
                    dotNodeNumMap[graphContentNodeNameStr]++;
                graphContentNodeNameStr = replaceAll(
                    graphContentNodeNameStr,
                    "{{GraphName}}",
                    graphName);
                graphContentNodeNameStr = replaceAll(
                    graphContentNodeNameStr,
                    "{{NodeNumber}}",
                    std::to_string(graphContentNodeNumber));
                graphContentNodeStr =
                    replaceAll(graphContentNodeStr, "{{GraphName}}", graphName);
                graphContentNodeStr = replaceAll(
                    graphContentNodeStr,
                    "{{NodeNumber}}",
                    std::to_string(graphContentNodeNumber));
                graphContentStr += indentContent(graphContentNodeStr);
                dotNodeOpMap.insert({graphContent, graphContentNodeNameStr});
            }

            // Embedded subgraphs
            for (auto graphContent : graphOp.getGraphSubGs()) {
                auto graphContentSubgraph = cast<GraphInterface>(graphContent);
                auto graphContentSubgraphName =
                    graphContentSubgraph.getGraphName();
                auto graphContentSubgraphNumber =
                    dotGraphNumMap[graphContentSubgraphName]++;
                std::string graphContentSubgraphStr =
                    dotGraphMap[graphContentSubgraphName];
                graphContentSubgraphStr = replaceAll(
                    graphContentSubgraphStr,
                    "{{GraphNumber}}",
                    std::to_string(graphContentSubgraphNumber));
                graphContentStr += graphContentSubgraphStr;
                dotSubgraphNameMap.insert(
                    {graphContent,
                     graphContentSubgraphName + "_"
                         + std::to_string(graphContentSubgraphNumber)});

                // Instantiate port maps for this embedded subgraph.
                // Iterate over a snapshot to safely insert new entries.
                for (auto [key, ports] : dotSubgraphInPortMap) {
                    if (key == graphContentSubgraphName + "_{{GraphNumber}}") {
                        auto newName =
                            graphContentSubgraphName + "_"
                            + std::to_string(graphContentSubgraphNumber);
                        SmallVector<std::string> newPorts;
                        for (auto &port : ports)
                            newPorts.push_back(replaceAll(
                                port,
                                "{{GraphNumber}}",
                                std::to_string(graphContentSubgraphNumber)));
                        dotSubgraphInPortMap.insert({newName, newPorts});
                    }
                }
                for (auto [key, ports] : dotSubgraphOutPortMap) {
                    if (key == graphContentSubgraphName + "_{{GraphNumber}}") {
                        auto newName =
                            graphContentSubgraphName + "_"
                            + std::to_string(graphContentSubgraphNumber);
                        SmallVector<std::string> newPorts;
                        for (auto &port : ports)
                            newPorts.push_back(replaceAll(
                                port,
                                "{{GraphNumber}}",
                                std::to_string(graphContentSubgraphNumber)));
                        dotSubgraphOutPortMap.insert({newName, newPorts});
                    }
                }
            }

            // Channel edges.
            // ChannelOp: input_port (result 0, DFGInputType) = write end,
            //            output_port (result 1, DFGOutputType) = read end.
            for (auto graphContent : graphOp.getGraphEdges()) {
                auto channelOp = cast<ChannelOp>(graphContent);
                std::string graphContentEdgeStr = "{{INPUT}} -> {{OUTPUT}};";

                // Producer: user of input_port (write end)
                auto inConnection = channelOp.getInputConnectedOp();
                Value inChanVal = channelOp.getInputPort();
                std::string edgeInputStr;
                if (isa<GraphInterface>(inConnection)) {
                    auto subGraphNameStr = dotSubgraphNameMap[inConnection];
                    auto subGraphOutPorts =
                        dotSubgraphOutPortMap[subGraphNameStr];
                    auto inConnectionAsNode =
                        dyn_cast<NodeInterface>(inConnection);
                    for (auto [outPort, outPortName] : llvm::zip(
                             inConnectionAsNode.getOutputPorts(),
                             subGraphOutPorts)) {
                        if (outPort == inChanVal) edgeInputStr = outPortName;
                    }
                } else {
                    edgeInputStr = dotNodeOpMap[inConnection];
                }
                graphContentEdgeStr =
                    replaceAll(graphContentEdgeStr, "{{INPUT}}", edgeInputStr);

                // Consumer: user of output_port (read end)
                auto outConnection = channelOp.getOutputConnectedOp();
                Value outChanVal = channelOp.getOutputPort();
                std::string edgeOutputStr;
                if (isa<GraphInterface>(outConnection)) {
                    auto subGraphNameStr = dotSubgraphNameMap[outConnection];
                    auto subGraphInPorts =
                        dotSubgraphInPortMap[subGraphNameStr];
                    auto outConnectionAsNode =
                        dyn_cast<NodeInterface>(outConnection);
                    for (auto [inPort, inPortName] : llvm::zip(
                             outConnectionAsNode.getInputPorts(),
                             subGraphInPorts)) {
                        if (inPort == outChanVal) edgeOutputStr = inPortName;
                    }
                } else {
                    edgeOutputStr = dotNodeOpMap[outConnection];
                }
                graphContentEdgeStr = replaceAll(
                    graphContentEdgeStr,
                    "{{OUTPUT}}",
                    edgeOutputStr);
                graphContentStr += indentContent(graphContentEdgeStr);
            }

            // Region port edges.
            // Draw edges for all graphs (not just subgraphs).
            for (auto inPort : graphOpAsNode.getInputPorts()) {
                if (inPort.use_empty()) continue;
                auto inputStr = graphPortMap[inPort];
                for (auto &use : inPort.getUses()) {
                    auto inPortUser = use.getOwner();
                    std::string outputStr;
                    if (isa<GraphInterface>(inPortUser)) {
                        auto subGraphNameStr = dotSubgraphNameMap[inPortUser];
                        auto subGraphInPorts =
                            dotSubgraphInPortMap[subGraphNameStr];
                        auto asNode = dyn_cast<NodeInterface>(inPortUser);
                        for (auto [port, portName] : llvm::zip(
                                 asNode.getInputPorts(),
                                 subGraphInPorts)) {
                            if (port == inPort) outputStr = portName;
                        }
                    } else if (isa<NodeInterface>(inPortUser)) {
                        outputStr = dotNodeOpMap[inPortUser];
                    }
                    if (!outputStr.empty())
                        graphContentStr +=
                            indentContent(inputStr + " -> " + outputStr + ";");
                }
            }
            for (auto outPort : graphOpAsNode.getOutputPorts()) {
                if (outPort.use_empty()) continue;
                auto outputStr = graphPortMap[outPort];
                for (auto &use : outPort.getUses()) {
                    auto outPortUser = use.getOwner();
                    std::string inputStr;
                    if (isa<GraphInterface>(outPortUser)) {
                        auto subGraphNameStr = dotSubgraphNameMap[outPortUser];
                        auto subGraphOutPorts =
                            dotSubgraphOutPortMap[subGraphNameStr];
                        auto asNode = dyn_cast<NodeInterface>(outPortUser);
                        for (auto [port, portName] : llvm::zip(
                                 asNode.getOutputPorts(),
                                 subGraphOutPorts)) {
                            if (port == outPort) inputStr = portName;
                        }
                    } else if (isa<NodeInterface>(outPortUser)) {
                        inputStr = dotNodeOpMap[outPortUser];
                    }
                    if (!inputStr.empty())
                        graphContentStr +=
                            indentContent(inputStr + " -> " + outputStr + ";");
                }
            }

            // Save the graph template
            graphStr = indentContent(
                replaceAll(graphStr, "{{Content}}", graphContentStr));
            dotGraphMap.insert({graphName, graphStr});
            if (opi.hasAttr(laksa::kRootAttrName)) {
                topGraphs.push_back(
                    replaceAll(graphStr, "{{GraphNumber}}", "0"));
            }

        } else if (auto nodeOp = dyn_cast<NodeInterface>(opi)) {
            // Module-level node definitions (ProcessOp, OperatorOp)
            auto nodeName = nodeOp.getNodeName();
            std::string nodeStr(kDotNode);
            nodeStr = replaceAll(nodeStr, "{{NodeName}}", nodeName);
            dotNodeMap.insert({nodeName, nodeStr});
            std::string nodeNameStr(kDotNodeName);
            nodeNameStr = replaceAll(nodeNameStr, "{{NodeName}}", nodeName);
            dotNodeNameMap.insert({nodeName, nodeNameStr});
        }
    }
}

LogicalResult DotEmitter::emitGraph(ModuleOp module, raw_ostream &os)
{
    getGraph(module);
    for (auto [i, graph] : llvm::enumerate(topGraphs)) {
        raw_indented_ostream ios(os);
        os << "digraph G_" << i << " {\n";
        ios.indent() << "rankdir=LR;\n";
        os << removeBlankLines(graph);
        ios.unindent() << "}\n";
    }
    return success();
}

LogicalResult dfg::translateDFGToDot(Operation* op, raw_ostream &os)
{
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) return op->emitError("expected builtin.module op");
    DotEmitter emitter;
    return emitter.emitGraph(module, os);
}
