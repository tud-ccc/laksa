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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

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

    // Lays the graph out on a grid, folding a long data flow into a snake.
    static std::string gridGraph(const std::string &graph);

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

namespace {

// Layout geometry in inches, after the Graphviz defaults.
constexpr double kNodeHeight = 0.5;
constexpr double kColumnGap = 0.6;
constexpr double kRowGap = 0.5;
constexpr double kCharWidth = 0.1;
constexpr double kNodeMargin = 0.3;
constexpr double kMinNodeWidth = 0.75;
// Shortest main path worth folding, and the fewest nodes per row.
constexpr unsigned kMinFoldNodes = 10;
constexpr unsigned kMinRowNodes = 3;
// Farthest lane a branch or a detouring edge may take.
constexpr int kMaxLane = 32;
// Route costs; a crossing outweighs any detour.
constexpr double kCrossingCost = 10;
constexpr double kBendCost = 1;
constexpr double kLengthCost = 0.1;
// Weight of crossings and unrouted edges against the aspect ratio.
constexpr double kPenaltyWeight = 2;
constexpr unsigned kUnroutedPenalty = 10;

// A grid cell; x grows rightwards and y downwards.
using Cell = std::pair<int, int>;

struct RouteCost {
    double cost;
    unsigned crossings;
};

// Grid occupancy by nodes and axis-aligned edge segments.
class Grid {
public:
    // Cost of a route that occupies the given nodes; none on an overlap.
    std::optional<RouteCost>
    cost(ArrayRef<Cell> pts, ArrayRef<Cell> nodes) const
    {
        std::map<Cell, char> used;
        for (Cell c : nodes)
            if (cells.count(c) || !used.try_emplace(c, 'n').second)
                return std::nullopt;
        unsigned crossings = 0, length = 0;
        for (auto [a, b] : llvm::zip(pts, llvm::drop_begin(pts))) {
            if (a == b || (a.first != b.first && a.second != b.second))
                return std::nullopt;
            char dir = a.first == b.first ? 'v' : 'h';
            length +=
                std::abs(b.first - a.first) + std::abs(b.second - a.second);
            for (Cell c : interior(a, b)) {
                char cur = 0;
                if (auto it = used.find(c); it != used.end())
                    cur = it->second;
                else if (auto it = cells.find(c); it != cells.end())
                    cur = it->second;
                if (!cur) {
                    used[c] = dir;
                } else if ((cur == 'h' || cur == 'v') && cur != dir) {
                    ++crossings;
                    used[c] = 'x';
                } else {
                    return std::nullopt;
                }
            }
        }
        double bends = static_cast<double>(pts.size()) - 2;
        return RouteCost{
            kCrossingCost * crossings + kBendCost * bends
                + kLengthCost * length,
            crossings};
    }

    void commit(ArrayRef<Cell> pts, ArrayRef<Cell> nodes)
    {
        for (Cell c : nodes) cells[c] = 'n';
        for (auto [a, b] : llvm::zip(pts, llvm::drop_begin(pts))) {
            char dir = a.first == b.first ? 'v' : 'h';
            for (Cell c : interior(a, b)) {
                auto [it, inserted] = cells.try_emplace(c, dir);
                if (!inserted) it->second = 'x';
            }
        }
    }

private:
    static SmallVector<Cell> interior(Cell a, Cell b)
    {
        int dx = (b.first > a.first) - (b.first < a.first);
        int dy = (b.second > a.second) - (b.second < a.second);
        SmallVector<Cell> result;
        for (Cell c{a.first + dx, a.second + dy}; c != b;
             c.first += dx, c.second += dy)
            result.push_back(c);
        return result;
    }

    std::map<Cell, char> cells; // 'n'ode, 'h'/'v' segment or 'x' crossing
};

// Node cells and edge polylines of a grid layout.
struct GridLayout {
    SmallVector<Cell> pos;
    // Endpoints included; empty for an edge drawn as a plain line.
    SmallVector<SmallVector<Cell>> routes;
    unsigned penalty = 0;
};

// Renumbers the used rows and columns densely.
void compactGrid(GridLayout &layout)
{
    std::set<int> xs, ys;
    for (Cell c : layout.pos) {
        xs.insert(c.first);
        ys.insert(c.second);
    }
    for (auto &route : layout.routes) {
        for (Cell c : route) {
            xs.insert(c.first);
            ys.insert(c.second);
        }
    }
    std::map<int, int> xIndex, yIndex;
    for (int x : xs) xIndex.try_emplace(x, xIndex.size());
    for (int y : ys) yIndex.try_emplace(y, yIndex.size());
    auto remap = [&](Cell &c) { c = {xIndex[c.first], yIndex[c.second]}; };
    for (Cell &c : layout.pos) remap(c);
    for (auto &route : layout.routes)
        for (Cell &c : route) remap(c);
}

// Merges neighbouring rows whose contents do not collide.
void mergeRows(GridLayout &layout)
{
    for (bool merged = true; merged;) {
        compactGrid(layout);
        merged = false;
        int numRows = 0;
        for (Cell c : layout.pos) numRows = std::max(numRows, c.second + 1);
        for (auto &route : layout.routes)
            for (Cell c : route) numRows = std::max(numRows, c.second + 1);
        for (int y1 = 0; y1 + 1 < numRows && !merged; ++y1) {
            int y2 = y1 + 1;
            // Owners of each cell in both rows: ~v for node v, e for edge e.
            std::map<Cell, std::set<int>> owners;
            bool clash = false;
            for (auto [v, c] : llvm::enumerate(layout.pos))
                if (c.second == y1 || c.second == y2)
                    owners[c].insert(~static_cast<int>(v));
            for (auto [e, route] : llvm::enumerate(layout.routes)) {
                for (auto [a, b] : llvm::zip(route, llvm::drop_begin(route))) {
                    if (a.first == b.first) {
                        int lo = std::min(a.second, b.second);
                        int hi = std::max(a.second, b.second);
                        // A segment between the two rows would vanish.
                        clash |= lo == y1 && hi == y2;
                        for (int y : {y1, y2})
                            if (lo <= y && y <= hi)
                                owners[{a.first, y}].insert(
                                    static_cast<int>(e));
                    } else if (a.second == y1 || a.second == y2) {
                        int lo = std::min(a.first, b.first);
                        int hi = std::max(a.first, b.first);
                        for (int x = lo; x <= hi; ++x)
                            owners[{x, a.second}].insert(static_cast<int>(e));
                    }
                }
            }
            // Only an edge passing straight through both rows may share a
            // column.
            for (auto &[cell, upper] : owners) {
                if (clash) break;
                if (cell.second != y1) continue;
                auto lower = owners.find({cell.first, y2});
                if (lower == owners.end()) continue;
                clash = upper != lower->second || upper.size() != 1
                        || *upper.begin() < 0;
            }
            if (clash) continue;
            auto lift = [&](Cell &c) {
                if (c.second == y2) c.second = y1;
            };
            for (Cell &c : layout.pos) lift(c);
            for (auto &route : layout.routes) {
                for (Cell &c : route) lift(c);
                route.erase(
                    std::unique(route.begin(), route.end()),
                    route.end());
            }
            merged = true;
        }
    }
}

// Column and row centres and the overall size of a layout, in inches.
struct GridGeometry {
    SmallVector<double> colX, rowY;
    double width = 0, height = 0;
};

GridGeometry measureGrid(const GridLayout &layout, ArrayRef<double> widths)
{
    int numCols = 0, numRows = 0;
    auto extend = [&](Cell c) {
        numCols = std::max(numCols, c.first + 1);
        numRows = std::max(numRows, c.second + 1);
    };
    for (Cell c : layout.pos) extend(c);
    for (auto &route : layout.routes)
        for (Cell c : route) extend(c);
    SmallVector<double> colWidth(numCols, 0.0);
    for (auto [c, w] : llvm::zip(layout.pos, widths))
        colWidth[c.first] = std::max(colWidth[c.first], w);
    GridGeometry geo;
    double x = 0;
    for (double w : colWidth) {
        geo.colX.push_back(x + w / 2);
        x += w + kColumnGap;
    }
    geo.width = x - kColumnGap;
    for (int r = 0; r < numRows; ++r)
        geo.rowY.push_back((numRows - 1 - r) * (kNodeHeight + kRowGap));
    geo.height = numRows * kNodeHeight + (numRows - 1) * kRowGap;
    return geo;
}

} // namespace

std::string DotEmitter::gridGraph(const std::string &graph)
{
    struct Edge {
        std::string src, dst, line;
    };

    // Recover the label, nodes and edges of the emitted graph.
    std::string label, rootName;
    SmallVector<std::string> nodes, nodeLines;
    SmallVector<double> widths;
    SmallVector<Edge> edges;
    unsigned level = 0;
    std::istringstream iss(graph);
    std::string line;
    while (std::getline(iss, line)) {
        StringRef s = StringRef(line).trim();
        if (s.starts_with("{rank=")) continue;
        if (s.consume_front("subgraph cluster_")) {
            if (level++ == 0) rootName = s.split(' ').first.str();
        } else if (s == "}") {
            if (level) --level;
        } else if (s.contains(" -> ")) {
            auto [src, dst] = s.rtrim(';').split(" -> ");
            edges.push_back({src.str(), dst.str(), s.str()});
        } else if (s.contains(" [")) {
            StringRef nodeLabel = s.split("label=\"").second.split('"').first;
            nodes.push_back(s.split(' ').first.str());
            nodeLines.push_back(s.str());
            widths.push_back(
                std::max(
                    kMinNodeWidth,
                    kCharWidth * nodeLabel.size() + kNodeMargin));
        } else if (level == 1 && s.consume_front("label=")) {
            label = s.rtrim(';').str();
        }
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    os << "  layout=neato;\n  splines=line;\n";
    if (!label.empty()) os << "  label=" << label << ";\n  labelloc=t;\n";
    unsigned numNodes = nodes.size();
    if (numNodes == 0) return result;

    std::map<std::string, unsigned> ids;
    for (auto [i, name] : llvm::enumerate(nodes)) ids.try_emplace(name, i);
    SmallVector<std::pair<int, int>> ends;
    SmallVector<SmallVector<std::pair<unsigned, size_t>>> outs(numNodes);
    std::map<std::pair<unsigned, unsigned>, size_t> edgeOf;
    for (auto [e, edge] : llvm::enumerate(edges)) {
        auto src = ids.find(edge.src);
        auto dst = ids.find(edge.dst);
        if (src == ids.end() || dst == ids.end()) {
            ends.push_back({-1, -1});
            continue;
        }
        ends.push_back({src->second, dst->second});
        outs[src->second].push_back({dst->second, e});
        edgeOf.try_emplace({src->second, dst->second}, e);
    }

    // Find back edges so that feedback loops do not shape the layout.
    enum Color { White, Gray, Black };
    SmallVector<Color> color(numNodes, White);
    SmallVector<bool> isBack(edges.size(), false);
    SmallVector<unsigned> postOrder;
    for (unsigned root = 0; root < numNodes; ++root) {
        if (color[root] != White) continue;
        SmallVector<std::pair<unsigned, unsigned>> stack{
            {root, 0}
        };
        color[root] = Gray;
        while (!stack.empty()) {
            auto &[v, next] = stack.back();
            if (next < outs[v].size()) {
                auto [w, e] = outs[v][next++];
                if (color[w] == Gray) {
                    isBack[e] = true;
                } else if (color[w] == White) {
                    color[w] = Gray;
                    stack.push_back({w, 0});
                }
            } else {
                color[v] = Black;
                postOrder.push_back(v);
                stack.pop_back();
            }
        }
    }
    SmallVector<SmallVector<unsigned>> succs(numNodes), preds(numNodes);
    for (auto [e, end] : llvm::enumerate(ends)) {
        if (end.first < 0 || isBack[e]) continue;
        succs[end.first].push_back(end.second);
        preds[end.second].push_back(end.first);
    }

    // Longest-path depth along the forward edges.
    SmallVector<unsigned> depth(numNodes, 0);
    for (unsigned v : llvm::reverse(postOrder))
        for (unsigned w : succs[v]) depth[w] = std::max(depth[w], depth[v] + 1);

    // The longest path is the spine the snake follows.
    unsigned last = 0;
    for (unsigned v = 1; v < numNodes; ++v)
        if (depth[v] > depth[last]) last = v;
    SmallVector<unsigned> spine{last};
    while (depth[spine.back()] > 0) {
        unsigned v = spine.back();
        spine.push_back(*llvm::find_if(preds[v], [&](unsigned p) {
            return depth[p] + 1 == depth[v];
        }));
    }
    std::reverse(spine.begin(), spine.end());

    // The other nodes form chains hanging off the spine or each other.
    SmallVector<bool> onSpine(numNodes, false), inChain(numNodes, false);
    for (unsigned v : spine) onSpine[v] = true;
    SmallVector<unsigned> byDepth(numNodes);
    std::iota(byDepth.begin(), byDepth.end(), 0);
    llvm::stable_sort(byDepth, [&](unsigned a, unsigned b) {
        return depth[a] < depth[b];
    });
    SmallVector<SmallVector<unsigned>> chains;
    for (unsigned v : byDepth) {
        if (onSpine[v] || inChain[v]) continue;
        SmallVector<unsigned> chain{v};
        inChain[v] = true;
        while (true) {
            SmallVector<unsigned> next;
            for (unsigned w : succs[chain.back()])
                if (!onSpine[w] && !inChain[w] && preds[w].size() == 1)
                    next.push_back(w);
            if (next.size() != 1) break;
            chain.push_back(next.front());
            inChain[next.front()] = true;
        }
        chains.push_back(std::move(chain));
    }

    // Rows leave room for lanes on both sides; empty rows are merged
    // afterwards.
    int maxLane = std::min(static_cast<int>(numNodes), kMaxLane);
    int gap = 2 * maxLane + 2;

    auto place = [&](int width) {
        GridLayout layout;
        auto &pos = layout.pos;
        pos.assign(numNodes, Cell{0, 0});
        layout.routes.resize(edges.size());
        SmallVector<bool> placed(numNodes, false), routed(edges.size(), false);
        SmallVector<int> dirs(numNodes, 1);
        Grid grid;
        auto setRoute = [&](size_t e, ArrayRef<Cell> pts) {
            layout.routes[e].assign(pts.begin(), pts.end());
            routed[e] = true;
        };

        // The spine snakes through rows of the given width.
        for (auto [s, v] : llvm::enumerate(spine)) {
            int row = s / width, offset = s % width;
            dirs[v] = row % 2 ? -1 : 1;
            pos[v] = {row % 2 ? width - 1 - offset : offset, row * gap};
            placed[v] = true;
            grid.commit({}, pos[v]);
        }
        for (auto [a, b] : llvm::zip(spine, llvm::drop_begin(spine))) {
            Cell pts[] = {pos[a], pos[b]};
            grid.commit(pts, {});
            setRoute(edgeOf.at({a, b}), pts);
        }

        // Each chain runs in a lane beside the node it leaves, ending beside
        // the node it joins.
        auto anchors = [&](ArrayRef<unsigned> chain) {
            int u = -1, w = -1;
            for (unsigned p : preds[chain.front()])
                if (u < 0 && placed[p]) u = p;
            for (unsigned s : succs[chain.back()])
                if (w < 0 && placed[s]) w = s;
            return std::make_pair(u, w);
        };
        auto span = [&](unsigned c) {
            auto [u, w] = anchors(chains[c]);
            if (u < 0 || w < 0) return std::numeric_limits<int>::max();
            return std::abs(pos[u].first - pos[w].first)
                   + std::abs(pos[u].second - pos[w].second);
        };
        auto ready = [&](unsigned c) {
            return llvm::all_of(preds[chains[c].front()], [&](unsigned p) {
                return placed[p];
            });
        };
        SmallVector<unsigned> pending(chains.size());
        std::iota(pending.begin(), pending.end(), 0);
        while (!pending.empty()) {
            // Tightest ready chain first, so that inner lanes go to inner
            // chains.
            auto pick = pending.end();
            for (auto it = pending.begin(); it != pending.end(); ++it)
                if (ready(*it)
                    && (pick == pending.end() || span(*it) < span(*pick)))
                    pick = it;
            if (pick == pending.end()) pick = pending.begin();
            ArrayRef<unsigned> chain = chains[*pick];
            pending.erase(pick);
            int u, w;
            std::tie(u, w) = anchors(chain);
            int m = chain.size();

            struct Candidate {
                RouteCost cost;
                SmallVector<Cell> pts, cells, bends;
                int dir;
            };
            std::optional<Candidate> best;
            auto consider = [&](ArrayRef<int> xs, int y, int dir) {
                Candidate cand{{}, {}, {}, {}, dir};
                if (u >= 0) {
                    cand.pts.push_back(pos[u]);
                    if (xs.front() != pos[u].first) {
                        cand.bends.push_back({pos[u].first, y});
                        cand.pts.push_back(cand.bends.back());
                    }
                }
                for (int x : xs) {
                    cand.cells.push_back({x, y});
                    cand.pts.push_back(cand.cells.back());
                }
                if (w >= 0) {
                    if (xs.back() != pos[w].first) {
                        cand.bends.push_back({pos[w].first, y});
                        cand.pts.push_back(cand.bends.back());
                    }
                    cand.pts.push_back(pos[w]);
                }
                SmallVector<Cell> occupied(cand.cells);
                occupied.append(cand.bends);
                auto cost = grid.cost(cand.pts, occupied);
                if (cost && (!best || cost->cost < best->cost.cost)) {
                    cand.cost = *cost;
                    best = std::move(cand);
                }
            };
            for (int k = 1; k <= maxLane && (u >= 0 || w >= 0); ++k) {
                for (int side : {1, -1}) {
                    SmallVector<int> xs;
                    if (u >= 0 && w >= 0) {
                        auto [ux, uy] = pos[u];
                        int wx = pos[w].first;
                        if (m - 1 > std::abs(wx - ux)) continue;
                        if (m == 1) {
                            consider({ux}, uy + side * k, dirs[u]);
                            consider({wx}, uy + side * k, dirs[u]);
                            continue;
                        }
                        int step = wx >= ux ? 1 : -1;
                        for (int j = 0; j + 1 < m; ++j)
                            xs.push_back(ux + step * j);
                        xs.push_back(wx);
                        consider(xs, uy + side * k, dirs[u]);
                    } else if (u >= 0) {
                        auto [ux, uy] = pos[u];
                        for (int j = 0; j < m; ++j)
                            xs.push_back(ux + dirs[u] * j);
                        consider(xs, uy + side * k, dirs[u]);
                    } else {
                        auto [wx, wy] = pos[w];
                        for (int j = 0; j < m; ++j)
                            xs.push_back(wx - dirs[w] * (m - 1 - j));
                        consider(xs, wy + side * k, dirs[w]);
                    }
                }
                // Farther lanes are only longer.
                if (best && best->cost.crossings == 0) break;
            }

            if (!best) {
                // Park the chain below everything; its edges are routed
                // later.
                layout.penalty += m;
                int bottom = 0;
                for (unsigned v = 0; v < numNodes; ++v)
                    if (placed[v]) bottom = std::max(bottom, pos[v].second);
                for (auto [j, v] : llvm::enumerate(chain)) {
                    pos[v] = {static_cast<int>(j), bottom + gap};
                    placed[v] = true;
                    grid.commit({}, pos[v]);
                }
                continue;
            }
            layout.penalty += best->cost.crossings;
            SmallVector<Cell> occupied(best->cells);
            occupied.append(best->bends);
            grid.commit(best->pts, occupied);
            for (auto [v, c] : llvm::zip(chain, best->cells)) {
                pos[v] = c;
                dirs[v] = best->dir;
                placed[v] = true;
            }
            // Split the polyline into the chain's edges.
            SmallVector<unsigned> seq;
            if (u >= 0) seq.push_back(u);
            seq.append(chain.begin(), chain.end());
            if (w >= 0) seq.push_back(w);
            SmallVector<Cell> piece;
            size_t j = 0;
            for (Cell c : best->pts) {
                piece.push_back(c);
                if (piece.size() > 1 && j + 1 < seq.size()
                    && c == pos[seq[j + 1]]) {
                    setRoute(edgeOf.at({seq[j], seq[j + 1]}), piece);
                    ++j;
                    piece = {c};
                }
            }
        }

        // Route the remaining edges straight, around one corner or through
        // a lane.
        for (auto [e, end] : llvm::enumerate(ends)) {
            auto [a, b] = end;
            if (a < 0 || a == b || routed[e]) continue;
            Cell pa = pos[a], pb = pos[b];
            std::optional<std::pair<RouteCost, SmallVector<Cell, 4>>> best;
            auto consider = [&](SmallVector<Cell, 4> pts) {
                pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
                if (pts.size() < 2) return;
                ArrayRef<Cell> bends =
                    ArrayRef<Cell>(pts).drop_front().drop_back();
                auto cost = grid.cost(pts, bends);
                if (cost && (!best || cost->cost < best->first.cost))
                    best.emplace(*cost, std::move(pts));
            };
            consider({pa, pb});
            consider({
                pa,
                {pb.first, pa.second},
                pb
            });
            consider({
                pa,
                {pa.first, pb.second},
                pb
            });
            for (int k = 1; k <= maxLane; ++k) {
                if (best && best->first.crossings == 0) break;
                for (int side : {1, -1}) {
                    int y = pa.second + side * k, x = pa.first + side * k;
                    consider({
                        pa,
                        {pa.first, y},
                        {pb.first, y},
                        pb
                    });
                    consider({
                        pa,
                        {x, pa.second},
                        {x, pb.second},
                        pb
                    });
                }
            }
            if (!best) {
                layout.penalty += kUnroutedPenalty;
                continue;
            }
            auto &pts = best->second;
            layout.penalty += best->first.crossings;
            grid.commit(pts, ArrayRef<Cell>(pts).drop_front().drop_back());
            setRoute(e, pts);
        }

        mergeRows(layout);
        return layout;
    };

    // Fold the spine into the row width that comes closest to a square.
    int spineLen = spine.size();
    int minWidth = spineLen < static_cast<int>(kMinFoldNodes)
                       ? spineLen
                       : std::min(spineLen, static_cast<int>(kMinRowNodes));
    GridLayout chosen;
    GridGeometry geo;
    double bestScore = std::numeric_limits<double>::infinity();
    for (int width = spineLen; width >= minWidth; --width) {
        GridLayout layout = place(width);
        GridGeometry candidate = measureGrid(layout, widths);
        double score = std::abs(std::log(candidate.width / candidate.height))
                       + kPenaltyWeight * layout.penalty;
        bool tooTall = candidate.height > 2 * candidate.width;
        if (score < bestScore) {
            bestScore = score;
            chosen = std::move(layout);
            geo = std::move(candidate);
        }
        if (tooTall) break;
    }

    auto coord = [&](Cell c) {
        return llvm::formatv(
                   "{0:F2},{1:F2}!",
                   geo.colX[c.first],
                   geo.rowY[c.second])
            .str();
    };
    for (auto [v, nodeLine] : llvm::enumerate(nodeLines)) {
        StringRef s(nodeLine);
        size_t close = s.rfind(']');
        os << "  " << s.take_front(close) << ", pos=\"" << coord(chosen.pos[v])
           << "\"" << s.drop_front(close) << "\n";
    }
    // Bends become invisible points so that every segment is straight.
    unsigned numBends = 0;
    for (auto [e, edge] : llvm::enumerate(edges)) {
        ArrayRef<Cell> route = chosen.routes[e];
        if (route.size() <= 2) {
            os << "  " << edge.line << "\n";
            continue;
        }
        SmallVector<std::string> names{edge.src};
        for (Cell c : route.drop_front().drop_back()) {
            names.push_back(rootName + "_bend_" + std::to_string(numBends++));
            os << "  " << names.back()
               << " [shape=point, width=0.01, label=\"\", pos=\"" << coord(c)
               << "\"];\n";
        }
        names.push_back(edge.dst);
        for (size_t i = 0; i + 1 < names.size(); ++i)
            os << "  " << names[i] << " -> " << names[i + 1]
               << (i + 2 < names.size() ? " [arrowhead=none];\n" : ";\n");
    }
    return result;
}

LogicalResult DotEmitter::emitGraph(ModuleOp module, raw_ostream &os)
{
    getGraph(module);
    for (auto [i, graph] : llvm::enumerate(topGraphs)) {
        os << "digraph G_" << i << " {\n";
        os << gridGraph(removeBlankLines(graph));
        os << "}\n";
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
