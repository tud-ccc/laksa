/// Implementation of the grid layout shared by the Dot translations
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Target/DotCommon/GridDot.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

using namespace mlir;

namespace {

// Layout geometry in inches, after the Graphviz defaults.
constexpr double kNodeHeight = 0.5;
constexpr double kColumnGap = 0.6;
constexpr double kRowGap = 0.5;
constexpr double kCharWidth = 0.1;
constexpr double kNodeMargin = 0.3;
constexpr double kMinNodeWidth = 0.75;
// Edge labels, drawn at font size 10.
constexpr double kEdgeCharWidth = 0.075;
constexpr double kEdgeLineHeight = 0.18;
constexpr double kLabelMargin = 0.4;
constexpr double kLabelOffset = 0.08;
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

GridGeometry measureGrid(
    const GridLayout &layout,
    ArrayRef<double> widths,
    double columnGap,
    double rowGap)
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
        x += w + columnGap;
    }
    geo.width = x - columnGap;
    for (int r = 0; r < numRows; ++r)
        geo.rowY.push_back((numRows - 1 - r) * (kNodeHeight + rowGap));
    geo.height = numRows * kNodeHeight + (numRows - 1) * rowGap;
    return geo;
}

} // namespace

std::string laksa::emitGridDot(
    StringRef title,
    StringRef bendPrefix,
    ArrayRef<GridDotNode> nodes,
    ArrayRef<GridDotEdge> edges)
{
    std::string result;
    llvm::raw_string_ostream os(result);
    os << "  layout=neato;\n  splines=line;\n";
    if (!title.empty()) os << "  label=\"" << title << "\";\n  labelloc=t;\n";
    unsigned numNodes = nodes.size();
    if (numNodes == 0) return result;

    SmallVector<double> widths;
    for (const auto &node : nodes)
        widths.push_back(
            std::max(
                kMinNodeWidth,
                kCharWidth * node.label.size() + kNodeMargin));

    // Edge labels need room between the nodes.
    auto labelSize = [](StringRef label) {
        SmallVector<StringRef> lines;
        label.split(lines, "\\n");
        size_t chars = 0;
        for (StringRef line : lines) chars = std::max(chars, line.size());
        return std::make_pair(
            kEdgeCharWidth * chars,
            kEdgeLineHeight * lines.size());
    };
    double columnGap = kColumnGap, rowGap = kRowGap;
    for (const auto &edge : edges) {
        if (edge.label.empty()) continue;
        auto [labelWidth, labelHeight] = labelSize(edge.label);
        columnGap = std::max(columnGap, labelWidth + kLabelMargin);
        rowGap = std::max(rowGap, labelHeight + kLabelMargin);
    }

    std::map<std::string, unsigned> ids;
    for (auto [i, node] : llvm::enumerate(nodes)) ids.try_emplace(node.name, i);
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
        GridGeometry candidate = measureGrid(layout, widths, columnGap, rowGap);
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
    for (auto [v, node] : llvm::enumerate(nodes))
        os << "  " << node.name << " [label=\"" << node.label
           << "\", shape=" << node.shape << ", pos=\"" << coord(chosen.pos[v])
           << "\"];\n";
    // Bends become invisible points; a label sits by the longest segment.
    unsigned numBends = 0, numLabels = 0;
    for (auto [e, edge] : llvm::enumerate(edges)) {
        ArrayRef<Cell> route = chosen.routes[e];
        if (route.size() < 2) {
            os << "  " << edge.src << " -> " << edge.dst;
            if (!edge.label.empty())
                os << " [label=\"" << edge.label << "\", fontsize=10]";
            os << ";\n";
            continue;
        }
        SmallVector<std::string> names{edge.src};
        for (Cell c : route.drop_front().drop_back()) {
            names.push_back((bendPrefix + "_bend_" + Twine(numBends++)).str());
            os << "  " << names.back()
               << " [shape=point, width=0.01, label=\"\", pos=\"" << coord(c)
               << "\"];\n";
        }
        names.push_back(edge.dst);
        for (size_t i = 0; i + 1 < names.size(); ++i) {
            os << "  " << names[i] << " -> " << names[i + 1];
            if (i + 2 < names.size()) os << " [arrowhead=none]";
            os << ";\n";
        }
        if (edge.label.empty()) continue;

        auto length = [&](size_t i) {
            return std::abs(route[i + 1].first - route[i].first)
                   + std::abs(route[i + 1].second - route[i].second);
        };
        size_t seg = 0;
        for (size_t i = 1; i + 1 < route.size(); ++i)
            if (length(i) > length(seg)) seg = i;
        int srcId = ends[e].first, dstId = ends[e].second;
        // Part of a segment end hidden inside its node.
        auto inset = [&](size_t i, bool horizontal) {
            if (i != 0 && i + 1 != route.size()) return 0.0;
            if (!horizontal) return kNodeHeight / 2;
            return widths[i == 0 ? srcId : dstId] / 2;
        };
        Cell p = route[seg], q = route[seg + 1];
        double x0 = geo.colX[p.first], y0 = geo.rowY[p.second];
        double x1 = geo.colX[q.first], y1 = geo.rowY[q.second];
        auto [labelWidth, labelHeight] = labelSize(edge.label);
        double x, y;
        if (p.second == q.second) {
            double d = x1 > x0 ? 1 : -1;
            x = (x0 + d * inset(seg, true) + x1 - d * inset(seg + 1, true)) / 2;
            y = y0 + labelHeight / 2 + kLabelOffset;
        } else {
            double d = y1 > y0 ? 1 : -1;
            x = x0 + labelWidth / 2 + kLabelOffset;
            y = (y0 + d * inset(seg, false) + y1 - d * inset(seg + 1, false))
                / 2;
        }
        os << "  " << bendPrefix << "_label_" << numLabels++
           << " [shape=plaintext, fontsize=10, width=0, height=0, margin=0, "
              "label=\""
           << edge.label << "\", pos=\""
           << llvm::formatv("{0:F2},{1:F2}!", x, y) << "\"];\n";
    }
    return result;
}
