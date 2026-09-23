// Implementation of PragmaDSE transform pass.
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "gurobi_c++.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <cmath>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "emithls-pragma-dse"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(llvm::dbgs() << "[emithls-pragma-dse] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSPRAGMADSE
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

SmallVector<int64_t> getValidFactors(int64_t n)
{
    SmallVector<int64_t> factors;
    if (n == 0) return factors;
    factors.push_back(1);
    for (int64_t i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            factors.push_back(i);
            if (i != n / i) factors.push_back(n / i);
        }
    }
    factors.push_back(n);
    std::sort(factors.begin(), factors.end());
    return factors;
}

//===----------------------------------------------------------------------===//
// GRBValue
//===----------------------------------------------------------------------===//

// Wraps a DSE quantity that may be a plain variable, a linear expression, or
// a quadratic expression, so callers can use it uniformly.
class GRBValue {
public:
    GRBValue() = default;
    GRBValue(GRBVar var) : kind(Kind::Var), var(std::move(var)) {}
    GRBValue(GRBLinExpr expr) : kind(Kind::Lin), lin(std::move(expr)) {}
    GRBValue(GRBQuadExpr expr) : kind(Kind::Quad), quad(std::move(expr)) {}
    GRBValue(int64_t constant) : kind(Kind::Lin), lin(double(constant)) {}

    bool isValid() const { return kind != Kind::None; }

    // Solved value, valid only after GRBModel::optimize() succeeded.
    double getValue() const
    {
        switch (kind) {
        case Kind::None: return 0.0;
        case Kind::Var: return var.get(GRB_DoubleAttr_X);
        case Kind::Lin: return valueOf(lin);
        case Kind::Quad: return valueOf(quad);
        }
        llvm_unreachable("unhandled GRBValue kind");
    }

    int64_t getIntValue() const
    { return static_cast<int64_t>(std::llround(getValue())); }

    // The expression itself, widened to a GRBQuadExpr so it works uniformly
    // in further constraints.
    GRBQuadExpr getExpr() const
    {
        switch (kind) {
        case Kind::None: return GRBQuadExpr(0.0);
        case Kind::Var: return GRBQuadExpr(var);
        case Kind::Lin: return GRBQuadExpr(lin);
        case Kind::Quad: return quad;
        }
        llvm_unreachable("unhandled GRBValue kind");
    }

private:
    static double valueOf(const GRBLinExpr &e)
    {
        double v = e.getConstant();
        for (int i = 0, n = e.size(); i < n; ++i)
            v += e.getCoeff(i) * e.getVar(i).get(GRB_DoubleAttr_X);
        return v;
    }
    static double valueOf(const GRBQuadExpr &q)
    {
        double v = valueOf(q.getLinExpr());
        for (int i = 0, n = q.size(); i < n; ++i)
            v += q.getCoeff(i) * q.getVar1(i).get(GRB_DoubleAttr_X)
                 * q.getVar2(i).get(GRB_DoubleAttr_X);
        return v;
    }

    enum class Kind { None, Var, Lin, Quad } kind = Kind::None;
    GRBVar var;
    GRBLinExpr lin;
    GRBQuadExpr quad;
};

//===----------------------------------------------------------------------===//
// DSE graph model
//===----------------------------------------------------------------------===//

// DSE decisions made for one "emithls.for" loop inside a node's callee.
struct LoopDSEInfo {
    ForOp loop;
    // How many iterations of this loop are peeled into parallel copies.
    GRBValue unrollFactor;
    // Array-partition factor this loop drives on the arrays it indexes
    GRBValue partitionFactor;
    // If this loop carries the PIPELINE pragma
    GRBValue pipelined;
    // Resolved nesting depth after interchange, outermost = 0
    GRBValue interchangePos;
};

// Each "CallOp" and its called-to "FuncOp" is a node in the graph
struct DSEEdge;
struct DSENode {
    CallOp call;
    FuncOp callee;
    // Edge connected to each call operand/port, indexed the same way as call's
    // operands
    SmallVector<DSEEdge*> ports;
    // Per-loop decisions inside "callee", keyed by the loop itself so the
    // rewrite step can look a ForOp up directly.
    DenseMap<Operation*, LoopDSEInfo> loops;
    // Per-dimension partition factor chosen for every array in "callee",
    // port or local.
    DenseMap<Value, SmallVector<std::pair<int64_t, GRBValue>>> arrayDimFactors;
    // Cycles needed to produce one output token at this node, this will later
    // feed the sizing of its outgoing edges' buffers.
    GRBValue producedTokenCycles;
    // Total cycles for one full invocation of "callee", feeding the
    // critical-path objective across nodes.
    GRBValue totalCycles;
    // This node's own share of the global DSP/BRAM totals, for the solution
    // print's per-callee breakdown.
    GRBValue totalDSP;
    GRBValue totalBRAM;
};
// One emithls.variable in the top function: connects exactly one producing node
// port to one consuming port.
struct DSEEdge {
    VariableOp variable;

    DSENode* producer = nullptr;
    int64_t producerPortIdx = -1;

    DSENode* consumer = nullptr;
    int64_t consumerPortIdx = -1;

    // FIFO depth is resolved from "producer->producedTokenCycles" once the
    // model is solved.
    GRBValue bufferDepth;
};
// The graph consists of nodes and edges
struct DSEGraph {
    SmallVector<std::unique_ptr<DSENode>> nodes;
    SmallVector<std::unique_ptr<DSEEdge>> edges;

    DenseMap<Operation*, DSENode*> callToNode;
    DenseMap<Value, DSEEdge*> variableToEdge;
};

bool isTopFunction(FuncOp funcOp)
{ return funcOp->hasAttr(laksa::kRootAttrName); }

// A memory/stream bridge has both a pointer port and a stream/array port,
// since it just moves data rather than computing.
bool isIOFunction(FuncOp funcOp)
{
    return llvm::any_of(
               funcOp.getArguments(),
               [](BlockArgument a) { return isa<PointerType>(a.getType()); })
           && llvm::any_of(funcOp.getArguments(), [](BlockArgument a) {
                  return isa<StreamType, emithls::ArrayType>(a.getType());
              });
}

// True for a direct, unindexed port read/write
bool isBarePortAccess(Operation* op)
{
    Value port =
        llvm::TypeSwitch<Operation*, Value>(op)
            .Case<StreamReadOp, StreamWriteOp>([](auto o) {
                return o.getIndices().empty() ? o.getStream() : Value();
            })
            .Case<ArrayPointerReadOp, ArrayPointerWriteOp>([](auto o) {
                return o.getIndices().empty() ? o.getPointer() : Value();
            })
            .Case<ArrayReadOp, ArrayWriteOp>([](auto o) {
                return o.getIndices().empty() ? o.getArray() : Value();
            })
            .Default(Value());
    return port && isa<BlockArgument>(port);
}

// Wraps ops into a size-1 loop
void normalizeLinearIslands(Block &block)
{
    SmallVector<Operation*> ops =
        llvm::to_vector(llvm::make_pointer_range(block));

    SmallVector<Operation*> run;
    auto flush = [&]() {
        if (run.empty()) return;
        for (Operation* op : run)
            if (auto nested = dyn_cast<ForOp>(op))
                normalizeLinearIslands(nested.getBody().front());
        if (llvm::any_of(run, isBarePortAccess)) {
            OpBuilder builder(run.front());
            auto wrapper =
                ForOp::create(builder, run.front()->getLoc(), 0, 1, 1);
            Block &body = wrapper.getBody().front();
            for (Operation* op : run) op->moveBefore(&body, body.end());
        }
        run.clear();
    };

    for (Operation* op : ops) {
        if (auto ifOp = dyn_cast<IfOp>(op)) {
            auto conflict = llvm::find_if(run, [&](Operation* run_op) {
                return llvm::is_contained(
                    run_op->getResults(),
                    ifOp.getCondition());
            });
            if (conflict != run.end()) run.erase(conflict, run.end());
            flush();
            normalizeLinearIslands(ifOp.getThenRegion().front());
            if (!ifOp.getElseRegion().empty())
                normalizeLinearIslands(ifOp.getElseRegion().front());
            continue;
        }
        run.push_back(op);
    }
    flush();
}

//===----------------------------------------------------------------------===//
// Split-candidate loop collection
//===----------------------------------------------------------------------===//

// How a candidate loop's split factor may be chosen.
enum class LoopSplitKind {
    // Any divisor of the trip count, via getValidFactors.
    Full,
    // Either 1 or the full trip count
    Binary,
    // A loop's split factor is deferred from connected producer/consumer
    Deferred,
};

struct CandidateLoop {
    ForOp loop;
    LoopSplitKind kind;
};

// How a loop's directly-touched stream port should be treated: Array for an
// indexed channel-port access, Scalar for an unindexed one, None if it touches
// no port.
enum class PortTouch { None, Array, Scalar };

PortTouch touchesPortDirectly(ForOp loop)
{
    for (Operation &op : loop.getBody().front()) {
        Value port;
        bool hasIndex;
        if (auto o = dyn_cast<StreamReadOp>(op)) {
            port = o.getStream();
            hasIndex = !o.getIndices().empty();
        } else if (auto o = dyn_cast<StreamWriteOp>(op)) {
            port = o.getStream();
            hasIndex = !o.getIndices().empty();
        } else
            continue;
        if (!isa<BlockArgument>(port)) continue;
        bool isArray = isa<emithls::ArrayType>(port.getType());
        if (isArray && hasIndex) return PortTouch::Array;
        if (!isArray && !hasIndex) return PortTouch::Scalar;
    }
    return PortTouch::None;
}

// True if "loop" directly writes a channel/pointer port
bool writesOutputPortDirectly(ForOp loop)
{
    for (Operation &op : loop.getBody().front()) {
        Value port;
        if (auto o = dyn_cast<StreamWriteOp>(op))
            port = o.getStream();
        else if (auto o = dyn_cast<ArrayPointerWriteOp>(op))
            port = o.getPointer();
        else if (auto o = dyn_cast<ArrayWriteOp>(op))
            port = o.getArray();
        else
            continue;
        if (isa<BlockArgument>(port)) return true;
    }
    return false;
}

// Inside an already-found candidate's subtree, a nested loop is itself a "Full"
// candidate only if it does real DSP work
void collectNestedCandidates(Block &block, SmallVectorImpl<CandidateLoop> &out)
{
    for (Operation &opRef : block) {
        if (auto loop = dyn_cast<ForOp>(&opRef)) {
            // A fused op only needs a DSP when it multiplies; "+=" and "-="
            // are just an adder.
            bool hasDSP =
                loop->walk([&](Operation* inner) {
                        auto fused = dyn_cast<ArithFusedOp>(inner);
                        bool isMul =
                            isa<ArithMulOp>(inner)
                            || (fused
                                && fused.getOpCode() == FusedOperator::mul);
                        return isMul ? WalkResult::interrupt()
                                     : WalkResult::advance();
                    })
                    .wasInterrupted();
            if (hasDSP) out.push_back({loop, LoopSplitKind::Full});
            collectNestedCandidates(loop.getBody().front(), out);
        } else if (auto ifOp = dyn_cast<IfOp>(&opRef)) {
            collectNestedCandidates(ifOp.getThenRegion().front(), out);
            if (!ifOp.getElseRegion().empty())
                collectNestedCandidates(ifOp.getElseRegion().front(), out);
        }
    }
}

// Classifies an anchor port loop. Accumulate feeding a direct port write is
// Binary; not writing a port directly, or writing one after real arithmetic,
// is Full; writing a port with no computation at all is Deferred.
LoopSplitKind classifyAnchor(ForOp loop)
{
    bool hasAccumulate = false;
    loop->walk([&](ArithFusedOp) { hasAccumulate = true; });
    if (hasAccumulate && writesOutputPortDirectly(loop))
        return LoopSplitKind::Binary;
    if (!writesOutputPortDirectly(loop)) return LoopSplitKind::Full;
    bool hasArithmetic =
        llvm::any_of(loop.getBody().front(), [](Operation &op) {
            return op.getName().getStringRef().starts_with("emithls.arith.");
        });
    return hasArithmetic ? LoopSplitKind::Full : LoopSplitKind::Deferred;
}

// Descends through outer order-preserving loops until it finds the first
// loop that directly touches a channel port; that loop and everything
// nested inside it become candidates.
void searchCandidateLoops(
    Block &block,
    bool ioFunc,
    SmallVectorImpl<CandidateLoop> &out)
{
    for (Operation &opRef : block) {
        if (auto loop = dyn_cast<ForOp>(&opRef)) {
            PortTouch touch = touchesPortDirectly(loop);
            // An IO bridge's scalar-port loop has exactly one channel by
            // construction, and it's just skipped like an order loop.
            bool isCandidate = touch == PortTouch::Array
                               || (touch == PortTouch::Scalar && !ioFunc);
            if (isCandidate) {
                LoopSplitKind kind =
                    ioFunc ? LoopSplitKind::Deferred : classifyAnchor(loop);
                out.push_back({loop, kind});
                collectNestedCandidates(loop.getBody().front(), out);
            } else {
                searchCandidateLoops(loop.getBody().front(), ioFunc, out);
            }
        } else if (auto ifOp = dyn_cast<IfOp>(&opRef)) {
            searchCandidateLoops(ifOp.getThenRegion().front(), ioFunc, out);
            if (!ifOp.getElseRegion().empty())
                searchCandidateLoops(ifOp.getElseRegion().front(), ioFunc, out);
        }
    }
}

//===----------------------------------------------------------------------===//
// Array memory classification (BRAM / LUTRAM)
//===----------------------------------------------------------------------===//

// If "v" is exactly the induction variable of some emithls.for, returns that
// loop; otherwise null.
ForOp getInductionVarOwner(Value v)
{
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg) return nullptr;
    auto loop = dyn_cast_or_null<ForOp>(arg.getOwner()->getParentOp());
    if (loop && loop.getInductionVariable() == v) return loop;
    return nullptr;
}

// A loop's relation to the found candidates
enum class LoopRole { Skip, Candidate, AutoUnrolled };

LoopRole classifyLoopRole(ForOp loop, const DenseSet<Operation*> &candidates)
{
    if (candidates.contains(loop)) return LoopRole::Candidate;
    for (Operation* anc = loop->getParentOp(); anc; anc = anc->getParentOp())
        if (auto ancFor = dyn_cast<ForOp>(anc))
            if (candidates.contains(ancFor)) return LoopRole::AutoUnrolled;
    return LoopRole::Skip;
}

struct ArrayAccess {
    Operation* op;
    SmallVector<Value> indices;
};
// Every read/write access to "arrayVar", plus both the target and the source
// side of an UpdateOp.
void collectArrayAccesses(Value arrayVar, SmallVectorImpl<ArrayAccess> &out)
{
    for (Operation* user : arrayVar.getUsers()) {
        llvm::TypeSwitch<Operation*>(user)
            .Case<ArrayReadOp, ArrayWriteOp>([&](auto op) {
                if (op.getArray() == arrayVar)
                    out.push_back({user, llvm::to_vector(op.getIndices())});
            })
            .Case<UpdateOp>([&](UpdateOp op) {
                if (op.getVariable() == arrayVar)
                    out.push_back({user, llvm::to_vector(op.getIndices())});
                if (op.getNewValue() == arrayVar)
                    out.push_back(
                        {user, llvm::to_vector(op.getNewValueIndices())});
            });
    }
}

// A shift-register style self-update, e.g. arr[i] = arr[j], can only be
// built from per-element registers, never a real addressable memory.
bool hasSelfReferentialUpdate(Value arrayVar)
{
    return llvm::any_of(arrayVar.getUsers(), [&](Operation* user) {
        auto update = dyn_cast<UpdateOp>(user);
        return update && update.getVariable() == arrayVar
               && update.getNewValue() == arrayVar;
    });
}

// Const tables and shift-register-style arrays both collapse to registers.
bool isLUTRAM(VariableOp varOp)
{ return varOp.getIsConst() || hasSelfReferentialUpdate(varOp.getVariable()); }

// How a dimension is partitioned:
// "Loop" if every access indexes it with some loop, combined via the max
// of their eventual factors;
// "Sequential" if every contributing loop is order-preserving;
// "Full" otherwise.
enum class DimPartitionKind { Loop, Sequential, Full };

struct DimPartition {
    DimPartitionKind kind;
    SmallVector<ForOp> loops; // non-empty iff kind == Loop
    // Set only when "kind == Full" because an index wasn't attributable to any
    // loop, purely so the printer can explain why the fallback fired.
    Value conflictA;
};

SmallVector<DimPartition> classifyArrayDims(
    Value arrayVar,
    emithls::ArrayType arrType,
    const DenseSet<Operation*> &candidates)
{
    size_t rank = arrType.getShape().size();
    SmallVector<ArrayAccess> accesses;
    collectArrayAccesses(arrayVar, accesses);

    SmallVector<DimPartition> dims(
        rank,
        DimPartition{DimPartitionKind::Full, {}, nullptr});
    for (size_t d = 0; d < rank; ++d) {
        SmallVector<ForOp> touchingLoops;
        bool unattributable = accesses.empty();
        bool autoUnrolled = false;
        for (ArrayAccess &access : accesses) {
            if (d >= access.indices.size()) {
                unattributable = true;
                dims[d].conflictA =
                    access.indices.empty() ? Value() : access.indices.back();
                break;
            }
            Value idx = access.indices[d];
            ForOp loop = getInductionVarOwner(idx);
            if (!loop) {
                unattributable = true;
                dims[d].conflictA = idx;
                break;
            }
            if (classifyLoopRole(loop, candidates) == LoopRole::AutoUnrolled)
                autoUnrolled = true;
            if (!llvm::is_contained(touchingLoops, loop))
                touchingLoops.push_back(loop);
        }
        if (unattributable || autoUnrolled) continue; // stays Full

        SmallVector<ForOp> candidateLoops;
        for (ForOp loop : touchingLoops)
            if (classifyLoopRole(loop, candidates) == LoopRole::Candidate)
                candidateLoops.push_back(loop);
        dims[d] = candidateLoops.empty()
                      ? DimPartition{DimPartitionKind::Sequential, {}, nullptr}
                      : DimPartition{
                            DimPartitionKind::Loop,
                            std::move(candidateLoops),
                            nullptr};
    }
    return dims;
}

int64_t getScalarBitWidth(Type type)
{
    if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
    return 0;
}

// A Xilinx BRAM18 block holds 18Kb = 18432 bits.
constexpr int64_t kBramBits = 18432;
// Largest split factor a Full or Deferred candidate loop may pick.
constexpr int64_t kMaxUnrollFactor = 32;
// Largest FIFO depth taken from a consumer's per-token cycles, before merge
// padding.
constexpr int64_t kMaxFIFODepth = 32;

//===----------------------------------------------------------------------===//
// DSE graph
//===----------------------------------------------------------------------===//

std::unique_ptr<DSEGraph> buildGraph(FuncOp topFunc)
{
    auto graph = std::make_unique<DSEGraph>();

    for (auto callOp : topFunc.getBody().front().getOps<CallOp>()) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
            callOp,
            callOp.getCalleeAttr());
        auto node = std::make_unique<DSENode>();
        node->call = callOp;
        node->callee = callee;
        node->ports.resize(callOp.getArgOperands().size());
        graph->callToNode[callOp] = node.get();
        graph->nodes.push_back(std::move(node));
    }

    for (auto varOp : topFunc.getBody().front().getOps<VariableOp>()) {
        auto edge = std::make_unique<DSEEdge>();
        edge->variable = varOp;

        for (Operation* user : varOp.getVariable().getUsers()) {
            auto callOp = dyn_cast<CallOp>(user);
            if (!callOp) continue;
            DSENode* node = graph->callToNode.lookup(callOp);
            if (!node) continue;

            for (auto [argIdx, callArg] :
                 llvm::enumerate(callOp.getArgOperands())) {
                if (callArg != varOp.getVariable()) continue;
                node->ports[argIdx] = edge.get();
                BlockArgument calleeArg = node->callee.getArgument(argIdx);
                // An output port is one some op inside the callee writes to.
                if (llvm::any_of(calleeArg.getUsers(), [](Operation* user) {
                        return isa<
                            StreamWriteOp,
                            ArrayWriteOp,
                            ArrayPointerWriteOp>(user);
                    })) {
                    edge->producer = node;
                    edge->producerPortIdx = argIdx;
                } else {
                    edge->consumer = node;
                    edge->consumerPortIdx = argIdx;
                }
            }
        }

        graph->variableToEdge[varOp.getVariable()] = edge.get();
        graph->edges.push_back(std::move(edge));
    }

    return graph;
}

//===----------------------------------------------------------------------===//
// ILP: loop split-factor variables
//===----------------------------------------------------------------------===//

// Every port, dimension pair "loop" directly drives via a stream read/write
// access indexed by its own induction variable. Usually one pair, but a
// port with rank > 1 needs one entry per dimension it indexes, and a
// passthrough copy loop can touch two different ports in the same body.
SmallVector<std::pair<Value, int64_t>> getTouchedPortDims(ForOp loop)
{
    SmallVector<std::pair<Value, int64_t>> result;
    for (Operation &op : loop.getBody().front()) {
        Value port;
        ValueRange indices;
        if (auto o = dyn_cast<StreamReadOp>(op)) {
            port = o.getStream();
            indices = o.getIndices();
        } else if (auto o = dyn_cast<StreamWriteOp>(op)) {
            port = o.getStream();
            indices = o.getIndices();
        } else
            continue;
        if (!isa<BlockArgument>(port)) continue;
        for (auto [d, idx] : llvm::enumerate(indices))
            if (getInductionVarOwner(idx) == loop)
                result.push_back({port, static_cast<int64_t>(d)});
    }
    return result;
}

// Adds a "pick exactly one of validFactors" selector to the model and returns
// the resulting split-factor value, printing what was available.
GRBValue chooseSplitFactor(
    GRBModel &model,
    ForOp loop,
    ArrayRef<int64_t> validFactors,
    const std::string &namePrefix)
{
    LAKSA_DEBUG(
        llvm::dbgs() << "    " << loop.getLoc() << ": valid factors = [";
        for (auto [i, f] : llvm::enumerate(validFactors)) {
            if (i) llvm::dbgs() << ", ";
            llvm::dbgs() << f;
        } llvm::dbgs()
        << "]");

    if (validFactors.size() == 1) return GRBValue(validFactors.front());

    SmallVector<GRBVar> choose;
    choose.reserve(validFactors.size());
    for (auto [idx, factor] : llvm::enumerate(validFactors))
        choose.push_back(model.addVar(
            0.0,
            1.0,
            0.0,
            GRB_BINARY,
            namePrefix + "_choose" + std::to_string(idx)));

    GRBLinExpr sumChoose = 0;
    for (GRBVar &c : choose) sumChoose += c;
    model.addConstr(sumChoose == 1, namePrefix + "_pick_one");

    GRBLinExpr link = 0;
    for (auto [factor, c] : llvm::zip(validFactors, choose))
        link += double(factor) * c;

    GRBVar factorVar = model.addVar(
        1.0,
        double(validFactors.back()),
        0.0,
        GRB_INTEGER,
        namePrefix + "_factor");
    model.addConstr(factorVar == link, namePrefix + "_factor_link");
    return GRBValue(factorVar);
}

// Sets up one candidate loop's split-factor variable: Full/Binary pick freely
// among their own valid factors; Deferred has no basis of its own and is
// instead tied to its connected port's factor once every node's own loops are
// set up.
GRBValue setupLoopUnrollFactor(
    GRBModel &model,
    CandidateLoop candidate,
    const std::string &namePrefix)
{
    ForOp loop = candidate.loop;
    int64_t tripCount = loop.getTripCount();
    switch (candidate.kind) {
    case LoopSplitKind::Full:
    case LoopSplitKind::Binary:
    {
        // Binary picks from every divisor, not capped like Full, since it's
        // also tied to whatever factor a connected port needs, which isn't
        // necessarily 1 or its own trip count. Being forced to exactly its
        // trip count only kicks in later, and only if it ends up pipelined.
        SmallVector<int64_t> factors = getValidFactors(tripCount);
        if (candidate.kind == LoopSplitKind::Full)
            llvm::erase_if(factors, [](int64_t f) {
                return f > kMaxUnrollFactor;
            });
        return chooseSplitFactor(model, loop, factors, namePrefix);
    }
    case LoopSplitKind::Deferred:
        LAKSA_DEBUG(
            llvm::dbgs()
            << "    " << loop.getLoc() << ": deferred, tied to connected port");
        return GRBValue(model.addVar(
            1.0,
            double(std::min(tripCount, kMaxUnrollFactor)),
            0.0,
            GRB_INTEGER,
            namePrefix + "_factor"));
    }
    llvm_unreachable("unhandled LoopSplitKind");
}

// Binds "value" to a fresh GRBVar via an equality constraint, so it can be
// used wherever Gurobi's API specifically requires a GRBVar rather than an
// arbitrary expression.
GRBVar
materializeVar(GRBModel &model, const GRBValue &value, const std::string &name)
{
    GRBVar v = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, name);
    // "value" can genuinely be quadratic, so this needs addQConstr since
    // addConstr rejects any quadratic terms at runtime.
    model.addQConstr(
        GRBQuadExpr(v) == value.getExpr(),
        (name + "_link").c_str());
    return v;
}

// The greater of two values, materializing each into a plain GRBVar first
// since addGenConstrMax requires that.
GRBValue maxOfTwo(
    GRBModel &model,
    const GRBValue &a,
    const GRBValue &b,
    const std::string &namePrefix)
{
    SmallVector<GRBVar> vars;
    vars.push_back(materializeVar(model, a, namePrefix + "_a"));
    vars.push_back(materializeVar(model, b, namePrefix + "_b"));
    GRBVar maxVar =
        model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, namePrefix + "_max");
    model.addGenConstrMax(
        maxVar,
        vars.data(),
        static_cast<int>(vars.size()),
        -GRB_INFINITY,
        namePrefix + "_max_constr");
    return GRBValue(maxVar);
}

// The greatest of "values" (which must be non-empty), folded pairwise via
// maxOfTwo.
GRBValue maxOfAll(
    GRBModel &model,
    ArrayRef<GRBValue> values,
    const std::string &namePrefix)
{
    GRBValue result = values.front();
    for (size_t i = 1; i < values.size(); ++i)
        result =
            maxOfTwo(model, result, values[i], namePrefix + std::to_string(i));
    return result;
}

// The lesser of "value" and the constant "bound".
GRBValue minWithConstant(
    GRBModel &model,
    const GRBValue &value,
    int64_t bound,
    const std::string &namePrefix)
{
    GRBVar var = materializeVar(model, value, namePrefix + "_in");
    GRBVar minVar =
        model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, namePrefix + "_min");
    model.addGenConstrMin(
        minVar,
        &var,
        1,
        double(bound),
        namePrefix + "_min_constr");
    return GRBValue(minVar);
}

// Populates "loops" on every node in "graph" with a split-factor variable for
// each of its callee's candidates, then ties every Deferred loop's factor to
// whatever its connected producer/consumer port resolved to.
void setupLoopSplitFactors(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    for (auto &node : graph.nodes) {
        std::string funcName = node->callee.getSymName().str();
        LAKSA_DEBUG(llvm::dbgs() << "  callee \"" << funcName << "\":");
        auto candidates = candidatesByFunc.lookup(node->callee.getOperation());
        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            GRBValue factor = setupLoopUnrollFactor(
                model,
                candidate,
                funcName + "_loop" + std::to_string(idx));
            node->loops[candidate.loop.getOperation()] =
                LoopDSEInfo{candidate.loop, factor, {}, {}, {}};
        }
    }

    // A port dimension needs exactly the factor of the loop(s) that index it
    // directly there, no separate choice of its own. A single dimension can
    // be driven by more than one candidate loop in the same node, which
    // must then all share that one factor, regardless of kind.
    LAKSA_DEBUG(llvm::dbgs() << "  Add constraints matching loop factor");
    for (auto &node : graph.nodes) {
        DenseMap<std::pair<Value, int64_t>, SmallVector<ForOp>> loopsByPortDim;
        for (CandidateLoop candidate :
             candidatesByFunc.lookup(node->callee.getOperation()))
            for (auto &portDim : getTouchedPortDims(candidate.loop))
                loopsByPortDim[portDim].push_back(candidate.loop);

        for (auto &[portDim, loops] : loopsByPortDim) {
            auto &[port, dim] = portDim;
            ForOp canonicalLoop = loops.front();
            GRBValue &canonical =
                node->loops[canonicalLoop.getOperation()].unrollFactor;
            int64_t portIdx = cast<BlockArgument>(port).getArgNumber();

            LAKSA_DEBUG(
                llvm::dbgs()
                << "    " << node->callee.getSymName() << " port" << portIdx
                << " dim" << dim << " <- " << canonicalLoop.getLoc());
            node->arrayDimFactors[port].push_back({dim, canonical});

            for (size_t i = 1; i < loops.size(); ++i) {
                ForOp sibling = loops[i];
                std::string name = node->callee.getSymName().str() + "_port"
                                   + std::to_string(portIdx) + "_dim"
                                   + std::to_string(dim) + "_sibling"
                                   + std::to_string(i);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    constraint " << name << ": " << sibling.getLoc()
                    << " == " << canonicalLoop.getLoc());
                GRBValue &siblingFactor =
                    node->loops[sibling.getOperation()].unrollFactor;
                model.addConstr(
                    siblingFactor.getExpr() == canonical.getExpr(),
                    name);
            }
        }
    }

    // A producer and its consumer only need to agree on the port's static
    // shape, not on matching parallelism: a slower producer feeding a
    // faster consumer is fine, since the buffer between them absorbs the
    // difference, but a producer running faster than its consumer would
    // grow that buffer unboundedly. So this is a one-directional
    // consumer-factor >= producer-factor constraint, not equality.
    LAKSA_DEBUG(
        llvm::dbgs()
        << "  Add constraints keeping up with connected producers");
    for (auto &node : graph.nodes) {
        for (CandidateLoop candidate :
             candidatesByFunc.lookup(node->callee.getOperation())) {
            for (auto &[port, dim] : getTouchedPortDims(candidate.loop)) {
                int64_t portIdx = cast<BlockArgument>(port).getArgNumber();
                DSEEdge* edge = node->ports[portIdx];
                if (!edge || edge->consumer != node.get() || !edge->producer)
                    continue;
                DSENode* producer = edge->producer;
                int64_t producerPortIdx = edge->producerPortIdx;

                // Any loop on the producer's side that drives the same port
                // dimension works: the sibling pass above already tied every
                // such loop in "producer" to one factor.
                ForOp producerLoop;
                for (CandidateLoop producerCandidate :
                     candidatesByFunc.lookup(producer->callee.getOperation())) {
                    for (auto &[producerPort, producerDim] :
                         getTouchedPortDims(producerCandidate.loop)) {
                        if (cast<BlockArgument>(producerPort).getArgNumber()
                                == producerPortIdx
                            && producerDim == dim) {
                            producerLoop = producerCandidate.loop;
                            break;
                        }
                    }
                    if (producerLoop) break;
                }
                if (!producerLoop) continue;

                std::string name = node->callee.getSymName().str() + "_port"
                                   + std::to_string(portIdx) + "_dim"
                                   + std::to_string(dim) + "_keeps_up_with_"
                                   + producer->callee.getSymName().str();
                LAKSA_DEBUG(
                    llvm::dbgs() << "    constraint " << name << ": "
                                 << candidate.loop.getLoc()
                                 << " >= " << producerLoop.getLoc());
                GRBValue &consumerFactor =
                    node->loops[candidate.loop.getOperation()].unrollFactor;
                GRBValue &producerFactor =
                    producer->loops[producerLoop.getOperation()].unrollFactor;
                model.addConstr(
                    consumerFactor.getExpr() >= producerFactor.getExpr(),
                    name);

                // The port's own physical width has to cover whichever end
                // needs more parallelism, so record the max of the two on
                // both ends' arrayDimFactors in place of each side's own
                // value.
                GRBValue portFactor = maxOfTwo(
                    model,
                    producerFactor,
                    consumerFactor,
                    name + "_max");
                Value producerPort =
                    producer->callee.getArgument(producerPortIdx);
                for (auto &[dd, f] : node->arrayDimFactors[port])
                    if (dd == dim) f = portFactor;
                for (auto &[dd, f] : producer->arrayDimFactors[producerPort])
                    if (dd == dim) f = portFactor;
            }
        }
    }
}

// Chain-multiplies "factors" into one product, materializing every
// intermediate product since Gurobi only supports quadratic (degree-2)
// expressions natively.
GRBValue chainProduct(
    GRBModel &model,
    ArrayRef<GRBValue> factors,
    const std::string &namePrefix)
{
    if (factors.empty()) return GRBValue(int64_t(1));
    if (factors.size() == 1) return factors.front();

    GRBVar acc = materializeVar(model, factors.front(), namePrefix + "_acc0");
    for (size_t i = 1; i < factors.size(); ++i) {
        GRBVar next = materializeVar(
            model,
            factors[i],
            namePrefix + "_f" + std::to_string(i));
        acc = materializeVar(
            model,
            GRBValue(GRBQuadExpr(acc * next)),
            namePrefix + "_acc" + std::to_string(i));
    }
    return GRBValue(acc);
}

// The factor a dimension driven by "loops" needs: the max of their
// individual factors, since each loop still picks its own independently and
// the dimension just needs enough partitions for whichever needs the most
// concurrent access.
GRBValue maxOfLoopFactors(
    GRBModel &model,
    DenseMap<Operation*, LoopDSEInfo> &nodeLoops,
    ArrayRef<ForOp> loops,
    const std::string &namePrefix)
{
    if (loops.size() == 1) {
        ForOp loop = loops.front();
        return nodeLoops[loop.getOperation()].unrollFactor;
    }

    SmallVector<GRBVar> vars;
    vars.reserve(loops.size());
    for (auto [i, constLoop] : llvm::enumerate(loops)) {
        ForOp loop = constLoop;
        vars.push_back(materializeVar(
            model,
            nodeLoops[loop.getOperation()].unrollFactor,
            namePrefix + "_in" + std::to_string(i)));
    }

    GRBVar maxVar =
        model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, namePrefix + "_max");
    model.addGenConstrMax(
        maxVar,
        vars.data(),
        static_cast<int>(vars.size()),
        -GRB_INFINITY,
        namePrefix + "_max_constr");
    return GRBValue(maxVar);
}

// Populates "arrayDimFactors" for every local array in each node's callee: a
// Loop-kind dimension needs the max factor of the loops that drive it.
// Sequential/Full dimensions need no loop-driven factor.
void setupArrayDimFactors(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    LAKSA_DEBUG(
        llvm::dbgs() << "  Add constraints for memory dimension factors");
    for (auto &node : graph.nodes) {
        DenseSet<Operation*> candidateSet;
        for (CandidateLoop c :
             candidatesByFunc.lookup(node->callee.getOperation()))
            candidateSet.insert(c.loop);

        int64_t arrayIdx = 0;
        node->callee.walk([&](VariableOp varOp) {
            auto arrType =
                dyn_cast<emithls::ArrayType>(varOp.getVariable().getType());
            if (!arrType) return;
            int64_t thisArrayIdx = arrayIdx++;

            auto dims =
                classifyArrayDims(varOp.getVariable(), arrType, candidateSet);
            for (auto [d, dim] : llvm::enumerate(dims)) {
                if (dim.kind != DimPartitionKind::Loop) continue;

                std::string namePrefix =
                    node->callee.getSymName().str() + "_array"
                    + std::to_string(thisArrayIdx) + "_dim" + std::to_string(d);
                GRBValue factor =
                    maxOfLoopFactors(model, node->loops, dim.loops, namePrefix);

                LAKSA_DEBUG(
                    llvm::dbgs()
                        << "    " << node->callee.getSymName() << " "
                        << varOp->getLoc() << " dim" << d << " <- max(";
                    for (auto [i, loop] : llvm::enumerate(dim.loops)) {
                        if (i) llvm::dbgs() << ", ";
                        llvm::dbgs() << loop.getLoc();
                    } llvm::dbgs()
                    << ")");
                node->arrayDimFactors[varOp.getVariable()].push_back(
                    {static_cast<int64_t>(d), factor});
            }
        });
    }
}

// Total BRAM18K blocks one local array needs: 0 for LUTRAM, otherwise a
// compile-time baseline from the Full-kind dimensions' static sizes, times the
// product of every Loop-kind dimension's now-resolved factor.
GRBValue computeArrayBRAM(
    GRBModel &model,
    DSENode &node,
    VariableOp varOp,
    emithls::ArrayType arrType,
    ArrayRef<DimPartition> dims,
    const std::string &namePrefix)
{
    if (isLUTRAM(varOp)) {
        LAKSA_DEBUG(
            llvm::dbgs() << "    " << varOp->getLoc() << " BRAM: 0 (LUTRAM)");
        return GRBValue(int64_t(0));
    }

    int64_t fullFactor = 1;
    for (auto [d, dim] : llvm::enumerate(dims))
        if (dim.kind == DimPartitionKind::Full)
            fullFactor *= arrType.getShape()[d];
    int64_t perInstanceBits = arrType.getNumElements() / fullFactor
                              * getScalarBitWidth(arrType.getElementType());
    int64_t baselineBlocks = (perInstanceBits + kBramBits - 1) / kBramBits;

    SmallVector<GRBValue> loopFactors;
    for (auto [d, dim] : llvm::enumerate(dims)) {
        if (dim.kind != DimPartitionKind::Loop) continue;
        for (auto &[dd, factor] : node.arrayDimFactors[varOp.getVariable()])
            if (dd == static_cast<int64_t>(d)) loopFactors.push_back(factor);
    }

    LAKSA_DEBUG(
        llvm::dbgs()
        << "    " << varOp->getLoc() << " BRAM: baseline=" << baselineBlocks
        << " BRAM18K x " << fullFactor << " (full-partitioned dims) x "
        << loopFactors.size() << " loop-driven dim(s)");
    GRBValue loopFactorProduct =
        chainProduct(model, loopFactors, namePrefix + "_loopfactor");
    return GRBValue(
        loopFactorProduct.getExpr() * double(baselineBlocks * fullFactor));
}

// Adds the single global constraint that every local array's BRAM usage,
// summed across every node, must fit within "availableBRAM" BRAM18K blocks;
// returns that total for callers that also want to report it.
GRBValue addBRAMBudgetConstraint(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc,
    int64_t availableBRAM)
{
    GRBQuadExpr totalBRAM = 0;
    for (auto &node : graph.nodes) {
        DenseSet<Operation*> candidateSet;
        for (CandidateLoop c :
             candidatesByFunc.lookup(node->callee.getOperation()))
            candidateSet.insert(c.loop);

        GRBQuadExpr nodeTotalBRAM = 0;
        int64_t arrayIdx = 0;
        node->callee.walk([&](VariableOp varOp) {
            auto arrType =
                dyn_cast<emithls::ArrayType>(varOp.getVariable().getType());
            if (!arrType) return;
            int64_t thisArrayIdx = arrayIdx++;

            auto dims =
                classifyArrayDims(varOp.getVariable(), arrType, candidateSet);
            GRBValue bram = computeArrayBRAM(
                model,
                *node,
                varOp,
                arrType,
                dims,
                node->callee.getSymName().str() + "_array"
                    + std::to_string(thisArrayIdx) + "_bram");
            nodeTotalBRAM += bram.getExpr();
        });
        node->totalBRAM = GRBValue(nodeTotalBRAM);
        totalBRAM += nodeTotalBRAM;
    }
    model.addQConstr(
        totalBRAM,
        GRB_LESS_EQUAL,
        double(availableBRAM),
        "bram_budget");
    return GRBValue(totalBRAM);
}

//===----------------------------------------------------------------------===//
// ILP: DSP usage
//===----------------------------------------------------------------------===//

// The type "value" was cast from, tracing back through intervening
// arithmetic to find a direct emithls.arith.cast; null if the chain bottoms
// out at a plain variable or constant instead.
Type getPreCastType(Value value)
{
    Operation* def = value.getDefiningOp();
    if (!def) return nullptr;
    if (auto castOp = dyn_cast<ArithCastOp>(def))
        return castOp.getFrom().getType();
    if (isa<VariableOp>(def)) return nullptr;
    for (Value operand : def->getOperands())
        if (Type preCast = getPreCastType(operand)) return preCast;
    return nullptr;
}

// One multiplication's DSP-relevant bitwidth: the pre-cast type of whichever
// operand was cast right before this use, or their own shared width if
// neither was.
int64_t getMulOperandBits(Value a, Value b)
{
    if (Type preCast = getPreCastType(a)) return getScalarBitWidth(preCast);
    if (Type preCast = getPreCastType(b)) return getScalarBitWidth(preCast);
    return getScalarBitWidth(a.getType());
}

// DSP48s one multiplication needs, 8 bits per DSP.
int64_t getMulDSPUsage(Value lhs, Value rhs)
{ return (getMulOperandBits(lhs, rhs) + 7) / 8; }

// Adds up the DSPs every multiplication directly in "block" needs, into a
// single iteration's total. Descends into if/else since that's the same
// iteration, but not into a nested "emithls.for", which gets its own
// iteration cost accounted for separately once composed in.
void accumulateIterationDSP(Block &block, int64_t &dsp)
{
    for (Operation &op : block) {
        if (isa<ForOp>(op)) continue;
        if (auto mulOp = dyn_cast<ArithMulOp>(op)) {
            dsp += getMulDSPUsage(mulOp.getLhs(), mulOp.getRhs());
        } else if (auto fusedOp = dyn_cast<ArithFusedOp>(op)) {
            if (fusedOp.getOpCode() == FusedOperator::mul)
                dsp += getMulDSPUsage(fusedOp.getAcc(), fusedOp.getVal());
        } else if (auto ifOp = dyn_cast<IfOp>(op)) {
            accumulateIterationDSP(ifOp.getThenRegion().front(), dsp);
            if (!ifOp.getElseRegion().empty())
                accumulateIterationDSP(ifOp.getElseRegion().front(), dsp);
        }
    }
}

// DSP48s one iteration of "loop"'s own body needs, not counting whatever
// nested loops inside it need for their own iterations.
int64_t computeIterationDSP(ForOp loop)
{
    int64_t dsp = 0;
    accumulateIterationDSP(loop.getBody().front(), dsp);
    return dsp;
}

// How many parallel hardware copies of "loop"'s body exist: its own factor
// times every enclosing candidate loop's factor.
GRBValue computeInstanceMultiplier(
    GRBModel &model,
    DenseMap<Operation*, LoopDSEInfo> &nodeLoops,
    ForOp loop,
    const DenseSet<Operation*> &candidateSet,
    const std::string &namePrefix)
{
    SmallVector<GRBValue> factors;
    factors.push_back(nodeLoops[loop.getOperation()].unrollFactor);
    for (Operation* anc = loop->getParentOp(); anc; anc = anc->getParentOp())
        if (auto ancFor = dyn_cast<ForOp>(anc))
            if (candidateSet.contains(ancFor))
                factors.push_back(
                    nodeLoops[ancFor.getOperation()].unrollFactor);
    return chainProduct(model, factors, namePrefix);
}

// Adds the single global constraint that total DSP usage, summed across
// every node, must fit within "availableDSP" DSP48s. Each candidate loop
// with non-zero per-iteration usage contributes that usage times however
// many parallel hardware copies of it exist. Returns the total for callers
// that also want to report or optimize it.
GRBValue addDSPBudgetConstraint(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc,
    int64_t availableDSP)
{
    GRBQuadExpr total = 0;
    for (auto &node : graph.nodes) {
        auto candidates = candidatesByFunc.lookup(node->callee.getOperation());
        DenseSet<Operation*> candidateSet;
        for (CandidateLoop c : candidates) candidateSet.insert(c.loop);
        std::string funcName = node->callee.getSymName().str();

        GRBQuadExpr nodeTotal = 0;
        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            int64_t iterDSP = computeIterationDSP(candidate.loop);
            if (iterDSP == 0) continue;

            GRBValue multiplier = computeInstanceMultiplier(
                model,
                node->loops,
                candidate.loop,
                candidateSet,
                funcName + "_loop" + std::to_string(idx) + "_dsp_instances");
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    " << funcName << " " << candidate.loop.getLoc() << ": "
                << iterDSP << " DSP/iter x instances");
            nodeTotal += multiplier.getExpr() * double(iterDSP);
        }
        node->totalDSP = GRBValue(nodeTotal);
        total += nodeTotal;
    }
    GRBValue totalDSP(total);
    model.addQConstr(
        totalDSP.getExpr(),
        GRB_LESS_EQUAL,
        double(availableDSP),
        "dsp_budget");
    return totalDSP;
}

//===----------------------------------------------------------------------===//
// ILP: pipeline decisions
//===----------------------------------------------------------------------===//

// Sets up a binary pipelined decision for every candidate loop, plus what
// pipelining one forces. Every other candidate loop nested inside it must
// fully unroll, since a pipelined loop's body has to be one purely
// combinational per-iteration block with no sequential sub-loops left
// inside it. A Binary-kind loop pipelining itself also forces its own
// factor to its trip count, since with only one shared accumulator each
// iteration still has to wait for the previous one to finish; it only
// pipelines usefully once fully duplicated across its trip count.
void setupPipelineConstraints(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    LAKSA_DEBUG(llvm::dbgs() << "  Add constraints for pipeline decisions");
    for (auto &node : graph.nodes) {
        auto candidates = candidatesByFunc.lookup(node->callee.getOperation());
        std::string funcName = node->callee.getSymName().str();

        DenseMap<Operation*, int64_t> indexOf;
        DenseSet<Operation*> candidateSet;
        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            indexOf[candidate.loop.getOperation()] = idx;
            candidateSet.insert(candidate.loop);
        }

        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            GRBVar pipelined = model.addVar(
                0.0,
                1.0,
                0.0,
                GRB_BINARY,
                funcName + "_loop" + std::to_string(idx) + "_pipelined");
            node->loops[candidate.loop.getOperation()].pipelined =
                GRBValue(pipelined);
        }

        // Exactly one loop in each independent pipeline chain (candidates
        // rooted at the same outermost candidate ancestor) is pipelined:
        // never more than one, and never none.
        DenseMap<Operation*, SmallVector<ForOp>> chainMembers;
        for (CandidateLoop candidate : candidates) {
            ForOp root = candidate.loop;
            for (Operation* anc = candidate.loop->getParentOp(); anc;
                 anc = anc->getParentOp())
                if (auto ancFor = dyn_cast<ForOp>(anc))
                    if (candidateSet.contains(ancFor)) root = ancFor;
            chainMembers[root.getOperation()].push_back(candidate.loop);
        }
        for (auto &[rootOp, members] : chainMembers) {
            GRBQuadExpr sum = 0;
            for (ForOp member : members)
                sum += node->loops[member.getOperation()].pipelined.getExpr();
            std::string name = funcName + "_loop"
                               + std::to_string(indexOf[rootOp])
                               + "_chain_pick_one_pipeline";
            LAKSA_DEBUG(
                llvm::dbgs() << "    constraint " << name << ": exactly one of "
                             << members.size() << " loop(s) pipelined");
            model.addQConstr(sum, GRB_EQUAL, 1.0, name);
        }

        DenseMap<Operation*, SmallVector<ForOp>> descendantsOf;
        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            ForOp loop = candidate.loop;
            int64_t tripCount = loop.getTripCount();
            GRBValue &factor = node->loops[loop.getOperation()].unrollFactor;

            if (candidate.kind == LoopSplitKind::Binary) {
                GRBValue &pipelined =
                    node->loops[loop.getOperation()].pipelined;
                std::string name = funcName + "_loop" + std::to_string(idx)
                                   + "_pipeline_forces_self_unroll";
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    constraint " << name << ": " << loop.getLoc()
                    << " pipelined => factor == " << tripCount);
                model.addConstr(
                    factor.getExpr() >= double(tripCount) * pipelined.getExpr(),
                    name);
            }

            for (Operation* anc = loop->getParentOp(); anc;
                 anc = anc->getParentOp()) {
                auto ancFor = dyn_cast<ForOp>(anc);
                if (!ancFor) continue;
                auto ancIdxIt = indexOf.find(ancFor.getOperation());
                if (ancIdxIt == indexOf.end()) continue;

                descendantsOf[ancFor.getOperation()].push_back(loop);

                GRBValue &ancPipelined =
                    node->loops[ancFor.getOperation()].pipelined;
                std::string name = funcName + "_loop" + std::to_string(idx)
                                   + "_unrolled_under_loop"
                                   + std::to_string(ancIdxIt->second);
                LAKSA_DEBUG(
                    llvm::dbgs()
                    << "    constraint " << name << ": " << ancFor.getLoc()
                    << " pipelined => " << loop.getLoc()
                    << " factor == " << tripCount);
                model.addConstr(
                    factor.getExpr()
                        >= double(tripCount) * ancPipelined.getExpr(),
                    name);
            }
        }

        // A loop that contains the chain's pipeline point, i.e. one of its
        // own descendants is the pipelined one, has no basis to split at
        // all and must stay fully sequential. This must only look at
        // descendants, never at "loop" itself being pipelined, since that
        // one case must stay free, or self-forced if Binary.
        for (auto [idx, candidate] : llvm::enumerate(candidates)) {
            ForOp loop = candidate.loop;
            int64_t tripCount = loop.getTripCount();
            GRBValue &factor = node->loops[loop.getOperation()].unrollFactor;

            // "pipelined" is always a plain binary (never a product), so
            // this stays a genuinely linear sum despite the GRBQuadExpr
            // static type .getExpr() returns.
            GRBLinExpr hasPipelinedDescendant = 0;
            for (ForOp descendant : descendantsOf[loop.getOperation()])
                hasPipelinedDescendant += node->loops[descendant.getOperation()]
                                              .pipelined.getExpr()
                                              .getLinExpr();

            // tripCount when no descendant is pipelined, so no extra
            // restriction; 1 when one is, forcing factor == 1.
            GRBLinExpr rhs = double(tripCount);
            rhs -= double(tripCount - 1) * hasPipelinedDescendant;

            std::string name = funcName + "_loop" + std::to_string(idx)
                               + "_contains_pipeline_point_forces_sequential";
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    constraint " << name << ": " << loop.getLoc()
                << " contains the pipeline point => factor == 1");
            model.addQConstr(factor.getExpr() <= rhs, name);
        }
    }
}

//===----------------------------------------------------------------------===//
// ILP: cycle computation
//===----------------------------------------------------------------------===//

// How many times a loop with the given trip count and ILP-chosen split
// factor actually executes in hardware: factor * iterations == tripCount,
// added as a genuine quadratic equality since factor is a variable and
// dividing by it can't be expressed linearly. Since factor only ever takes
// divisor values, iterations gets pinned to the exact matching quotient.
GRBValue computeIterationCount(
    GRBModel &model,
    const GRBValue &factor,
    int64_t tripCount,
    const std::string &name)
{
    GRBVar iterations =
        model.addVar(1.0, double(tripCount), 0.0, GRB_INTEGER, name);
    GRBLinExpr factorLin = factor.getExpr().getLinExpr();
    model.addQConstr(
        factorLin * iterations,
        GRB_EQUAL,
        double(tripCount),
        name + "_link");
    return GRBValue(iterations);
}

// Every op directly in "block", 1 cycle each: the same walk shape as
// accumulateIterationDSP, but counting every op instead of just
// multiplications.
void accumulateDirectOps(Block &block, int64_t &count)
{
    for (Operation &op : block) {
        if (isa<ForOp>(op)) continue;
        if (auto ifOp = dyn_cast<IfOp>(op)) {
            accumulateDirectOps(ifOp.getThenRegion().front(), count);
            if (!ifOp.getElseRegion().empty())
                accumulateDirectOps(ifOp.getElseRegion().front(), count);
            continue;
        }
        // A fused op is a read-modify-write into the accumulator, so it
        // costs 2 cycles, not 1.
        count += isa<ArithFusedOp>(op) ? 2 : 1;
    }
}

int64_t countDirectOps(Block &block)
{
    int64_t count = 0;
    accumulateDirectOps(block, count);
    return count;
}

// Per-iteration latency of a chain's leaf loop.
int64_t computeLeafLatency(ForOp leaf)
{
    return computeIterationDSP(leaf) > 0
               ? countDirectOps(leaf.getBody().front())
               : 2;
}

// Latency of one pass through the chain rooted at "root" with every loop in
// it fully unrolled.
int64_t computeUnrolledChainLatency(
    ForOp root,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf)
{
    ForOp cur = root;
    while (true) {
        SmallVector<ForOp> children =
            immediateChildOf.lookup(cur.getOperation());
        if (children.empty()) return computeLeafLatency(cur);
        cur = children.front();
    }
}

// One candidate chain's total cycles, from anchor "root" down to its leaf.
// Every chain has exactly one pipeline point, so it runs as one continuous
// pipeline: the leaf's own op count sets the fill/drain latency, and every
// loop's own iteration count along the way feeds the total step count.
// "extraIterationMultiplier" folds in the trip count of any enclosing
// non-candidate wrapper loop, since those are just more iterations of the
// same pipeline. Assumes a linear chain; branching isn't handled yet.
GRBValue computeChainCycles(
    GRBModel &model,
    DSENode &node,
    ForOp root,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    int64_t extraIterationMultiplier,
    const std::string &namePrefix)
{
    SmallVector<GRBValue> iterationsChain;
    ForOp cur = root;
    int64_t leafCyclesPerIter = 0;
    for (int64_t depth = 0;; ++depth) {
        GRBValue &factor = node.loops[cur.getOperation()].unrollFactor;
        GRBValue iterations = computeIterationCount(
            model,
            factor,
            cur.getTripCount(),
            namePrefix + "_iters" + std::to_string(depth));
        iterationsChain.push_back(iterations);

        SmallVector<ForOp> children =
            immediateChildOf.lookup(cur.getOperation());
        if (children.empty()) {
            leafCyclesPerIter = computeLeafLatency(cur);
            LAKSA_DEBUG(
                llvm::dbgs() << "    " << cur.getLoc() << ": leaf, "
                             << leafCyclesPerIter << " cycles/iter");
            break;
        }
        LAKSA_DEBUG(llvm::dbgs() << "    " << cur.getLoc() << ": chained");
        cur = children.front();
    }

    GRBValue combinedIterations =
        chainProduct(model, iterationsChain, namePrefix + "_combined_iters");
    GRBLinExpr rhs = combinedIterations.getExpr().getLinExpr();
    rhs *= double(extraIterationMultiplier);
    rhs += double(leafCyclesPerIter - 1);
    GRBVar cyclesVar =
        materializeVar(model, GRBValue(rhs), namePrefix + "_cycles");
    LAKSA_DEBUG(
        llvm::dbgs() << "    " << root.getLoc()
                     << ": chain cycles = latency + combined-iterations x "
                     << extraIterationMultiplier << " - 1");
    return GRBValue(cyclesVar);
}

GRBValue computeBlockCycles(
    GRBModel &model,
    DSENode &node,
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    const std::string &namePrefix);

// Collects the chain roots in "block" and its fully unrolled latency, if
// every loop in it is a chain root or a trip-count-1 wrapper of such.
bool collectMergeableChains(
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    SmallVectorImpl<ForOp> &roots,
    int64_t &latency)
{
    latency = 0;
    for (Operation &op : block) {
        if (auto loop = dyn_cast<ForOp>(op)) {
            if (candidateSet.contains(loop)) {
                roots.push_back(loop);
                latency += computeUnrolledChainLatency(loop, immediateChildOf);
                continue;
            }
            int64_t inner = 0;
            if (loop.getTripCount() != 1
                || !collectMergeableChains(
                    loop.getBody().front(),
                    candidateSet,
                    immediateChildOf,
                    roots,
                    inner))
                return false;
            latency += inner;
        } else if (auto ifOp = dyn_cast<IfOp>(op)) {
            int64_t thenLatency = 0;
            int64_t elseLatency = 0;
            if (!collectMergeableChains(
                    ifOp.getThenRegion().front(),
                    candidateSet,
                    immediateChildOf,
                    roots,
                    thenLatency))
                return false;
            if (!ifOp.getElseRegion().empty()
                && !collectMergeableChains(
                    ifOp.getElseRegion().front(),
                    candidateSet,
                    immediateChildOf,
                    roots,
                    elseLatency))
                return false;
            latency += std::max(thenLatency, elseLatency);
        }
    }
    return true;
}

// Cycles for a block of several sibling chains under non-candidate loops
// totalling "extraIterationMultiplier" iterations: one pipeline over those
// iterations if every chain is pipelined at its root and fully unrolled,
// otherwise the chains' plain sequential sum repeated per iteration.
std::optional<GRBValue> tryMergeBlock(
    GRBModel &model,
    DSENode &node,
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    int64_t extraIterationMultiplier,
    const std::string &namePrefix)
{
    SmallVector<ForOp> roots;
    int64_t latency = 0;
    if (!collectMergeableChains(
            block,
            candidateSet,
            immediateChildOf,
            roots,
            latency)
        || roots.empty())
        return std::nullopt;

    GRBVar merged =
        model.addVar(0.0, 1.0, 0.0, GRB_BINARY, namePrefix + "_merged");
    for (auto [i, root] : llvm::enumerate(roots)) {
        LoopDSEInfo &info = node.loops[root.getOperation()];
        std::string name = namePrefix + "_merged_root" + std::to_string(i);
        model.addConstr(
            GRBLinExpr(merged) <= info.pipelined.getExpr().getLinExpr(),
            name + "_pipelined");
        model.addConstr(
            info.unrollFactor.getExpr().getLinExpr()
                >= double(root.getTripCount()) * merged,
            name + "_unrolled");
    }

    GRBValue sequential = computeBlockCycles(
        model,
        node,
        block,
        candidateSet,
        immediateChildOf,
        namePrefix + "_seq");
    GRBVar sequentialCycles = materializeVar(
        model,
        GRBValue(sequential.getExpr() * double(extraIterationMultiplier)),
        namePrefix + "_seq_cycles");

    GRBVar cyclesVar = model.addVar(
        0.0,
        GRB_INFINITY,
        0.0,
        GRB_INTEGER,
        namePrefix + "_cycles");
    model.addGenConstrIndicator(
        merged,
        1,
        GRBLinExpr(cyclesVar) == double(extraIterationMultiplier + latency - 1),
        namePrefix + "_use_merged");
    model.addGenConstrIndicator(
        merged,
        0,
        GRBLinExpr(cyclesVar) == sequentialCycles,
        namePrefix + "_use_sequential");
    LAKSA_DEBUG(
        llvm::dbgs()
        << "    " << roots.size() << " sibling chain(s): merged cycles = "
        << extraIterationMultiplier << " + " << latency
        << " - 1, else sequential x " << extraIterationMultiplier);
    return GRBValue(cyclesVar);
}

// If "block" contains nothing but a single loop, all the way down to a
// candidate chain's root, returns that chain's cycles with every unwrapped
// loop's own trip count folded into its combined iterations. Also succeeds
// through a single if/else whose branches both flatten, since a
// port-touching if/else's two branches are always tied to the same factor,
// so max(then, else) gives the same one-pipeline formula. Returns null if
// "block" has anything else in it anywhere along the way.
std::optional<GRBValue> tryFlattenToChain(
    GRBModel &model,
    DSENode &node,
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    int64_t extraIterationMultiplier,
    const std::string &namePrefix)
{
    // Only a ForOp or IfOp is ever "real work" to computeBlockCycles;
    // anything else is already treated as free and silently skipped, so
    // only those count against "exactly one thing here" for flattening.
    auto isChainOp = [](Operation &op) { return isa<ForOp, IfOp>(op); };
    if (llvm::count_if(block, isChainOp) > 1)
        return tryMergeBlock(
            model,
            node,
            block,
            candidateSet,
            immediateChildOf,
            extraIterationMultiplier,
            namePrefix);
    if (llvm::count_if(block, isChainOp) != 1) return std::nullopt;
    Operation &only = *llvm::find_if(block, isChainOp);

    if (auto loop = dyn_cast<ForOp>(only)) {
        if (candidateSet.contains(loop)) {
            LAKSA_DEBUG(
                llvm::dbgs()
                << "    " << loop.getLoc()
                << ": chain root, wrapped by non-candidate loop(s) x"
                << extraIterationMultiplier);
            return computeChainCycles(
                model,
                node,
                loop,
                immediateChildOf,
                extraIterationMultiplier,
                namePrefix);
        }
        return tryFlattenToChain(
            model,
            node,
            loop.getBody().front(),
            candidateSet,
            immediateChildOf,
            extraIterationMultiplier * loop.getTripCount(),
            namePrefix);
    }

    if (auto ifOp = dyn_cast<IfOp>(only)) {
        if (ifOp.getElseRegion().empty()) return std::nullopt;
        auto thenCycles = tryFlattenToChain(
            model,
            node,
            ifOp.getThenRegion().front(),
            candidateSet,
            immediateChildOf,
            extraIterationMultiplier,
            namePrefix + "_then");
        auto elseCycles = tryFlattenToChain(
            model,
            node,
            ifOp.getElseRegion().front(),
            candidateSet,
            immediateChildOf,
            extraIterationMultiplier,
            namePrefix + "_else");
        if (!thenCycles || !elseCycles) return std::nullopt;
        LAKSA_DEBUG(
            llvm::dbgs() << "    " << ifOp.getLoc()
                         << ": both branches flatten, max(then, else)");
        return maxOfTwo(model, *thenCycles, *elseCycles, namePrefix + "_max");
    }

    return std::nullopt;
}

// Cycles for everything directly in "block": candidate loops get
// computeChainCycles; a non-candidate loop contributes its own body's
// cycles times its raw trip count; if/else branches are mutually
// exclusive, so contribute max(then, else), not both summed. Every direct
// child adds linearly.
GRBValue computeBlockCycles(
    GRBModel &model,
    DSENode &node,
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    const std::string &namePrefix)
{
    GRBQuadExpr total = 0;
    int64_t idx = 0;
    for (Operation &opRef : block) {
        if (auto loop = dyn_cast<ForOp>(&opRef)) {
            std::string name = namePrefix + "_b" + std::to_string(idx++);
            if (candidateSet.contains(loop)) {
                total += computeChainCycles(
                             model,
                             node,
                             loop,
                             immediateChildOf,
                             /*extraIterationMultiplier=*/1,
                             name)
                             .getExpr();
            } else if (
                auto flattened = tryFlattenToChain(
                    model,
                    node,
                    loop.getBody().front(),
                    candidateSet,
                    immediateChildOf,
                    loop.getTripCount(),
                    name)) {
                total += flattened->getExpr();
            } else {
                int64_t tripCount = loop.getTripCount();
                GRBValue inner = computeBlockCycles(
                    model,
                    node,
                    loop.getBody().front(),
                    candidateSet,
                    immediateChildOf,
                    name);
                LAKSA_DEBUG(
                    llvm::dbgs() << "    " << loop.getLoc()
                                 << ": not a candidate, x" << tripCount);
                total += inner.getExpr() * double(tripCount);
            }
        } else if (auto ifOp = dyn_cast<IfOp>(&opRef)) {
            std::string name = namePrefix + "_b" + std::to_string(idx++);
            GRBValue thenCycles = computeBlockCycles(
                model,
                node,
                ifOp.getThenRegion().front(),
                candidateSet,
                immediateChildOf,
                name + "_then");
            GRBValue elseCycles = !ifOp.getElseRegion().empty()
                                      ? computeBlockCycles(
                                            model,
                                            node,
                                            ifOp.getElseRegion().front(),
                                            candidateSet,
                                            immediateChildOf,
                                            name + "_else")
                                      : GRBValue(int64_t(0));
            GRBValue branchCycles =
                maxOfTwo(model, thenCycles, elseCycles, name + "_max");
            LAKSA_DEBUG(
                llvm::dbgs() << "    " << ifOp.getLoc() << ": max(then, else)");
            total += branchCycles.getExpr();
        }
    }
    return GRBValue(total);
}

// Cycles for a single token's own traversal through "node": the cycles of
// one execution of its innermost repeating body, skipping past any outer
// non-candidate loop that does nothing but wrap the whole thing, the same
// "single thing here" test tryFlattenToChain uses, but without multiplying
// its trip count in, since that outer loop is what repeats across
// different tokens, not part of any one token's own cost.
GRBValue computeProducedTokenCycles(
    GRBModel &model,
    DSENode &node,
    Block &block,
    const DenseSet<Operation*> &candidateSet,
    const DenseMap<Operation*, SmallVector<ForOp>> &immediateChildOf,
    const std::string &namePrefix)
{
    auto isChainOp = [](Operation &op) { return isa<ForOp, IfOp>(op); };
    if (llvm::count_if(block, isChainOp) == 1) {
        Operation &only = *llvm::find_if(block, isChainOp);
        auto loop = dyn_cast<ForOp>(only);
        if (loop && !candidateSet.contains(loop))
            return computeProducedTokenCycles(
                model,
                node,
                loop.getBody().front(),
                candidateSet,
                immediateChildOf,
                namePrefix);
    }
    return computeBlockCycles(
        model,
        node,
        block,
        candidateSet,
        immediateChildOf,
        namePrefix);
}

// Total cycles for one full invocation of "node"'s callee, walking its
// whole body; sets node.totalCycles. Also sets node.producedTokenCycles to
// one token's own traversal cost (see computeProducedTokenCycles), used for
// FIFO sizing rather than the whole-invocation total. Builds each
// candidate's set of immediate nested candidate children once, since
// computeChainCycles' leaf detection depends on it.
GRBValue computeNodeCycles(
    GRBModel &model,
    DSENode &node,
    ArrayRef<CandidateLoop> candidates)
{
    DenseSet<Operation*> candidateSet;
    for (CandidateLoop c : candidates) candidateSet.insert(c.loop);

    DenseMap<Operation*, SmallVector<ForOp>> immediateChildOf;
    for (CandidateLoop c : candidates) {
        ForOp nearest;
        for (Operation* anc = c.loop->getParentOp(); anc;
             anc = anc->getParentOp()) {
            auto ancFor = dyn_cast<ForOp>(anc);
            if (ancFor && candidateSet.contains(ancFor)) {
                nearest = ancFor;
                break;
            }
        }
        if (nearest) immediateChildOf[nearest.getOperation()].push_back(c.loop);
    }

    std::string funcName = node.callee.getSymName().str();
    LAKSA_DEBUG(llvm::dbgs() << "  callee \"" << funcName << "\":");
    GRBValue cycles = computeBlockCycles(
        model,
        node,
        node.callee.getBody().front(),
        candidateSet,
        immediateChildOf,
        funcName);
    node.totalCycles = cycles;
    node.producedTokenCycles = computeProducedTokenCycles(
        model,
        node,
        node.callee.getBody().front(),
        candidateSet,
        immediateChildOf,
        funcName + "_token");
    LAKSA_DEBUG(
        llvm::dbgs()
        << "  \"" << funcName << "\": one token's own traversal cost computed");
    return cycles;
}

// The longest path summed along "field" from any source down to and including
// "node": its own value plus whichever producer has the longest path of its
// own, maxed at a merge and summed along a plain chain. Memoized per field,
// since totalCycles and producedTokenCycles are different quantities that must
// not share a memo.
GRBValue computePathTo(
    GRBModel &model,
    DSENode &node,
    GRBValue DSENode::* field,
    DenseMap<DSENode*, GRBValue> &memo,
    const std::string &namePrefix)
{
    if (auto it = memo.find(&node); it != memo.end()) return it->second;

    SmallVector<GRBValue> producerPaths;
    for (DSEEdge* edge : node.ports) {
        if (!edge || edge->consumer != &node || !edge->producer) continue;
        producerPaths.push_back(computePathTo(
            model,
            *edge->producer,
            field,
            memo,
            edge->producer->callee.getSymName().str()));
    }

    GRBValue longestIncoming =
        producerPaths.empty()
            ? GRBValue(int64_t(0))
            : maxOfAll(model, producerPaths, namePrefix + "_in");

    GRBQuadExpr sum = (node.*field).getExpr();
    sum += longestIncoming.getExpr();
    GRBValue result(sum);
    LAKSA_DEBUG(
        llvm::dbgs()
        << "  " << node.callee.getSymName() << ": path = own value + max("
        << producerPaths.size() << " incoming producer path(s))");

    memo[&node] = result;
    return result;
}

GRBValue computeCriticalPathTo(
    GRBModel &model,
    DSENode &node,
    DenseMap<DSENode*, GRBValue> &memo,
    const std::string &namePrefix)
{ return computePathTo(model, node, &DSENode::totalCycles, memo, namePrefix); }

// Minimizing this is the actual DSE objective: not a flat sum of every
// node's cycles, but the graph's critical path, the longest source-to-sink
// path through the dataflow graph, maxed over every sink. A single linear
// chain of nodes degenerates to that chain's plain sum; only an actual
// fork/merge in the dataflow makes this differ from just adding every node
// up.
GRBValue computeGraphTotalCycles(
    GRBModel &model,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    for (auto &node : graph.nodes) {
        if (isIOFunction(node->callee)) continue;
        computeNodeCycles(
            model,
            *node,
            candidatesByFunc.lookup(node->callee.getOperation()));
    }

    if (graph.nodes.empty()) return GRBValue(int64_t(0));

    DenseSet<DSENode*> hasConsumer;
    for (auto &edge : graph.edges)
        if (edge->producer && edge->consumer)
            hasConsumer.insert(edge->producer);

    DenseMap<DSENode*, GRBValue> memo;
    SmallVector<GRBValue> sinkPaths;
    for (auto &node : graph.nodes) {
        if (hasConsumer.contains(node.get())) continue;
        LAKSA_DEBUG(
            llvm::dbgs() << "  sink \"" << node->callee.getSymName() << "\":");
        sinkPaths.push_back(computeCriticalPathTo(
            model,
            *node,
            memo,
            node->callee.getSymName().str()));
    }

    return maxOfAll(model, sinkPaths, "graph_critical_path");
}

//===----------------------------------------------------------------------===//
// FIFO buffer sizing
//===----------------------------------------------------------------------===//

// Sets "producedTokenCycles" to a fixed 2 cycles for every IO bridge node,
// since computeNodeCycles never runs on those. Sizes every producer or
// consumer edge's FIFO depth to the consumer's producedTokenCycles, so its
// producer never stalls pushing into it, then at a merge pads every
// incoming edge by the gap between its own producer's per-token path and
// the slowest producer's, so a faster branch doesn't stall waiting on a
// slower one and deadlock the diamond.
void computeBufferDepths(GRBModel &model, DSEGraph &graph)
{
    LAKSA_DEBUG(llvm::dbgs() << "  Sizing FIFO buffer depths:");
    for (auto &node : graph.nodes)
        if (isIOFunction(node->callee))
            node->producedTokenCycles = GRBValue(int64_t(2));

    for (auto [idx, edge] : llvm::enumerate(graph.edges)) {
        if (!edge->producer || !edge->consumer) continue;
        edge->bufferDepth = minWithConstant(
            model,
            edge->consumer->producedTokenCycles,
            kMaxFIFODepth,
            "edge" + std::to_string(idx) + "_depth");
        LAKSA_DEBUG(
            llvm::dbgs() << "    " << edge->variable.getLoc() << ": "
                         << edge->producer->callee.getSymName() << " -> "
                         << edge->consumer->callee.getSymName()
                         << ", depth <- min(consumer's producedTokenCycles, "
                         << kMaxFIFODepth << ")");
    }

    DenseMap<DSENode*, SmallVector<DSEEdge*>> incomingByConsumer;
    for (auto &edge : graph.edges)
        if (edge->producer && edge->consumer)
            incomingByConsumer[edge->consumer].push_back(edge.get());

    DenseMap<DSENode*, GRBValue> tokenPathMemo;
    for (auto &[consumer, incoming] : incomingByConsumer) {
        if (incoming.size() < 2) continue;
        LAKSA_DEBUG(
            llvm::dbgs() << "    merge at \"" << consumer->callee.getSymName()
                         << "\": " << incoming.size()
                         << " incoming branch(es), padding for skew");

        SmallVector<GRBValue> producerPaths;
        for (DSEEdge* edge : incoming)
            producerPaths.push_back(computePathTo(
                model,
                *edge->producer,
                &DSENode::producedTokenCycles,
                tokenPathMemo,
                edge->producer->callee.getSymName().str()));
        GRBValue slowest = maxOfAll(
            model,
            producerPaths,
            consumer->callee.getSymName().str() + "_merge_path");

        for (auto [i, edge] : llvm::enumerate(incoming)) {
            GRBQuadExpr depth = edge->bufferDepth.getExpr();
            depth += slowest.getExpr();
            depth -= producerPaths[i].getExpr();
            edge->bufferDepth = GRBValue(depth);
        }
    }
}

//===----------------------------------------------------------------------===//
// Custom Printing
//===----------------------------------------------------------------------===//

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, LoopSplitKind kind)
{
    switch (kind) {
    case LoopSplitKind::Full: return os << "Full(getValidFactors)";
    case LoopSplitKind::Binary:
        return os << "Binary(getValidFactors, tripcount if pipelined)";
    case LoopSplitKind::Deferred: return os << "Deferred(matches edge)";
    }
    llvm_unreachable("unhandled LoopSplitKind");
}

void printCandidateLoops(llvm::raw_ostream &os, ArrayRef<CandidateLoop> loops)
{
    for (CandidateLoop c : loops) {
        os << "    " << c.loop.getLoc() << ": "
           << c.loop.getLowerBound().getSExtValue() << " to "
           << c.loop.getUpperBound().getSExtValue() << " step "
           << c.loop.getStep().getSExtValue() << "  kind=" << c.kind << "\n";
    }
}

void printIterationDSP(llvm::raw_ostream &os, ArrayRef<CandidateLoop> loops)
{
    for (CandidateLoop c : loops)
        os << "    " << c.loop.getLoc() << ": " << computeIterationDSP(c.loop)
           << " DSP/iter\n";
}

void printArrayMemories(
    llvm::raw_ostream &os,
    FuncOp func,
    ArrayRef<CandidateLoop> candidates)
{
    DenseSet<Operation*> candidateSet;
    DenseMap<Operation*, LoopSplitKind> kindOf;
    for (CandidateLoop c : candidates) {
        candidateSet.insert(c.loop);
        kindOf[c.loop] = c.kind;
    }

    func.walk([&](VariableOp varOp) {
        auto arrType =
            dyn_cast<emithls::ArrayType>(varOp.getVariable().getType());
        if (!arrType) return;

        // Which dimensions need concurrent access is about the access
        // pattern, not the memory technology, so LUTRAM arrays get the same
        // per-dimension analysis as BRAM ones.
        auto dims =
            classifyArrayDims(varOp.getVariable(), arrType, candidateSet);

        os << "    " << varOp->getLoc() << ": " << arrType
           << "  elemBits=" << getScalarBitWidth(arrType.getElementType())
           << "  mem=";
        if (isLUTRAM(varOp)) {
            os << "LUTRAM\n";
        } else {
            // Every “Full”-partitioned dimension mandatorily splits the array
            // into that many independent physical memories
            int64_t fullFactor = 1;
            for (auto [d, dim] : llvm::enumerate(dims))
                if (dim.kind == DimPartitionKind::Full)
                    fullFactor *= arrType.getShape()[d];
            int64_t perInstanceBits =
                arrType.getNumElements() / fullFactor
                * getScalarBitWidth(arrType.getElementType());
            int64_t perInstanceBlocks =
                (perInstanceBits + kBramBits - 1) / kBramBits;
            os << "BRAM  baseline=" << perInstanceBlocks << " BRAM18K";
            if (fullFactor > 1)
                os << " x " << fullFactor << " = "
                   << perInstanceBlocks * fullFactor << " BRAM18K";
            os << "\n";
        }

        for (auto [d, dim] : llvm::enumerate(dims)) {
            os << "      dim" << d << "[" << arrType.getShape()[d] << "]: ";
            switch (dim.kind) {
            case DimPartitionKind::Loop:
                os << "factor <- max(";
                for (auto [i, loop] : llvm::enumerate(dim.loops)) {
                    if (i) os << ", ";
                    os << loop.getLoc() << " (" << kindOf.lookup(loop) << ")";
                }
                os << ")\n";
                break;
            case DimPartitionKind::Sequential:
                os << "factor = 1 (sequential loop)\n";
                break;
            case DimPartitionKind::Full:
                os << "factor = " << arrType.getShape()[d]
                   << " (full partition";
                if (Value v = dim.conflictA) {
                    os << ": ";
                    if (ForOp loop = getInductionVarOwner(v))
                        os << "loop@" << loop.getLoc();
                    else if (Operation* def = v.getDefiningOp())
                        os << def->getName() << "@" << def->getLoc();
                    else
                        os << "<block arg>";
                }
                os << ")\n";
                break;
            }
        }
    });
}

// Custom printer for DSEGraph
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DSEGraph &graph)
{
    DenseMap<const DSENode*, size_t> nodeIdx;
    for (auto [idx, node] : llvm::enumerate(graph.nodes))
        nodeIdx[node.get()] = idx;
    DenseMap<const DSEEdge*, size_t> edgeIdx;
    for (auto [idx, edge] : llvm::enumerate(graph.edges))
        edgeIdx[edge.get()] = idx;

    auto printPort = [&](DSENode* node, int64_t portIdx) {
        if (!node) {
            os << "<none>";
            return;
        }
        os << "node" << nodeIdx.lookup(node) << ":port" << portIdx;
    };

    os << "DSEGraph {\n  nodes:\n";
    for (auto [idx, node] : llvm::enumerate(graph.nodes)) {
        os << "    node" << idx << ": callee = \"" << node->callee.getSymName()
           << "\", ports = " << node->ports.size() << "\n";
        for (auto [portIdx, edge] : llvm::enumerate(node->ports)) {
            os << "      port" << portIdx << " -> ";
            if (edge)
                os << "edge" << edgeIdx.lookup(edge);
            else
                os << "<unconnected>";
            os << "\n";
        }
    }
    os << "  edges:\n";
    for (auto [idx, edge] : llvm::enumerate(graph.edges)) {
        os << "    edge" << idx << ": "
           << edge->variable.getVariable().getType() << "  producer=";
        printPort(edge->producer, edge->producerPortIdx);
        os << "  consumer=";
        printPort(edge->consumer, edge->consumerPortIdx);
        os << "\n";
    }
    os << "}\n";
    return os;
}

// Post-solve: each candidate loop's resolved factor and whether it was
// chosen to be pipelined. Only valid once GRBModel::optimize() has found an
// optimal solution.
void printSolution(
    llvm::raw_ostream &os,
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    for (auto &node : graph.nodes) {
        os << "  callee \"" << node->callee.getSymName()
           << "\": DSP=" << node->totalDSP.getIntValue()
           << "  BRAM=" << node->totalBRAM.getIntValue()
           << "  cycles=" << node->totalCycles.getIntValue() << "  {";
        bool firstPort = true;
        for (auto [argIdx, arg] :
             llvm::enumerate(node->callee.getArguments())) {
            auto dimFactors = node->arrayDimFactors.lookup(arg);
            if (dimFactors.empty()) continue;
            if (!firstPort) os << ", ";
            firstPort = false;
            os << "port" << argIdx << ":";
            for (auto [i, df] : llvm::enumerate(dimFactors)) {
                if (i) os << ",";
                os << " dim" << df.first << "=" << df.second.getIntValue();
            }
        }
        os << "}\n";
        for (CandidateLoop c :
             candidatesByFunc.lookup(node->callee.getOperation())) {
            LoopDSEInfo &info = node->loops[c.loop.getOperation()];
            os << "    " << c.loop.getLoc()
               << ": factor=" << info.unrollFactor.getIntValue()
               << "  pipelined=" << info.pipelined.getIntValue() << "\n";
        }
    }
}

// Post-solve: writes every DSE decision onto its corresponding op in the
// already-normalized IR as an attribute, so a separate later rewrite pass
// can pick them back up without re-running the solver. A loop's factor and
// pipeline decision go on the loop itself, a port's per-dimension factor
// goes on the callee func, and a local array's memory technology and
// per-dimension factor go on its VariableOp.
void attachSolutionAttributes(
    DSEGraph &graph,
    const DenseMap<Operation*, SmallVector<CandidateLoop>> &candidatesByFunc)
{
    if (graph.nodes.empty()) return;
    OpBuilder builder(graph.nodes.front()->callee.getContext());
    auto topFunc = cast<FuncOp>(graph.nodes.front()->call->getParentOp());
    topFunc->setAttr("dse.top", builder.getBoolAttr(true));
    SmallVector<Attribute> topFuncPortFactors(topFunc.getNumArguments());

    // One factor per dimension of "arrType", in order, so the position in
    // the array alone says which dimension it's for; dimensions with no
    // loop-driven factor default to 1. Ports are only ever touched through
    // stream reads/writes, so this is as far as their classification goes.
    auto factorsAttr = [&](emithls::ArrayType arrType,
                           ArrayRef<std::pair<int64_t, GRBValue>> dimFactors) {
        SmallVector<Attribute> factors(
            arrType.getShape().size(),
            builder.getI64IntegerAttr(1));
        for (auto &[dim, factor] : dimFactors)
            factors[dim] = builder.getI64IntegerAttr(factor.getIntValue());
        return builder.getArrayAttr(factors);
    };

    // A raw pointer's own factor needs to be in bits, not elements, since
    // it has no array shape of its own to say what an "element" even is:
    // scales every dimension of "base" (an array-typed sibling's own
    // factorsAttr) by "elemBits".
    auto scaledFactorsAttr = [&](Attribute base, int64_t elemBits) {
        SmallVector<Attribute> scaled;
        for (Attribute a : cast<ArrayAttr>(base))
            scaled.push_back(builder.getI64IntegerAttr(
                cast<IntegerAttr>(a).getInt() * elemBits));
        return builder.getArrayAttr(scaled);
    };

    for (auto &node : graph.nodes) {
        DenseSet<Operation*> candidateSet;
        for (CandidateLoop c :
             candidatesByFunc.lookup(node->callee.getOperation()))
            candidateSet.insert(c.loop);

        // So a later rewrite pass can tell an I/O bridge apart from an
        // ordinary node without re-deriving it from the argument types.
        bool isIO = isIOFunction(node->callee);
        if (isIO) node->callee->setAttr("dse.io_func", builder.getUnitAttr());

        // Same, but for a local array: unlike a port, its untouched
        // dimensions aren't all implicitly 1. classifyArrayDims tells apart
        // a Sequential dimension, factor 1, from a Full one, which is
        // mandatorily split into its whole static extent.
        auto localArrayFactorsAttr = [&](Value arrayVar,
                                         emithls::ArrayType arrType) {
            auto dims = classifyArrayDims(arrayVar, arrType, candidateSet);
            SmallVector<Attribute> factors;
            factors.reserve(dims.size());
            for (auto [d, dim] : llvm::enumerate(dims))
                factors.push_back(builder.getI64IntegerAttr(
                    dim.kind == DimPartitionKind::Full ? arrType.getShape()[d]
                                                       : 1));
            for (auto &[dim, factor] : node->arrayDimFactors.lookup(arrayVar))
                factors[dim] = builder.getI64IntegerAttr(factor.getIntValue());
            return builder.getArrayAttr(factors);
        };

        // Loop's chosen factor and pipeline decision -> the loop op.
        for (CandidateLoop c :
             candidatesByFunc.lookup(node->callee.getOperation())) {
            LoopDSEInfo &info = node->loops[c.loop.getOperation()];
            c.loop->setAttr(
                "dse.factor",
                builder.getI64IntegerAttr(info.unrollFactor.getIntValue()));
            c.loop->setAttr(
                "dse.pipelined",
                builder.getBoolAttr(info.pipelined.getIntValue() != 0));
        }

        // An IO bridge's scalar-port loop (see searchCandidateLoops) was
        // never made a candidate, since there's only ever one channel to
        // explore, but it's still the loop that actually touches the
        // port, so it still needs marking: always factor 1, always
        // pipelined.
        if (isIO)
            node->callee.walk([&](ForOp loop) {
                if (touchesPortDirectly(loop) != PortTouch::Scalar) return;
                loop->setAttr("dse.factor", builder.getI64IntegerAttr(1));
                loop->setAttr("dse.pipelined", builder.getBoolAttr(true));
            });

        // Every port's per-dimension factor -> one ArrayAttr on the callee
        // func, one entry per argument in order. A non-array argument has
        // no factor of its own, so it just copies whichever array-typed
        // argument's, scaled to bits instead of elements if it's a raw
        // pointer, since a pointer's own access width is what actually
        // matters for it, not an element count it has no shape to define.
        SmallVector<Attribute> portFactors(node->callee.getNumArguments());
        Attribute copyFrom;
        int64_t copyFromElemBits = 0;
        for (auto [argIdx, arg] :
             llvm::enumerate(node->callee.getArguments())) {
            auto arrType = dyn_cast<emithls::ArrayType>(arg.getType());
            if (!arrType) continue;
            portFactors[argIdx] =
                factorsAttr(arrType, node->arrayDimFactors.lookup(arg));
            if (!copyFrom) {
                copyFrom = portFactors[argIdx];
                // A port array's element is a channel, not a scalar; the
                // bits that actually cross it are its stream's own element
                // type.
                Type elemType = arrType.getElementType();
                if (auto streamType = dyn_cast<StreamType>(elemType))
                    elemType = streamType.getElementType();
                copyFromElemBits = getScalarBitWidth(elemType);
            }
        }
        if (copyFrom) {
            for (auto [argIdx, arg] :
                 llvm::enumerate(node->callee.getArguments())) {
                if (portFactors[argIdx]) continue;
                portFactors[argIdx] =
                    isa<PointerType>(arg.getType())
                        ? scaledFactorsAttr(copyFrom, copyFromElemBits)
                        : copyFrom;
            }
            node->callee->setAttr(
                "dse.factors",
                builder.getArrayAttr(portFactors));
        }

        // Every loop that directly touches a port is marked for the later
        // rewrite that resizes the port: an I/O bridge's loop just gets
        // "dse.io"; an ordinary loop gets "dse.port", the resolved max
        // factor across the port dimensions it touches, since it may need
        // to run wider than its own chosen unroll factor to keep pace.
        if (isIO) {
            node->callee.walk([&](ForOp loop) {
                bool touchesAnyPort =
                    llvm::any_of(loop.getBody().front(), [](Operation &op) {
                        Value port =
                            llvm::TypeSwitch<Operation*, Value>(&op)
                                .Case<StreamReadOp, StreamWriteOp>(
                                    [](auto o) { return o.getStream(); })
                                .Case<ArrayPointerReadOp, ArrayPointerWriteOp>(
                                    [](auto o) { return o.getPointer(); })
                                .Default(Value());
                        return port && isa<BlockArgument>(port);
                    });
                if (touchesAnyPort)
                    loop->setAttr("dse.io", builder.getUnitAttr());
            });
        } else {
            node->callee.walk([&](ForOp loop) {
                auto touched = getTouchedPortDims(loop);
                if (touched.empty()) return;
                int64_t factor = 1;
                for (auto &[port, dim] : touched) {
                    int64_t argIdx = cast<BlockArgument>(port).getArgNumber();
                    if (!portFactors[argIdx]) continue;
                    int64_t dimFactor =
                        cast<IntegerAttr>(
                            cast<ArrayAttr>(portFactors[argIdx])[dim])
                            .getInt();
                    if (dimFactor > factor) factor = dimFactor;
                }
                loop->setAttr("dse.port", builder.getI64IntegerAttr(factor));
            });
        }

        // Any argument of the top function passed straight through as
        // this call's argument, an IO bridge's pointer tying it back to
        // the outside world, inherits that call argument's own factors;
        // collected into "topFuncPortFactors" for one combined attribute
        // once every node has been visited.
        for (auto [argIdx, callArg] :
             llvm::enumerate(node->call.getArgOperands())) {
            auto topFuncArg = dyn_cast<BlockArgument>(callArg);
            if (!topFuncArg || !portFactors[argIdx]
                || topFuncArg.getOwner() != &topFunc.getBody().front())
                continue;
            topFuncPortFactors[topFuncArg.getArgNumber()] = portFactors[argIdx];
        }

        // Local array's memory technology and per-dimension factor -> its
        // VariableOp.
        node->callee.walk([&](VariableOp varOp) {
            auto arrType =
                dyn_cast<emithls::ArrayType>(varOp.getVariable().getType());
            if (!arrType) return;
            varOp->setAttr(
                "dse.mem",
                builder.getStringAttr(isLUTRAM(varOp) ? "LUTRAM" : "BRAM"));
            varOp->setAttr(
                "dse.factors",
                localArrayFactorsAttr(varOp.getVariable(), arrType));
        });
    }

    // One combined ArrayAttr on the top function itself, one entry per
    // argument in order, same as every callee's own "dse.factors".
    if (llvm::all_of(topFuncPortFactors, [](Attribute a) { return a; }))
        topFunc->setAttr(
            "dse.factors",
            builder.getArrayAttr(topFuncPortFactors));

    // Every edge's own per-dimension factor -> its VariableOp in the top
    // function, read off whichever side of the edge is connected: a
    // producer and consumer sharing an edge were already tied to the same
    // recorded factor when their port was set up.
    for (auto &edge : graph.edges) {
        DSENode* side = edge->producer ? edge->producer : edge->consumer;
        int64_t portIdx =
            edge->producer ? edge->producerPortIdx : edge->consumerPortIdx;
        auto arrType = dyn_cast<emithls::ArrayType>(
            edge->variable.getVariable().getType());
        if (side && arrType)
            edge->variable->setAttr(
                "dse.factors",
                factorsAttr(
                    arrType,
                    side->arrayDimFactors.lookup(
                        side->callee.getArgument(portIdx))));
    }

    // Every producer/consumer edge's FIFO depth (see computeBufferDepths)
    // -> its own VariableOp in the top function.
    for (auto &edge : graph.edges) {
        if (!edge->producer || !edge->consumer) continue;
        int64_t depth = edge->bufferDepth.getIntValue();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "  " << edge->variable.getLoc() << ": fifo depth=" << depth);
        edge->variable->setAttr("dse.fifo", builder.getBoolAttr(true));
        edge->variable->setAttr("dse.depth", builder.getI64IntegerAttr(depth));
    }
}

} // namespace

namespace {
struct EmitHLSPragmaDSEPass
        : public emithls::impl::EmitHLSPragmaDSEBase<EmitHLSPragmaDSEPass> {
    EmitHLSPragmaDSEPass(
        int64_t availableBRAM = 288,
        int64_t availableDSP = 1248)
    {
        this->availableBRAM = availableBRAM;
        this->availableDSP = availableDSP;
    }

    void runOnOperation() override;
};
} // namespace

void EmitHLSPragmaDSEPass::runOnOperation()
{
    auto availableDSPBudget = availableDSP / 10 * 8;
    LAKSA_DEBUG(
        llvm::dbgs() << "available BRAM: " << availableBRAM
                     << ", available DSP: " << availableDSPBudget);

    // Create GRB model
    GRBEnv env = GRBEnv(true);
    // save log into file
    env.set("LogFile", "dse.log");
    // no output to console to keep mlir clean
    env.set("LogToConsole", "0");
    env.start();
    GRBModel model = GRBModel(env);

    FuncOp topFunc;
    getOperation()->walk([&](FuncOp funcOp) {
        if (!isTopFunction(funcOp)) return WalkResult::advance();
        topFunc = funcOp;
        return WalkResult::interrupt();
    });
    if (!topFunc) {
        LAKSA_DEBUG(llvm::dbgs() << "no top function found\n");
        signalPassFailure();
        return;
    }

    getOperation()->walk([&](FuncOp funcOp) {
        if (isTopFunction(funcOp)) return;
        normalizeLinearIslands(funcOp.getBody().front());
    });

    DenseMap<Operation*, SmallVector<CandidateLoop>> candidatesByFunc;
    getOperation()->walk([&](FuncOp funcOp) {
        if (isTopFunction(funcOp)) return;
        SmallVector<CandidateLoop> candidates;
        searchCandidateLoops(
            funcOp.getBody().front(),
            isIOFunction(funcOp),
            candidates);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Analyzing callee \"" << funcOp.getSymName() << "\":");
        LAKSA_DEBUG(llvm::dbgs() << "  loop candidates:");
        LAKSA_DEBUG(printCandidateLoops(llvm::dbgs(), candidates));
        LAKSA_DEBUG(llvm::dbgs() << "  array memory type:");
        LAKSA_DEBUG(printArrayMemories(llvm::dbgs(), funcOp, candidates));
        LAKSA_DEBUG(llvm::dbgs() << "  per-iteration DSP usage:");
        LAKSA_DEBUG(printIterationDSP(llvm::dbgs(), candidates));
        candidatesByFunc[funcOp.getOperation()] = std::move(candidates);
    });

    auto graph = buildGraph(topFunc);
    LAKSA_DEBUG(llvm::dbgs() << "Found top function as graph " << *graph);

    LAKSA_DEBUG(llvm::dbgs() << "Setting up loop split-factor variables:");
    setupLoopSplitFactors(model, *graph, candidatesByFunc);

    LAKSA_DEBUG(llvm::dbgs() << "Setting up memory dimension factors:");
    setupArrayDimFactors(model, *graph, candidatesByFunc);

    LAKSA_DEBUG(llvm::dbgs() << "Setting up BRAM budget:");
    GRBValue totalBRAM =
        addBRAMBudgetConstraint(model, *graph, candidatesByFunc, availableBRAM);

    LAKSA_DEBUG(llvm::dbgs() << "Setting up pipeline decisions:");
    setupPipelineConstraints(model, *graph, candidatesByFunc);

    LAKSA_DEBUG(llvm::dbgs() << "Setting up DSP budget:");
    GRBValue totalDSP = addDSPBudgetConstraint(
        model,
        *graph,
        candidatesByFunc,
        availableDSPBudget);

    LAKSA_DEBUG(llvm::dbgs() << "Computing total cycles:");
    GRBValue totalCycles =
        computeGraphTotalCycles(model, *graph, candidatesByFunc);

    LAKSA_DEBUG(llvm::dbgs() << "Computing FIFO buffer depths:");
    computeBufferDepths(model, *graph);

    LAKSA_DEBUG(llvm::dbgs() << "Solving (minimize critical-path cycles):");
    model.setObjective(totalCycles.getExpr(), GRB_MINIMIZE);
    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    LAKSA_DEBUG(llvm::dbgs() << "  status = " << status);
    if (status == GRB_OPTIMAL) {
        LAKSA_DEBUG(llvm::dbgs() << "  total DSP = " << totalDSP.getIntValue());
        LAKSA_DEBUG(
            llvm::dbgs() << "  total BRAM = " << totalBRAM.getIntValue());
        LAKSA_DEBUG(
            llvm::dbgs() << "  total cycles = " << totalCycles.getIntValue());
        LAKSA_DEBUG(printSolution(llvm::dbgs(), *graph, candidatesByFunc));
        attachSolutionAttributes(*graph, candidatesByFunc);
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSPragmaDSEPass()
{ return std::make_unique<EmitHLSPragmaDSEPass>(); }

std::unique_ptr<Pass> mlir::emithls::createEmitHLSPragmaDSEPass(
    int64_t availableBRAM,
    int64_t availableDSP)
{ return std::make_unique<EmitHLSPragmaDSEPass>(availableBRAM, availableDSP); }
