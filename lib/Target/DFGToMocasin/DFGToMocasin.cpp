/// Implementation of translation DFGToMocasin
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "laksa-mlir/Target/DFGToMocasin/MocasinEmitter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::dfg;

namespace {

/// Mocasin's native YAML application format: a `graph` section describing
/// process ports and channel topology, and an `execution` section describing
/// per-actor timing profiles, per-port firing rates, and channel initial-token
/// counts. The format currently models SDF applications only.

/// A process's named input/output interfaces. `dfg` has no persistent named
/// port concept of its own (port names in the textual syntax, e.g. `%out`,
/// are parser-local and never stored on the operation -- see
/// `ProcessOp::getAsmBlockArgumentNames`), so port names are synthesized
/// positionally as `in0`, `in1`, ..., `out0`, `out1`, ..., the same
/// convention the pretty-printer already uses.
struct MocasinPortList {
    std::vector<std::string> values;

    auto begin() { return values.begin(); }
    auto end() { return values.end(); }
    std::string &operator[](size_t index) { return values[index]; }
    void push_back(std::string value) { values.push_back(std::move(value)); }
};

struct MocasinPorts {
    MocasinPortList in;
    MocasinPortList out;
};

/// A process entry in `graph.processes`: just its `ports` today, but a
/// distinct type (rather than `MocasinPorts` directly) since the format
/// nests ports one level under a `ports:` key.
struct MocasinProcessDef {
    MocasinPorts ports;
};

/// `{process: ..., port: ...}`, printed in flow style to match the sample
/// format (`src: {process: src, port: output}`).
struct MocasinEndpoint {
    std::string process;
    std::string port;
};

struct MocasinChannelDef {
    MocasinEndpoint src;
    MocasinEndpoint dst;
    uint64_t tokenSize = 0;
};

struct MocasinRates {
    std::map<std::string, int64_t> values;

    int64_t &operator[](const std::string &port) { return values[port]; }
};

struct MocasinGraphSection {
    std::map<std::string, MocasinProcessDef> processes;
    std::map<std::string, MocasinChannelDef> channels;
};

/// Per-processor-type cycle count for one execution profile. The `dfg`
/// dialect carries no execution-time/profiling information at all, so every
/// profile emitted here is a placeholder -- see the note on
/// `translateDFGToMocasinYAML` below.
struct CyclesEntry {
    uint64_t cycles = 0;
};
using CyclesByProcessor = std::map<std::string, CyclesEntry>;

struct MocasinInstance {
    std::string model = "static";
    std::string profile;
    MocasinRates rates;
};

struct ChannelExec {
    uint64_t initialTokens = 0;
};

struct MocasinExecutionProcesses {
    std::map<std::string, CyclesByProcessor> profiles;
    std::map<std::string, MocasinInstance> instances;
};

struct MocasinExecutionSection {
    MocasinExecutionProcesses processes;
    std::map<std::string, ChannelExec> channels;
};

struct MocasinDocument {
    std::string name;
    MocasinGraphSection graph;
    MocasinExecutionSection execution;
};

/// Internal-only: where a channel end (a Value that is a node's input or
/// output port operand) resolves to once nested regions are flattened.
struct PortRef {
    std::string leafName;
    std::string portName;
};

} // namespace

LLVM_YAML_IS_STRING_MAP(MocasinProcessDef)
LLVM_YAML_IS_STRING_MAP(MocasinChannelDef)
LLVM_YAML_IS_STRING_MAP(CyclesEntry)
LLVM_YAML_IS_STRING_MAP(CyclesByProcessor)
LLVM_YAML_IS_STRING_MAP(MocasinInstance)
LLVM_YAML_IS_STRING_MAP(ChannelExec)

namespace llvm::yaml {

template<>
struct SequenceTraits<MocasinPortList> {
    static const bool flow = true;

    static size_t size(IO &, MocasinPortList &ports)
    { return ports.values.size(); }

    static std::string &element(IO &, MocasinPortList &ports, size_t index)
    {
        if (index >= ports.values.size()) ports.values.resize(index + 1);
        return ports.values[index];
    }
};

template<>
struct MappingTraits<MocasinPorts> {
    static void mapping(IO &io, MocasinPorts &p)
    {
        io.mapOptional("in", p.in);
        io.mapOptional("out", p.out);
    }
};

template<>
struct MappingTraits<MocasinProcessDef> {
    static void mapping(IO &io, MocasinProcessDef &p)
    { io.mapRequired("ports", p.ports); }
};

template<>
struct MappingTraits<MocasinEndpoint> {
    static const bool flow = true;
    static void mapping(IO &io, MocasinEndpoint &e)
    {
        io.mapRequired("process", e.process);
        io.mapRequired("port", e.port);
    }
};

template<>
struct MappingTraits<MocasinRates> {
    static const bool flow = true;

    static void mapping(IO &io, MocasinRates &rates)
    {
        if (io.outputting()) {
            for (auto &[port, rate] : rates.values) io.mapRequired(port, rate);
            return;
        }

        for (StringRef port : io.keys())
            io.mapRequired(port, rates.values[port.str()]);
    }
};

template<>
struct MappingTraits<MocasinChannelDef> {
    static void mapping(IO &io, MocasinChannelDef &c)
    {
        io.mapRequired("src", c.src);
        io.mapRequired("dst", c.dst);
        io.mapRequired("token_size", c.tokenSize);
    }
};

template<>
struct MappingTraits<MocasinGraphSection> {
    static void mapping(IO &io, MocasinGraphSection &g)
    {
        io.mapRequired("processes", g.processes);
        io.mapRequired("channels", g.channels);
    }
};

template<>
struct MappingTraits<CyclesEntry> {
    static void mapping(IO &io, CyclesEntry &c)
    { io.mapRequired("cycles", c.cycles); }
};

template<>
struct MappingTraits<MocasinInstance> {
    static void mapping(IO &io, MocasinInstance &i)
    {
        io.mapRequired("model", i.model);
        io.mapRequired("profile", i.profile);
        io.mapRequired("rates", i.rates);
    }
};

template<>
struct MappingTraits<ChannelExec> {
    static void mapping(IO &io, ChannelExec &c)
    { io.mapRequired("initial_tokens", c.initialTokens); }
};

template<>
struct MappingTraits<MocasinExecutionProcesses> {
    static void mapping(IO &io, MocasinExecutionProcesses &p)
    {
        io.mapRequired("profiles", p.profiles);
        io.mapRequired("instances", p.instances);
    }
};

template<>
struct MappingTraits<MocasinExecutionSection> {
    static void mapping(IO &io, MocasinExecutionSection &e)
    {
        io.mapRequired("processes", e.processes);
        io.mapRequired("channels", e.channels);
    }
};

template<>
struct MappingTraits<MocasinDocument> {
    static void mapping(IO &io, MocasinDocument &d)
    {
        io.mapRequired("name", d.name);
        io.mapRequired("graph", d.graph);
        io.mapRequired("execution", d.execution);
    }
};

} // namespace llvm::yaml

namespace {

/// Flattens nested `dfg.region`/`dfg.embed` graphs into the flat
/// process/channel model Mocasin expects.
///
/// Two facts about the DFG IR make this tractable (see DFGOps.td):
///   - `RegionOp` is IsolatedFromAbove, so every `dfg.channel` is defined and
///     consumed entirely within one region; channels never cross a region
///     boundary.
///   - `dfg.embed`'s operand list lines up positionally with the embedded
///     region's own ports (`EmbedOp::getInputPort(i)` is the same logical
///     channel end as `RegionOp::getInputPort(i)` of the region it embeds,
///     likewise for outputs).
///
/// This lets a single Value -> PortRef map, threaded through the recursive
/// walk, resolve every connection: instantiate/embed nodes are visited
/// first (populating the map for every operand they use, including a
/// region's own block arguments), then channels are resolved by lookup.
///
/// Top-level region ports are materialized in the emitted document as
/// synthetic `<region>-inN-source` and `<region>-outN-sink` processes. Their
/// single port has rate 1, and the connecting channel starts empty; the source
/// DFG IR is not modified.
class MocasinEmitter {
public:
    explicit MocasinEmitter(DataLayout &dataLayout) : dataLayout(dataLayout) {}

    LogicalResult
    emitModule(ModuleOp module, std::vector<MocasinDocument> &docs);

private:
    std::string uniqueName(StringRef base);
    PortRef
    addBoundaryProcess(StringRef baseName, bool isSource, MocasinDocument &doc);
    void addChannel(
        Type tokenType,
        const PortRef &producer,
        const PortRef &consumer,
        MocasinDocument &doc);
    void flattenRegion(
        RegionOp region,
        const std::string &prefix,
        MocasinDocument &doc,
        llvm::DenseMap<Value, PortRef> &portOwner);

    std::map<std::string, unsigned> nameCounters;
    unsigned channelCounter = 0;
    DataLayout &dataLayout;
};

} // namespace

std::string MocasinEmitter::uniqueName(StringRef base)
{
    std::string name = base.str();
    unsigned &counter = nameCounters[name];
    if (counter == 0) {
        ++counter;
        return name;
    }
    return name + "_" + std::to_string(counter++);
}

PortRef MocasinEmitter::addBoundaryProcess(
    StringRef baseName,
    bool isSource,
    MocasinDocument &doc)
{
    std::string processName = uniqueName(baseName);
    std::string portName = isSource ? "out0" : "in0";

    MocasinPorts ports;
    if (isSource)
        ports.out.push_back(portName);
    else
        ports.in.push_back(portName);
    doc.graph.processes.emplace(
        processName,
        MocasinProcessDef{std::move(ports)});

    MocasinInstance instance;
    instance.profile = processName;
    instance.rates[portName] = 1;
    doc.execution.processes.instances.emplace(processName, std::move(instance));
    doc.execution.processes.profiles.emplace(
        processName,
        CyclesByProcessor{
            {"UNKNOWN", CyclesEntry{0}}
    });

    return {processName, portName};
}

void MocasinEmitter::addChannel(
    Type tokenType,
    const PortRef &producer,
    const PortRef &consumer,
    MocasinDocument &doc)
{
    std::string chanName = "ch" + std::to_string(channelCounter++);

    MocasinChannelDef chanDef;
    chanDef.src = {producer.leafName, producer.portName};
    chanDef.dst = {consumer.leafName, consumer.portName};
    chanDef.tokenSize = dataLayout.getTypeSize(tokenType).getFixedValue();
    doc.graph.channels.emplace(chanName, std::move(chanDef));

    // DFG channels and synthesized graph-boundary channels both start empty.
    doc.execution.channels.emplace(chanName, ChannelExec{0});
}

void MocasinEmitter::flattenRegion(
    RegionOp region,
    const std::string &prefix,
    MocasinDocument &doc,
    llvm::DenseMap<Value, PortRef> &portOwner)
{
    // Leaf actor instances: each becomes one flat Mocasin process, with
    // ports named positionally and firing rates read off `multiplicity`.
    for (auto* nodeOp : region.getGraphNodes()) {
        auto inst = cast<InstantiateOp>(nodeOp);
        std::string leafName = uniqueName(prefix + inst.getNodeName());
        std::string actorName = inst.getNodeName();
        unsigned numIn = inst.getNumInputPorts();
        unsigned numOut = inst.getNumOutputPorts();

        MocasinPorts ports;
        for (unsigned i = 0; i < numIn; ++i)
            ports.in.push_back("in" + std::to_string(i));
        for (unsigned i = 0; i < numOut; ++i)
            ports.out.push_back("out" + std::to_string(i));

        // Firing rates: `dfg.process` may declare a `multiplicity` (empty
        // means all-ones); `dfg.operator` has no such attribute and is
        // always single-rate (its `output` op writes each result exactly
        // once per invocation).
        ArrayRef<int64_t> multiplicity;
        if (auto processOp =
                dyn_cast_or_null<ProcessOp>(inst.getInstantiatedOperation()))
            multiplicity = processOp.getMultiplicity();

        MocasinInstance instance;
        instance.profile = actorName;
        for (unsigned i = 0; i < numIn; ++i)
            instance.rates[ports.in[i]] =
                i < multiplicity.size() ? multiplicity[i] : 1;
        for (unsigned i = 0; i < numOut; ++i)
            instance.rates[ports.out[i]] =
                (numIn + i) < multiplicity.size() ? multiplicity[numIn + i] : 1;

        for (unsigned i = 0; i < numIn; ++i)
            portOwner[inst.getInputPort(i)] = {leafName, ports.in[i]};
        for (unsigned i = 0; i < numOut; ++i)
            portOwner[inst.getOutputPort(i)] = {leafName, ports.out[i]};

        doc.graph.processes.emplace(
            leafName,
            MocasinProcessDef{std::move(ports)});
        doc.execution.processes.instances.emplace(
            leafName,
            std::move(instance));
        // One placeholder profile per distinct actor definition; multiple
        // instances of the same actor share it (`emplace` is a no-op if the
        // key already exists).
        doc.execution.processes.profiles.emplace(
            actorName,
            CyclesByProcessor{
                {"UNKNOWN", CyclesEntry{0}}
        });
    }

    // Embedded subgraphs: recurse and splice the child's leaves/channels
    // into the same flat graph, then bridge the embed call site's operands
    // to the child region's own ports so channels on either side resolve.
    for (auto* subOp : region.getGraphSubGs()) {
        auto embed = cast<EmbedOp>(subOp);
        auto child = cast<RegionOp>(embed.getEmbeddedOperation());
        flattenRegion(
            child,
            prefix + embed.getNodeName() + ".",
            doc,
            portOwner);

        for (unsigned i = 0; i < embed.getNumInputPorts(); ++i)
            portOwner[embed.getInputPort(i)] = portOwner[child.getInputPort(i)];
        for (unsigned i = 0; i < embed.getNumOutputPorts(); ++i)
            portOwner[embed.getOutputPort(i)] =
                portOwner[child.getOutputPort(i)];
    }

    // Channels: now that every node operand in this region has a resolved
    // leaf, each channel is just a lookup of its two ends.
    for (auto* edgeOp : region.getGraphEdges()) {
        auto channel = cast<ChannelOp>(edgeOp);
        PortRef producer = portOwner.lookup(channel.getInputPort());
        PortRef consumer = portOwner.lookup(channel.getOutputPort());

        addChannel(channel.getTokenType(), producer, consumer, doc);
    }
}

LogicalResult
MocasinEmitter::emitModule(ModuleOp module, std::vector<MocasinDocument> &docs)
{
    unsigned rootCount = 0;
    for (auto &opi : module.getBodyRegion().front()) {
        auto regionOp = dyn_cast<RegionOp>(opi);
        if (!regionOp || !regionOp->hasAttr(laksa::kRootAttrName)) continue;
        if (++rootCount > 1)
            return module.emitError(
                "cannot export to Mocasin: expected exactly one operation "
                "marked 'laksa.root'");

        MocasinDocument doc;
        doc.name = regionOp.getGraphName();
        llvm::DenseMap<Value, PortRef> portOwner;
        nameCounters.clear();
        channelCounter = 0;
        flattenRegion(regionOp, /*prefix=*/"", doc, portOwner);

        // Mocasin has no external graph interfaces. Represent every top-level
        // input and output port as its own synthetic environment process and
        // connect it to the leaf actor resolved by flattenRegion. These
        // processes exist only in the emitted YAML; the source DFG is not
        // modified.
        for (unsigned i = 0; i < regionOp.getNumInputPorts(); ++i) {
            Value regionPort = regionOp.getInputPort(i);
            std::string portName = "in" + std::to_string(i);
            if (!regionPort.hasOneUse())
                return regionOp.emitError()
                       << "cannot export to Mocasin: top-level region '"
                       << regionOp.getGraphName() << "' input port '"
                       << portName
                       << "' must have exactly one consumer inside the region";

            auto consumerIt = portOwner.find(regionPort);
            if (consumerIt == portOwner.end())
                return regionOp.emitError()
                       << "cannot resolve Mocasin consumer for top-level input "
                          "port '"
                       << portName << "'";

            PortRef source = addBoundaryProcess(
                regionOp.getGraphName() + "-" + portName + "-source",
                /*isSource=*/true,
                doc);
            addChannel(
                cast<OutputType>(regionPort.getType()).getElementType(),
                source,
                consumerIt->second,
                doc);
        }

        for (unsigned i = 0; i < regionOp.getNumOutputPorts(); ++i) {
            Value regionPort = regionOp.getOutputPort(i);
            std::string portName = "out" + std::to_string(i);
            if (!regionPort.hasOneUse())
                return regionOp.emitError()
                       << "cannot export to Mocasin: top-level region '"
                       << regionOp.getGraphName() << "' output port '"
                       << portName
                       << "' must have exactly one producer inside the region";

            auto producerIt = portOwner.find(regionPort);
            if (producerIt == portOwner.end())
                return regionOp.emitError()
                       << "cannot resolve Mocasin producer for top-level "
                          "output port '"
                       << portName << "'";

            PortRef sink = addBoundaryProcess(
                regionOp.getGraphName() + "-" + portName + "-sink",
                /*isSource=*/false,
                doc);
            addChannel(
                cast<InputType>(regionPort.getType()).getElementType(),
                producerIt->second,
                sink,
                doc);
        }

        docs.push_back(std::move(doc));
    }
    if (rootCount == 0)
        return module.emitError(
            "cannot export to Mocasin: expected exactly one operation marked "
            "'laksa.root'");
    return success();
}

/// Translates a `dfg` graph to Mocasin's native YAML application format.
///
/// Only the `graph` section (process ports, channel topology, token sizes)
/// and the `rates`/`initial_tokens` fields of `execution` are derived from
/// real IR data. The `execution.processes.profiles` cycle counts are always
/// a placeholder (`UNKNOWN: {cycles: 0}`): the `dfg` dialect has no
/// execution-time/profiling model at all, so real per-processor-type cycle
/// counts must be filled in downstream once that data exists.
LogicalResult dfg::translateDFGToMocasinYAML(Operation* op, raw_ostream &os)
{
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) return op->emitError("expected builtin.module op");

    DataLayout dataLayout(module);
    MocasinEmitter emitter(dataLayout);
    std::vector<MocasinDocument> docs;
    if (failed(emitter.emitModule(module, docs))) return failure();

    llvm::yaml::Output yout(os);
    for (auto &doc : docs) yout << doc;
    return success();
}
