/// Implements the DFG dialect ops.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"

#include "laksa-mlir/Dialect/DFG/DFGEnums.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGBase.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGTypes.h"
#include "laksa-mlir/Dialect/DFG/Interfaces/NodeInterface.h"
#include "laksa-mlir/IR/LaksaAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <ranges>

using namespace mlir;
using namespace mlir::dfg;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.cpp.inc"

//===----------------------------------------------------------------------===//

// For multiple variadic attributes
constexpr char kOperandSegmentSizesAttr[] = "operandSegmentSizes";

/// @brief Parses a function argument list for inputs or outputs.
/// The type of each argument must be written explicitly as the full channel
/// type, e.g. `!dfg.output<i32>` or `!dfg.input<2x3xi32>`.
/// @param parser The currently used parser
/// @param arguments A list of arguments to parse
/// @return A parse result indicating success or failure to parse.
static ParseResult parseChannelArgumentList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::Argument> &arguments)
{
    return parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
            OpAsmParser::Argument argument;
            if (parser.parseArgument(
                    argument,
                    /*allowType=*/true,
                    /*allowAttrs=*/false))
                return failure();

            arguments.push_back(argument);
            return success();
        });
}

//===----------------------------------------------------------------------===//
// DFGDialect Node Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

/// @brief Names the block arguments of the `ProcessOp` body region.
///
/// The entry block carries a flat argument list that holds all input port
/// arguments followed by all output port arguments.  This callback assigns
/// human-readable SSA names so that printed IR uses `%in0`, `%in1`, … for
/// input ports and `%out0`, `%out1`, … for output ports, making the
/// argument list easier to read than the default `%arg0`, `%arg1`, … scheme.
/// The split point between the two groups is determined by the number of
/// inputs reported by the stored `FunctionType`.
void ProcessOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setNameFn)
{
    unsigned numInputs = getFunctionType().getNumInputs();
    for (unsigned i = 0, e = region.getNumArguments(); i < e; ++i)
        if (i < numInputs)
            setNameFn(region.getArgument(i), ("in" + std::to_string(i)));
        else
            setNameFn(
                region.getArgument(i),
                ("out" + std::to_string(i - numInputs)));
}

/// @brief Builds a ProcessOp with the given name, port types, and body.
///
/// Input and output types in @p functionType are automatically wrapped in
/// the corresponding DFG channel types if they are not already:
/// - function inputs  -> `dfg::OutputType` (incoming channel ends)
/// - function results -> `dfg::InputType`  (outgoing channel ends)
///
/// The region is always created. If @p bodyBuilder is provided it is
/// called immediately with the builder positioned at the entry block, so
/// callers can populate the body inline.
///
/// @param name         Symbol name of the process (must be unique in the
/// module).
/// @param functionType Port signature before DFG-type wrapping.
/// @param multiplicity Per-port firing multiplicity; empty means all-ones.
/// @param bodyBuilder  Optional callback to fill the entry block; may be null.
void ProcessOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType functionType,
    ArrayRef<int64_t> multiplicity,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    // Get the function type with dfg types from base
    SmallVector<Type> inPortTypes, outPortTypes;
    for (auto inTy : functionType.getInputs())
        inPortTypes.push_back(
            isa<OutputType>(inTy)   ? inTy
            : isa<ShapedType>(inTy) ? OutputType::get(cast<ShapedType>(inTy))
                                    : OutputType::get(inTy));
    for (auto outTy : functionType.getResults())
        outPortTypes.push_back(
            isa<InputType>(outTy)    ? outTy
            : isa<ShapedType>(outTy) ? InputType::get(cast<ShapedType>(outTy))
                                     : InputType::get(outTy));
    auto dfgFunctionType =
        FunctionType::get(builder.getContext(), inPortTypes, outPortTypes);
    state.addAttribute(
        ProcessOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(dfgFunctionType));

    if (!multiplicity.empty())
        state.addAttribute(
            "multiplicity",
            DenseI64ArrayAttr::get(state.getContext(), multiplicity));

    OpBuilder::InsertionGuard guard(builder);
    Region* region = state.addRegion();
    auto regionBlock = builder.createBlock(region);

    // Create block arguments from the new function type
    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        dfgFunctionType.getInputs().begin(),
        dfgFunctionType.getInputs().end());
    blockArgTypes.append(
        dfgFunctionType.getResults().begin(),
        dfgFunctionType.getResults().end());
    regionBlock->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));

    if (bodyBuilder)
        bodyBuilder(builder, state.location, regionBlock->getArguments());
}

/// @brief Parses a `dfg::ProcessOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.process` @name
///     (`inputs` `(` arg-list `)`)?
///     (`outputs` `(` arg-list `)`)?
///     (`attributes` attr-dict)?
///     region?
/// @endcode
///
/// The `inputs` and `outputs` keyword-prefixed argument lists are parsed
/// separately so the two groups can be distinguished when reconstructing the
/// `FunctionType` attribute: input types become the function's argument types
/// and output types become its result types.  Both argument lists are then
/// merged into a single flat list that becomes the entry block arguments of the
/// attached body region.
///
/// An absent region indicates an external (declaration-only) process; a
/// present but empty region is rejected as ill-formed.  The `sym_name` and
/// `function_type` attributes are populated directly during parsing and are
/// therefore not expected to appear in the optional attribute dictionary.
ParseResult ProcessOp::parse(OpAsmParser &parser, OperationState &state)
{
    auto &builder = parser.getBuilder();

    // Parse the process' name
    StringAttr nameAttr;
    if (parser.parseSymbolName(
            nameAttr,
            getSymNameAttrName(state.name),
            state.attributes))
        return failure();

    // Parse the signature of the process
    SmallVector<OpAsmParser::Argument> inputValues, outputValues;
    SMLoc signatureLocation = parser.getCurrentLocation();

    // Parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList(parser, inputValues)) return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList(parser, outputValues)) return failure();
    }

    SmallVector<Type> inputTypes, outputTypes;
    inputTypes.reserve(inputValues.size());
    outputTypes.reserve(outputValues.size());

    for (auto &input : inputValues) inputTypes.push_back(input.type);
    for (auto &output : outputValues) outputTypes.push_back(output.type);
    Type funcType = builder.getFunctionType(inputTypes, outputTypes);

    if (!funcType) {
        return parser.emitError(signatureLocation)
               << "Failed to construct process type";
    }

    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(funcType));

    // Merge both argument lists for the block arguments
    inputValues.append(outputValues);

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(state.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // Parse the attached region, if any
    auto* body = state.addRegion();
    SMLoc loc = parser.getCurrentLocation();
    OptionalParseResult parseResult = parser.parseOptionalRegion(
        *body,
        inputValues,
        /*enableNameShadowing=*/false);

    if (parseResult.has_value()) {
        if (failed(*parseResult)) return failure();
        if (body->empty())
            return parser.emitError(loc, "expected non-empty process body");
    }

    return success();
}

/// @brief Prints a `dfg::ProcessOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// The symbol name is printed first, followed by the `inputs(...)` clause if
/// the process has any input ports, then the `outputs(...)` clause if it has
/// any output ports.  Within each clause, every port is printed as
/// `%name : type`.  For concrete (non-external) processes the SSA value names
/// come from the body region's block arguments; for external
/// (declaration-only) processes synthetic names `%arg0`, `%arg1`, … are
/// emitted instead since there is no region to source names from.
///
/// The `sym_name` and `function_type` attributes are always elided from the
/// optional attribute dictionary because they are already encoded in the
/// symbol name and the `inputs`/`outputs` clauses respectively.  An empty
/// `multiplicity` array attribute is also elided to keep the output clean.
/// The body region is only printed for concrete processes; external processes
/// have no region and are therefore printed as bare declarations.
void ProcessOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = getBody();
    bool isExternProcess = isExternal();

    // Print the operation and function name
    auto name = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();

    p << ' ';
    p.printSymbolName(name);

    auto funcType = getFunctionType();
    auto inputTypes = funcType.getInputs();
    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); ++i) {
            if (i > 0) p << ", ";

            if (isExternProcess)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << ": " << inputTypes[i];
        }
        p << ")";
    }

    auto outputTypes = funcType.getResults();
    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned numInputs = inputTypes.size();
        for (unsigned i = 0; i < outputTypes.size(); ++i) {
            if (i > 0) p << ", ";

            if (isExternProcess)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i + numInputs));
            p << ": " << outputTypes[i];
        }
        p << ")";
    }

    SmallVector<StringRef, 3> elidedAttrs = {
        getFunctionTypeAttrName(),
        getSymNameAttrName()};
    if (auto multiplicityAttr =
            cast<DenseI64ArrayAttr>(op->getAttr("multiplicity"))) {
        if (multiplicityAttr.empty()) elidedAttrs.push_back("multiplicity");
    }

    // Print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/elidedAttrs);

    // Print the region
    if (!isExternProcess) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

/// @brief Verifies if the process is correct.
///
/// All input ports should be of type `dfg::OutputType` and output ports of
/// type `dfg::InputType`.
///
/// If there is multiplicity, its size should match the number of arguments in
/// this process.
LogicalResult ProcessOp::verify()
{
    FunctionType funcType = getFunctionType();

    // All input ports must be dfg::OutputType and output ports dfg::InputType.
    if (!llvm::all_of(funcType.getInputs(), [](Type t) {
            return isa<OutputType>(t);
        }))
        return emitError("Input ports must be of type dfg::OutputType");
    if (!llvm::all_of(funcType.getResults(), [](Type t) {
            return isa<InputType>(t);
        }))
        return emitError("Output ports must be of type dfg::InputType");

    // If a multiplicity is defined, it must cover all arguments!
    ArrayRef<int64_t> multiplicity = getMultiplicity();
    size_t numArguments = funcType.getNumInputs() + funcType.getNumResults();
    if (!multiplicity.empty() && multiplicity.size() != numArguments)
        return emitError(
            "ProcessOp multiplicity must have a multiplicity for each "
            "channel");

    return success();
}

namespace {
struct EliminateDeadProcess final : public OpRewritePattern<ProcessOp> {
    using OpRewritePattern<ProcessOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(ProcessOp op, PatternRewriter &rewriter) const override
    {
        auto module = op->getParentOfType<ModuleOp>();
        if (!module) return failure();

        auto processName = op.getNodeName();
        bool isInstantiated = false;
        module->walk([&](InstantiateOp instantiateOp) {
            if (instantiateOp.getNodeName() == processName)
                isInstantiated = true;
        });

        if (isInstantiated) return failure();

        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for ProcessOp.
///
/// - EliminateDeadProcess: Erases the process if it's not instantiated any
/// where in the IR.
void ProcessOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{ patterns.add<EliminateDeadProcess>(context); }

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

/// @brief Builds a LoopOp with the given input, output values, and body.
///
/// The region is always created. @p bodyBuilder is called immediately with the
/// builder positioned at the entry block, so callers can populate the body
/// inline.
///
/// @param inputChannels Values that are used as inputs.
/// @param outputChannels Values that are used as outputs.
/// @param bodyBuilder  Callback to fill the entry block, can't be null.
void LoopOp::build(
    OpBuilder &builder,
    OperationState &state,
    ValueRange inputChannels,
    ValueRange outputChannels,
    function_ref<void(OpBuilder &, Location)> bodyBuilder)
{
    assert(bodyBuilder && "loop must have contents");

    state.addOperands(inputChannels);
    state.addOperands(outputChannels);
    llvm::copy(
        ArrayRef<int32_t>(
            {static_cast<int32_t>(inputChannels.size()),
             static_cast<int32_t>(outputChannels.size())}),
        state.getOrAddProperties<Properties>().operandSegmentSizes.begin());

    // Add init body (no arguments)
    OpBuilder::InsertionGuard guard(builder);
    Region* region = state.addRegion();
    builder.createBlock(region);

    bodyBuilder(builder, state.location);
}

/// @brief Parses a `dfg::LoopOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.loop`
///     (`inputs` `(` operand-list `)`)?
///     (`outputs` `(` operand-list `)`)?
///     region
/// @endcode
///
/// Unlike `ProcessOp`, where ports are declared as fresh block arguments,
/// `LoopOp` refers to channel values already live in the enclosing process
/// body.  The `inputs`/`outputs` lists are therefore parsed as resolved
/// operands (`%ssaName : type`) rather than new argument declarations.
/// The `operandSegmentSizes` attribute is derived from the parsed counts and
/// stored automatically.  The body region is mandatory and must be non-empty.
ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::Argument> inputValues, outputValues;

    // Parse inputs
    SMLoc inputLocation = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList(parser, inputValues)) return failure();
    }
    int32_t numInputs = inputValues.size();
    // Parse outputs
    SMLoc outputLocation = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList(parser, outputValues)) return failure();
    }
    int32_t numOutputs = outputValues.size();

    // Operand and types
    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs, outputs;
    inputs.reserve(numInputs);
    outputs.reserve(numOutputs);
    SmallVector<Type> inputTypes, outputTypes;
    inputTypes.reserve(numInputs);
    outputTypes.reserve(numOutputs);

    for (auto &input : inputValues) {
        inputs.push_back(input.ssaName);
        inputTypes.push_back(input.type);
    }
    for (auto &output : outputValues) {
        outputs.push_back(output.ssaName);
        outputTypes.push_back(output.type);
    }

    // Resolve the inputs and outputs
    if (parser
            .resolveOperands(inputs, inputTypes, inputLocation, state.operands))
        return failure();
    if (parser.resolveOperands(
            outputs,
            outputTypes,
            outputLocation,
            state.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    // Parse the body region
    auto* body = state.addRegion();
    if (parser.parseRegion(*body, /*arguments=*/{})) return failure();

    return success();
}

/// @brief Prints a `dfg::LoopOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// If any input channel operands are present the `inputs(...)` clause is
/// emitted, followed by the `outputs(...)` clause for output channels.
/// Within each clause every operand is printed as `%value : type`.  The
/// body region is always printed since a `LoopOp` without a region is
/// ill-formed.
void LoopOp::print(OpAsmPrinter &p)
{
    assert(
        !getOperands().empty()
        && "There has to be ports to be closed in LoopOp.");

    // Print inputs if exist
    auto inputs = getInputChannels();
    if (!inputs.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputs.size(); ++i) {
            if (i > 0) p << ", ";
            p.printOperand(inputs[i]);
            p << ": ";
            p.printType(inputs[i].getImpl()->getType());
        }
        p << ")";
    }

    // Print outputs if existent
    Operation::operand_range outputs = getOutputChannels();
    if (!outputs.empty()) {
        p << " outputs(";
        for (unsigned i = 0; i < outputs.size(); ++i) {
            if (i > 0) p << ", ";
            p.printOperand(outputs[i]);
            p << ": ";
            p.printType(outputs[i].getImpl()->getType());
        }
        p << ")";
    }

    // Print body region
    Region &body = getBody();
    if (!body.empty()) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

/// @brief Verifies that the `dfg::LoopOp` has a valid port mapping.
///
/// A `LoopOp` must monitor either all input ports of its parent `ProcessOp`
/// or none of them, and likewise for output ports.  Partial coverage —
/// monitoring only a subset — is rejected to avoid ambiguity in scheduling
/// semantics where the loop body would need to fire on an undefined subset
/// of channels.
LogicalResult LoopOp::verify()
{
    auto loopFuncType = getFunctionType();
    auto loopNumInputs = loopFuncType.getNumInputs();
    auto loopNumOutputs = loopFuncType.getNumResults();

    auto processFuncType = getParentOp().getFunctionType();
    auto processNumInputs = processFuncType.getNumInputs();
    auto processNumOutputs = processFuncType.getNumResults();

    if (loopNumInputs != 0 && loopNumInputs != processNumInputs)
        return emitOpError("LoopOp should monitor all input ports or none.");

    if (loopNumOutputs != 0 && loopNumOutputs != processNumOutputs)
        return emitOpError("LoopOp should monitor all output ports or none.");

    return success();
}

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

/// @brief Names the block arguments of the `OperatorOp` body region.
///
/// Input port arguments are named `%in0`, `%in1`, … in declaration order.
/// Note that `OperatorOp` only exposes input ports as block arguments; output
/// values are produced by `OutputOp` inside the body rather than received as
/// arguments, so the `else` branch that would emit `%outN` names is never
/// reached in practice.
void OperatorOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setNameFn)
{
    unsigned numInputs = getFunctionType().getNumInputs();
    for (unsigned i = 0, e = region.getNumArguments(); i < e; ++i)
        if (i < numInputs)
            setNameFn(region.getArgument(i), ("in" + std::to_string(i)));
        else
            setNameFn(
                region.getArgument(i),
                ("out" + std::to_string(i - numInputs)));
}

/// @brief Builds an `OperatorOp` with the given name, port types, and body.
///
/// Unlike `ProcessOp::build`, the function type is stored as-is without
/// DFG-type wrapping, so callers are responsible for supplying well-formed
/// types.  Only the input port types become block arguments in the entry
/// block; output ports are not represented as arguments because the body
/// yields them via `OutputOp` instead.
///
/// @param name         Symbol name of the operator (must be unique in the
///                     module).
/// @param functionType Port signature; types are used verbatim without
///                     wrapping.
/// @param bodyBuilder  Optional callback invoked with the builder positioned
///                     at the entry block; may be null for external
///                     declarations.
void OperatorOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType functionType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));
    state.addAttribute(
        OperatorOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(functionType));

    // Add init body (no arguments)
    OpBuilder::InsertionGuard guard(builder);
    Region* region = state.addRegion();
    auto regionBlock = builder.createBlock(region);

    // Create block arguments from the new function type
    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        functionType.getInputs().begin(),
        functionType.getInputs().end());
    regionBlock->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));

    if (bodyBuilder)
        bodyBuilder(builder, state.location, regionBlock->getArguments());
}

/// @brief Parses an `OperatorOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.operator` @name
///     (`inputs` `(` arg-list `)`)?
///     (`outputs` `(` arg-list `)`)?
///     (`attributes` attr-dict)?
///     region?
/// @endcode
///
/// The structure mirrors `ProcessOp::parse` with one key difference: only
/// the `inputs` argument list is forwarded as block arguments to the body
/// region.  The `outputs` list is used solely to reconstruct the
/// `FunctionType` attribute; output values are produced inside the body by
/// `OutputOp`, not received as arguments.  An absent region denotes an
/// external (declaration-only) operator.
ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &state)
{
    auto &builder = parser.getBuilder();

    // Parse the operator's name
    StringAttr nameAttr;
    if (parser.parseSymbolName(
            nameAttr,
            getSymNameAttrName(state.name),
            state.attributes))
        return failure();

    // Parse the signature of the operator
    SmallVector<OpAsmParser::Argument> inputValues, outputValues;
    SMLoc signatureLocation = parser.getCurrentLocation();

    // Parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList(parser, inputValues)) return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList(parser, outputValues)) return failure();
    }

    SmallVector<Type> argTypes, resultTypes;
    argTypes.reserve(inputValues.size());
    resultTypes.reserve(outputValues.size());

    for (auto &input : inputValues) argTypes.push_back(input.type);
    for (auto &output : outputValues) resultTypes.push_back(output.type);
    Type funcType = builder.getFunctionType(argTypes, resultTypes);

    if (!funcType) {
        return parser.emitError(signatureLocation)
               << "Failed to construct process type";
    }

    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(funcType));

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(state.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // Parse the attached region, if any
    auto* body = state.addRegion();
    SMLoc loc = parser.getCurrentLocation();
    OptionalParseResult parseResult = parser.parseOptionalRegion(
        *body,
        inputValues,
        /*enableNameShadowing=*/false);

    if (parseResult.has_value()) {
        if (failed(*parseResult)) return failure();
        if (body->empty())
            return parser.emitError(loc, "expected non-empty operator body");
    }

    return success();
}

/// @brief Prints an `OperatorOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// The symbol name is printed first, followed by optional `inputs(...)` and
/// `outputs(...)` clauses.  Input port names come from the body region's
/// block arguments for concrete operators, or use synthetic `%argN` names
/// for external declarations.  Output port names are always synthetic `%argN`
/// because outputs are not block arguments — they are produced by `OutputOp`
/// inside the body and therefore have no region argument to source names
/// from.  The body region is printed only for concrete operators; `sym_name`
/// and `function_type` are always elided from the attribute dictionary.
void OperatorOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = getBody();
    bool isExternOperator = isExternal();

    // Print the operation and its name
    auto name = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();

    p << ' ';
    p.printSymbolName(name);

    // Print input list
    auto inputTypes = getInputPortTypes();
    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); ++i) {
            if (i > 0) p << ", ";

            if (isExternOperator)
                p << "\%arg" << i;
            else
                p.printOperand(body.getArgument(i));
            p << ": " << inputTypes[i];
        }
        p << ")";
    }

    // Print output list
    auto outputTypes = getOutputPortTypes();
    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned numInputs = inputTypes.size();
        for (unsigned i = 0; i < outputTypes.size(); ++i) {
            if (i > 0) p << ", ";

            p << "\%arg" << i + numInputs;

            p << ": " << outputTypes[i];
        }
        p << ")";
    }

    SmallVector<StringRef, 3> elidedAttrs = {
        getFunctionTypeAttrName(),
        getSymNameAttrName()};

    // Print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/elidedAttrs);

    // Print the region
    if (!isExternOperator) {
        p << ' ';
        p.printRegion(
            body,
            /*printEntryBlockArgs =*/false,
            /*printBlockTerminators =*/true);
    }
}

/// @brief Verifies that the `OperatorOp` body contains only permitted ops.
///
/// The body may contain arbitrary non-DFG operations (e.g. arithmetic,
/// memory accesses), but the only DFG-dialect operation allowed inside is
/// `OutputOp`, which terminates the operator by writing computed values to
/// the output channels.  Any other DFG op found inside the body is rejected
/// because it would imply dataflow nesting that `OperatorOp` does not
/// support.
LogicalResult OperatorOp::verify()
{
    for (auto &opi : getBody().getOps()) {
        // Only allow OutputOp if the op is from this dialect
        if (opi.getDialect()->getNamespace() == "dfg") {
            if (!isa<OutputOp>(opi))
                return emitOpError(
                    "Only OutputOp from DFG dialect is allowed inside "
                    "OperatorOp.");
        }
    }
    return success();
}

namespace {
struct EliminateDeadOperator final : public OpRewritePattern<OperatorOp> {
    using OpRewritePattern<OperatorOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override
    {
        auto module = op->getParentOfType<ModuleOp>();
        if (!module) return failure();

        auto operatorName = op.getNodeName();
        bool isInstantiated = false;
        module->walk([&](InstantiateOp instantiateOp) {
            if (instantiateOp.getNodeName() == operatorName)
                isInstantiated = true;
        });

        if (isInstantiated) return failure();

        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for `OperatorOp`.
///
/// - EliminateDeadOperator: Erases an operator that is never referenced by
///   any `InstantiateOp` in the enclosing module, mirroring the dead-process
///   elimination applied to `ProcessOp`.
void OperatorOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{ patterns.add<EliminateDeadOperator>(context); }

//===----------------------------------------------------------------------===//
// RegionOp
//===----------------------------------------------------------------------===//

/// @brief Names the block arguments of the `RegionOp` body.
///
/// Input port arguments are named `%in0`, `%in1`, … and output port
/// arguments `%out0`, `%out1`, … in declaration order.  Both groups are
/// present as block arguments because `RegionOp` acts as a structural
/// sub-graph container: channels inside the body connect to both sides of
/// the port boundary.
void RegionOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setNameFn)
{
    unsigned numInputs = getFunctionType().getNumInputs();
    for (unsigned i = 0, e = region.getNumArguments(); i < e; ++i)
        if (i < numInputs)
            setNameFn(region.getArgument(i), ("in" + std::to_string(i)));
        else
            setNameFn(
                region.getArgument(i),
                ("out" + std::to_string(i - numInputs)));
}

/// @brief Builds a `RegionOp` with the given name, port types, and body.
///
/// Input and output types in @p functionType are wrapped in the corresponding
/// DFG channel types if not already present, following the same convention as
/// `ProcessOp::build`:
///  - function inputs  → `dfg::OutputType`
///  - function results → `dfg::InputType`
///
/// Both wrapped input and output types are added as block arguments so that
/// channel declarations inside the body can connect to either side of the
/// port boundary.
///
/// @param name         Symbol name of the region (must be unique in the
///                     module).
/// @param functionType Port signature before DFG-type wrapping.
/// @param bodyBuilder  Callback invoked with the builder at the entry block;
///                     may be null.
void RegionOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType functionType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder)
{
    state.addAttribute(
        SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    // Get the function type with dfg types from base
    SmallVector<Type> inPortTypes, outPortTypes;
    for (auto inTy : functionType.getInputs())
        inPortTypes.push_back(
            isa<OutputType>(inTy)   ? inTy
            : isa<ShapedType>(inTy) ? OutputType::get(cast<ShapedType>(inTy))
                                    : OutputType::get(inTy));
    for (auto outTy : functionType.getResults())
        outPortTypes.push_back(
            isa<InputType>(outTy)    ? outTy
            : isa<ShapedType>(outTy) ? InputType::get(cast<ShapedType>(outTy))
                                     : InputType::get(outTy));
    auto dfgFunctionType =
        FunctionType::get(builder.getContext(), inPortTypes, outPortTypes);
    state.addAttribute(
        RegionOp::getFunctionTypeAttrName(state.name),
        TypeAttr::get(dfgFunctionType));

    OpBuilder::InsertionGuard guard(builder);
    Region* region = state.addRegion();
    auto regionBlock = builder.createBlock(region);

    // Create block arguments from the new function type
    SmallVector<Type> blockArgTypes;
    blockArgTypes.append(
        dfgFunctionType.getInputs().begin(),
        dfgFunctionType.getInputs().end());
    blockArgTypes.append(
        dfgFunctionType.getResults().begin(),
        dfgFunctionType.getResults().end());
    regionBlock->addArguments(
        blockArgTypes,
        SmallVector<Location>(blockArgTypes.size(), builder.getUnknownLoc()));

    if (bodyBuilder)
        bodyBuilder(builder, state.location, regionBlock->getArguments());
}

/// @brief Parses a `dfg::RegionOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.region` @name
///     (`inputs` `(` arg-list `)`)?
///     (`outputs` `(` arg-list `)`)?
///     (`attributes` attr-dict)?
///     region
/// @endcode
///
/// The structure is the same as `ProcessOp::parse` — both argument lists are
/// merged and forwarded as block arguments — but a `RegionOp` is always
/// concrete: the body region is mandatory.  External `RegionOp` declarations
/// are not supported.
ParseResult RegionOp::parse(OpAsmParser &parser, OperationState &state)
{
    auto &builder = parser.getBuilder();

    // Parse region name
    StringAttr nameAttr;
    if (parser.parseSymbolName(
            nameAttr,
            getSymNameAttrName(state.name),
            state.attributes))
        return failure();

    // Parse the signature of this region
    SmallVector<OpAsmParser::Argument> inputValues, outputValues;
    SMLoc signatureLocation = parser.getCurrentLocation();

    // Parse inputs/outputs separately for later distinction
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parseChannelArgumentList(parser, inputValues)) return failure();
    }
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parseChannelArgumentList(parser, outputValues)) return failure();
    }

    // Create the signature
    SmallVector<Type> inputTypes, outputTypes;
    inputTypes.reserve(inputValues.size());
    outputTypes.reserve(outputValues.size());

    for (auto &input : inputValues) inputTypes.push_back(input.type);
    for (auto &output : outputValues) outputTypes.push_back(output.type);
    Type funcType = builder.getFunctionType(inputTypes, outputTypes);
    if (!funcType) {
        return parser.emitError(signatureLocation)
               << "Failed to construct operator type";
    }
    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        TypeAttr::get(funcType));

    // Merge both argument lists for the block arguments
    inputValues.append(outputValues);

    OptionalParseResult attrResult =
        parser.parseOptionalAttrDictWithKeyword(state.attributes);
    if (attrResult.has_value() && failed(*attrResult)) return failure();

    // Parse the attached region
    auto* body = state.addRegion();
    SMLoc loc = parser.getCurrentLocation();
    OptionalParseResult parseResult = parser.parseRegion(
        *body,
        inputValues,
        /*enableNameShadowing=*/false);

    if (parseResult.has_value()) {
        if (failed(*parseResult)) return failure();
        if (body->empty())
            return parser.emitError(loc, "expected non-empty operator body");
    }

    return success();
}

/// @brief Prints a `dfg::RegionOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// The symbol name is printed first, followed by optional `inputs(...)` and
/// `outputs(...)` clauses where port names are sourced from the body's block
/// arguments.  The `sym_name` and `function_type` attributes are always
/// elided.  Unlike `ProcessOp`, the body region is unconditionally emitted
/// because `RegionOp` has no external-declaration form.
void RegionOp::print(OpAsmPrinter &p)
{
    Operation* op = getOperation();
    Region &body = op->getRegion(0);

    // Print the region's name
    auto name = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    .getValue();
    p << ' ';
    p.printSymbolName(name);

    // Print input list
    auto funcType = getFunctionType();
    auto inputTypes = funcType.getInputs();
    if (!inputTypes.empty()) {
        p << " inputs(";
        for (unsigned i = 0; i < inputTypes.size(); ++i) {
            if (i > 0) p << ", ";
            p.printOperand(body.getArgument(i));
            p << " : " << inputTypes[i];
        }
        p << ") ";
    }

    // Print output list
    auto outputTypes = funcType.getResults();
    if (!outputTypes.empty()) {
        p << " outputs(";
        unsigned numInputs = inputTypes.size();
        for (unsigned i = 0; i < outputTypes.size(); ++i) {
            if (i > 0) p << ", ";
            p.printOperand(body.getArgument(i + numInputs));
            p << " : " << outputTypes[i];
        }
        p << ") ";
    }

    // Print any attributes in the attribute list into the dict
    if (!op->getAttrs().empty())
        p.printOptionalAttrDictWithKeyword(
            op->getAttrs(),
            /*elidedAttrs=*/{getFunctionTypeAttrName(), getSymNameAttrName()});

    // Print the region
    p << ' ';
    p.printRegion(
        body,
        /*printEntryBlockArgs =*/false,
        /*printBlockTerminators =*/true);
}

/// @brief Verifies that a `dfg::RegionOp` is structurally well-formed.
///
///
/// All input ports should be of type `dfg::OutputType` and output ports of
/// type `dfg::InputType`.
///
/// The body is expected to contain only `ChannelOp`, `InstantiateOp`, and
/// `EmbedOp`; any other operation triggers a warning (not an error) to allow
/// experimental extensions without hard breakage.
///
/// Additionally, each body block argument — representing a port on the region
/// boundary — is checked for connectivity: a port with no uses is reported as
/// dangling, and a port used more than once is reported as multiply connected.
/// Both conditions are emitted as warnings rather than errors because they
/// may be intentional during incremental IR construction.
LogicalResult RegionOp::verify()
{
    auto funcType = getFunctionType();
    // All input ports must be dfg::OutputType and output ports dfg::InputType.
    if (!llvm::all_of(funcType.getInputs(), [](Type t) {
            return isa<OutputType>(t);
        }))
        return emitError("Input ports must be of type dfg::OutputType");
    if (!llvm::all_of(funcType.getResults(), [](Type t) {
            return isa<InputType>(t);
        }))
        return emitError("Output ports must be of type dfg::InputType");
    // Check if the content operations are supported
    for (auto &opi : getBody().getOps())
        if (!isa<ChannelOp, InstantiateOp, EmbedOp>(opi))
            ::emitWarning(
                opi.getLoc(),
                "Unsupported op used in this region with name ")
                << opi.getName();

    // Check if all ports are connected
    for (auto [idx, arg] : llvm::enumerate(getBody().getArguments())) {
        if (arg.use_empty()) {
            if (idx <= getNumInputPorts()) {
                ::emitWarning(getLoc(), "Detecting dangling input port #")
                    << idx;
            } else {
                ::emitWarning(getLoc(), "Detecting dangling output port #")
                    << idx - getNumInputPorts();
            }
        } else if (!arg.hasOneUse()) {
            if (idx <= getNumInputPorts()) {
                ::emitWarning(getLoc(), "Detecting input port #")
                    << idx << " used more than once";
            } else {
                ::emitWarning(getLoc(), "Detecting output port #")
                    << idx - getNumInputPorts() << " used more than once";
            }
        }
    }

    return success();
}

namespace {
struct ReorderContentOperations final : public OpRewritePattern<RegionOp> {
    using OpRewritePattern<RegionOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(RegionOp op, PatternRewriter &rewriter) const override
    {
        if (isReordered[op]) return success();

        auto loc = op.getLoc();
        auto funcTy = op.getFunctionType();

        // Create a new region with same name and signature
        auto newRegion = RegionOp::create(
            rewriter,
            loc,
            op.getNodeName(),
            funcTy,
            [&](OpBuilder &, Location, ValueRange regionBlkArgs) {
                IRMapping mapper;
                for (auto [oldArg, newArg] :
                     llvm::zip(op.getBody().getArguments(), regionBlkArgs)) {
                    mapper.map(oldArg, newArg);
                }

                // Save operations
                SmallVector<Operation*> channels, instances;
                for (auto &opOldRegion : op.getBody().getOps())
                    if (isa<ChannelOp>(opOldRegion))
                        channels.push_back(&opOldRegion);
                    else if (isa<InstantiateOp, EmbedOp>(opOldRegion))
                        instances.push_back(&opOldRegion);

                // Copy operations in order
                for (auto &channel : channels) rewriter.clone(*channel, mapper);
                for (auto &instance : instances)
                    rewriter.clone(*instance, mapper);
            });

        if (Attribute rootAttr = op->getAttr(laksa::kRootAttrName))
            newRegion->setAttr(laksa::kRootAttrName, rootAttr);

        // Save this new region as reordered
        rewriter.replaceOp(op, newRegion);
        isReordered[newRegion] = true;
        return success();
    }

private:
    mutable DenseMap<RegionOp, bool> isReordered;
};
} // namespace

/// @brief Canonicalization patterns for `dfg::RegionOp`.
///
/// - ReorderContentOperations: Rebuilds the region body so that all
///   `ChannelOp` declarations appear before any `InstantiateOp` or `EmbedOp`.
///   This canonical ordering simplifies analyses that expect channel SSA
///   values to be defined before the nodes that consume them.  An internal
///   visited-set prevents re-rewriting an already reordered region.
void RegionOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{ patterns.add<ReorderContentOperations>(context); }

//===----------------------------------------------------------------------===//
// DFGDialect Edge Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ChannelOp
//===----------------------------------------------------------------------===//

/// @brief Assigns pretty names to the two results of a `dfg::ChannelOp`.
///
/// The input-port result is named `%in_port_N` and the output-port result
/// `%out_port_N`, where N is the zero-based index of this channel among all
/// `ChannelOp`s defined earlier in the same block.  The count is computed by
/// walking the block until the current operation is reached, so names are
/// stable and unique regardless of how many channels share a block.
void ChannelOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Block* block = getOperation()->getBlock();
    int count = 0;

    for (auto &opi : *block) {
        if (&opi == getOperation()) break;
        if (opi.getName() == getOperation()->getName()) ++count;
    }

    setNameFn(getResult(0), "in_port_" + std::to_string(count));
    setNameFn(getResult(1), "out_port_" + std::to_string(count));
}

/// @brief Builds a `dfg::ChannelOp` for a channel with the given token type.
///
/// The two result types (`dfg::InputType` and `dfg::OutputType`) are derived
/// automatically from @p shape and @p tokenType.
///
/// @param shape      Dimension sizes for shaped channels; empty for scalar.
/// @param tokenType  Element type of each token transferred on the channel.
/// @param bufferSize Optional bound on the FIFO depth; absent means
///                   unbounded.
void ChannelOp::build(
    OpBuilder &builder,
    OperationState &state,
    ArrayRef<int64_t> shape,
    Type tokenType,
    std::optional<int32_t> bufferSize)
{
    state.addAttribute(
        getTokenTypeAttrName(state.name),
        TypeAttr::get(tokenType));
    if (bufferSize.has_value())
        state.addAttribute(
            getBufferSizeAttrName(state.name),
            builder.getI32IntegerAttr(bufferSize.value()));
    state.addTypes(InputType::get(shape, tokenType));
    state.addTypes(OutputType::get(shape, tokenType));
}

/// @brief Builds a `dfg::ChannelOp` for a channel with the given token type.
///
/// The two result types (`dfg::InputType` and `dfg::OutputType`) are derived
/// automatically from @p shapedType.
void ChannelOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type type,
    std::optional<int32_t> bufferSize)
{
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
        build(
            builder,
            state,
            shapedType.getShape(),
            shapedType.getElementType(),
            bufferSize);
    } else {
        build(builder, state, ArrayRef<int64_t>{}, type, bufferSize);
    }
}

/// @brief Parses a `dfg::ChannelOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.channel` `(` bufferSize? `)` `:` dimension-list token-type
/// @endcode
///
/// The parenthesised integer is optional; when omitted, the buffer-size
/// attribute is stored as 0 to indicate an unbounded channel.  The dimension
/// list uses the standard `x`-separated dimension form and may be empty for
/// scalar channels.  The two result types are derived from the parsed shape
/// and token type and do not appear explicitly in the source text.
ParseResult ChannelOp::parse(OpAsmParser &parser, OperationState &state)
{
    if (parser.parseLParen()) return failure();

    // Parse an optional buffer size
    int32_t bufferSize = 0;
    OptionalParseResult bufferSizeResult =
        parser.parseOptionalInteger(bufferSize);
    if (bufferSizeResult.has_value() && failed(*bufferSizeResult))
        return failure();
    state.addAttribute(
        getBufferSizeAttrName(state.name),
        parser.getBuilder().getI32IntegerAttr(bufferSize));

    if (parser.parseRParen() || parser.parseColon()) return failure();

    // Parse optional shape (dimension list) followed by token type
    SmallVector<int64_t> shape;
    Type tokenType;
    if (parser.parseDimensionList(shape, /*allowDynamic=*/false))
        return failure();
    if (parser.parseType(tokenType)) return failure();
    state.addAttribute(
        getTokenTypeAttrName(state.name),
        TypeAttr::get(tokenType));

    // Add result types
    state.addTypes(InputType::get(shape, tokenType));
    state.addTypes(OutputType::get(shape, tokenType));

    return success();
}

/// @brief Prints a `dfg::ChannelOp` in the custom assembly syntax consumed
///        by `parse`.
///
/// Emits `(bufferSize) : dimensionList x tokenType`.  The buffer-size integer
/// is omitted — leaving empty parentheses — when no explicit size is set.
/// For shaped channels the dimension list is followed by `x` before the
/// element type; for scalar channels both the shape and the separator are
/// suppressed.
void ChannelOp::print(OpAsmPrinter &p)
{
    p << '(';
    if (const auto bufferSize = getBufferSize())
        if (bufferSize.value() != 0) p << bufferSize.value();
    p << ") : ";
    auto inputType = cast<InputType>(getInputPort().getType());
    p.printDimensionList(inputType.getShape());
    if (!inputType.getShape().empty()) p << 'x';
    p.printType(inputType.getElementType());
}

/// @brief Verifies that both ports of the channel are in use.
///
/// A channel whose input port or output port has no uses represents a
/// dangling endpoint that will never transfer data.  This condition is
/// reported as a warning rather than an error to permit incremental IR
/// construction where port connectivity is established in a later step.
LogicalResult ChannelOp::verify()
{
    // Both ports should be used
    if (getInputPort().getUses().empty() || getOutputPort().getUses().empty())
        ::emitWarning(getLoc(), "Detect dangling channel port.");
    return success();
}

//===----------------------------------------------------------------------===//
// DFGDialect Instance Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InstantiateOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `InstantiateOp` that references an actor by name.
///
/// Records the actor symbol, input and output channel operands,
/// `operandSegmentSizes`, and an optional offload annotation.
///
/// @param actorName  Symbol name of the process or operator to instantiate.
/// @param inputs     Channel values connected to the actor's input ports.
/// @param outputs    Channel values connected to the actor's output ports.
/// @param offloaded  Optional hardware backend for offloading this instance.
void InstantiateOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef actorName,
    ValueRange inputs,
    ValueRange outputs,
    std::optional<OffloadHardware> offloaded)
{
    auto actorNameAttr = SymbolRefAttr::get(builder.getContext(), actorName);
    state.addAttribute(getActorAttrName(state.name), actorNameAttr);

    state.addOperands(inputs);
    state.addOperands(outputs);
    state.addAttribute(
        kOperandSegmentSizesAttr,
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(inputs.size()),
             static_cast<int32_t>(outputs.size())}));

    if (offloaded)
        state.addAttribute(
            getOffloadedAttrName(state.name),
            OffloadHardwareAttr::get(builder.getContext(), offloaded.value()));
}

/// @brief Convenience overload that takes the actor as a `ProcessOp`.
///
/// Extracts the node name from @p actorProcess and delegates to the
/// string-based primary build method.
///
/// @param actorProcess The process definition to instantiate.
/// @param inputs       Channel values connected to the process's input ports.
/// @param outputs      Channel values connected to the process's output ports.
/// @param offloaded    Optional hardware backend for offloading this instance.
void InstantiateOp::build(
    OpBuilder &builder,
    OperationState &state,
    ProcessOp actorProcess,
    ValueRange inputs,
    ValueRange outputs,
    std::optional<OffloadHardware> offloaded)
{
    build(
        builder,
        state,
        actorProcess.getNodeName(),
        inputs,
        outputs,
        offloaded);
}

/// @brief Convenience overload that takes the actor as an `OperatorOp`.
///
/// Extracts the node name from @p actorOperator and delegates to the
/// string-based primary build method.
///
/// @param actorOperator The operator definition to instantiate.
/// @param inputs        Channel values connected to the operator's input
///                      ports.
/// @param outputs       Channel values connected to the operator's output
///                      ports.
/// @param offloaded     Optional hardware backend for offloading this
///                      instance.
void InstantiateOp::build(
    OpBuilder &builder,
    OperationState &state,
    OperatorOp actorOperator,
    ValueRange inputs,
    ValueRange outputs,
    std::optional<OffloadHardware> offloaded)
{
    build(
        builder,
        state,
        actorOperator.getNodeName(),
        inputs,
        outputs,
        offloaded);
}

/// @brief Parses an `InstantiateOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.instantiate`
///     (`offloaded` (`=` hardware-name)?)?
///     @actorName
///     (`inputs` `(` operand-list `)`)?
///     (`outputs` `(` operand-list `)`)?
///     `:` func-type
/// @endcode
///
/// The optional `offloaded` keyword enables hardware offloading; appending
/// `=name` selects a specific backend (default: `EmitHLS`).  The
/// `inputs`/`outputs` lists contain pre-existing SSA operands rather than
/// new declarations, and port types are not written inline — they are
/// resolved through the trailing `FunctionType`.  The
/// `operandSegmentSizes` attribute is derived automatically from the parsed
/// operand counts.
ParseResult InstantiateOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Optionally mark as offloaded, which optionally mark to which hardware
    if (succeeded(parser.parseOptionalKeyword("offloaded"))) {
        OffloadHardware offHW = OffloadHardware::EmitHLS; // Default backend
        if (succeeded(parser.parseEqual())) {
            StringRef hwStr;
            if (parser.parseKeyword(&hwStr)) return failure();
            auto hwVal = symbolizeOffloadHardware(hwStr);
            if (!hwVal)
                return parser.emitError(
                           parser.getNameLoc(),
                           "unknown offload hardware: ")
                       << hwStr;
            offHW = *hwVal;
        }
        state.addAttribute(
            getOffloadedAttrName(state.name),
            OffloadHardwareAttr::get(parser.getContext(), offHW));
    }

    // Parse instantiated process/operator's name
    StringAttr actorName;
    if (parser.parseSymbolName(actorName)) return failure();
    state.addAttribute(
        getActorAttrName(state.name),
        SymbolRefAttr::get(actorName));

    // Parse inputs
    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(inputs) ||
            // parser.resolveOperands(inputs, opTy, result.operands)) {
            parser.parseRParen())
            return failure();
    }

    // Parse outputs
    SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs;
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(outputs)
            || parser.parseRParen())
            return failure();
    }

    // Parse function type
    if (parser.parseColon()) return failure();
    SMLoc location = parser.getCurrentLocation();
    FunctionType funcType;
    if (parser.parseType(funcType)) return failure();

    // Check function type and operands sizes
    auto inputTypes = funcType.getInputs();
    auto outputTypes = funcType.getResults();
    int32_t numInputs = inputTypes.size();
    int32_t numOutputs = outputTypes.size();
    if ((unsigned)numInputs != inputs.size()
        || (unsigned)numOutputs != outputs.size()) {
        parser.emitError(
            location,
            "Call signature does not match operand count");
    }

    // Resolve operands
    if (parser.resolveOperands(inputs, inputTypes, location, state.operands))
        return failure();
    if (parser.resolveOperands(outputs, outputTypes, location, state.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

/// @brief Prints an `InstantiateOp` in the custom assembly syntax consumed
///        by `parse`.
///
/// Emits the optional `offloaded=HW` annotation, the actor symbol name,
/// optional `inputs(...)` and `outputs(...)` operand lists, and the
/// trailing function type.
void InstantiateOp::print(OpAsmPrinter &p)
{
    // Has offloaded notation
    if (auto offloaded = getOffloadedAttr())
        p << " offloaded=" << stringifyOffloadHardware(offloaded.getValue());

    // Actor name
    p << ' ';
    p.printAttributeWithoutType(getActorAttr());

    // Print inputs and output if exists
    if (!getInputs().empty()) p << " inputs(" << getInputs() << ")";
    if (!getOutputs().empty()) p << " outputs(" << getOutputs() << ")";

    // Print function type
    p << " : " << getFunctionType();
}

/// @brief Verifies that the instantiated actor exists and has a matching
///        signature.
///
/// Looks up the referenced actor by symbol and confirms that its base
/// function type matches the one declared by this `InstantiateOp`.  A
/// missing actor or a type mismatch are both hard errors because either
/// condition would leave port connections unresolvable at code-generation
/// time.
LogicalResult InstantiateOp::verify()
{
    auto instantiatedOp = getInstantiatedOperation();
    if (!instantiatedOp)
        return emitOpError("Cannot find embedded region operation.");

    auto baseFuncTy = getBaseFunctionType();
    auto instantiatedBaseFuncTy =
        dyn_cast<NodeInterface>(instantiatedOp).getBaseFunctionType();
    if (baseFuncTy != instantiatedBaseFuncTy)
        return emitOpError(
            "Expected same signature of embedded region and this operation.");

    return success();
}

//===----------------------------------------------------------------------===//
// EmbedOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `EmbedOp` that references a sub-graph region by name.
///
/// Mirrors `InstantiateOp::build`: records the actor symbol, input and
/// output channel operands, `operandSegmentSizes`, and an optional offload
/// annotation.
///
/// @param actorName  Symbol name of the region to embed.
/// @param inputs     Channel values connected to the region's input ports.
/// @param outputs    Channel values connected to the region's output ports.
/// @param offloaded  Optional hardware backend for offloading this embed.
void EmbedOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef actorName,
    ValueRange inputs,
    ValueRange outputs,
    std::optional<OffloadHardware> offloaded)
{
    auto actorNameAttr = SymbolRefAttr::get(builder.getContext(), actorName);
    state.addAttribute(getActorAttrName(state.name), actorNameAttr);

    state.addOperands(inputs);
    state.addOperands(outputs);
    state.addAttribute(
        kOperandSegmentSizesAttr,
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(inputs.size()),
             static_cast<int32_t>(outputs.size())}));

    if (offloaded)
        state.addAttribute(
            getOffloadedAttrName(state.name),
            OffloadHardwareAttr::get(builder.getContext(), offloaded.value()));
}

/// @brief Convenience overload that takes the target as a `RegionOp`.
///
/// Extracts the node name from @p actorRegion and delegates to the
/// string-based primary build method.
///
/// @param actorRegion The region definition to embed.
/// @param inputs      Channel values connected to the region's input ports.
/// @param outputs     Channel values connected to the region's output ports.
/// @param offloaded   Optional hardware backend for offloading this embed.
void EmbedOp::build(
    OpBuilder &builder,
    OperationState &state,
    RegionOp actorRegion,
    ValueRange inputs,
    ValueRange outputs,
    std::optional<OffloadHardware> offloaded)
{
    build(
        builder,
        state,
        actorRegion.getNodeName(),
        inputs,
        outputs,
        offloaded);
}

/// @brief Parses an `EmbedOp` from its custom assembly syntax.
///
/// The syntax and parsing logic are identical to `InstantiateOp::parse`;
/// the only semantic distinction is that the referenced actor must resolve
/// to a `RegionOp` (a structural sub-graph) rather than a process or
/// operator node.
ParseResult EmbedOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Optionally mark as offloaded, which optionally mark to which hardware
    if (succeeded(parser.parseOptionalKeyword("offloaded"))) {
        OffloadHardware offHW = OffloadHardware::EmitHLS; // Default backend
        if (succeeded(parser.parseEqual())) {
            StringRef hwStr;
            if (parser.parseKeyword(&hwStr)) return failure();
            auto hwVal = symbolizeOffloadHardware(hwStr);
            if (!hwVal)
                return parser.emitError(
                           parser.getNameLoc(),
                           "unknown offload hardware: ")
                       << hwStr;
            offHW = *hwVal;
        }
        state.addAttribute(
            getOffloadedAttrName(state.name),
            OffloadHardwareAttr::get(parser.getContext(), offHW));
    }

    // Parse embed region's name
    StringAttr actorName;
    if (parser.parseSymbolName(actorName)) return failure();
    state.addAttribute(
        getActorAttrName(state.name),
        SymbolRefAttr::get(actorName));

    // Parse inputs
    SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
    if (succeeded(parser.parseOptionalKeyword("inputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(inputs) ||
            // parser.resolveOperands(inputs, opTy, result.operands)) {
            parser.parseRParen())
            return failure();
    }

    // Parse outputs
    SmallVector<OpAsmParser::UnresolvedOperand, 4> outputs;
    if (succeeded(parser.parseOptionalKeyword("outputs"))) {
        if (parser.parseLParen() || parser.parseOperandList(outputs)
            || parser.parseRParen())
            return failure();
    }

    // Parse function type
    if (parser.parseColon()) return failure();
    SMLoc location = parser.getCurrentLocation();
    FunctionType funcType;
    if (parser.parseType(funcType)) return failure();

    // Check function type and operands sizes
    auto inputTypes = funcType.getInputs();
    auto outputTypes = funcType.getResults();
    int32_t numInputs = inputTypes.size();
    int32_t numOutputs = outputTypes.size();
    if ((unsigned)numInputs != inputs.size()
        || (unsigned)numOutputs != outputs.size()) {
        parser.emitError(
            location,
            "Call signature does not match operand count");
    }

    // Resolve operands
    if (parser.resolveOperands(inputs, inputTypes, location, state.operands))
        return failure();
    if (parser.resolveOperands(outputs, outputTypes, location, state.operands))
        return failure();

    // Add derived `operand_segment_sizes` attribute based on parsed
    // operands.
    auto operandSegmentSizes =
        parser.getBuilder().getDenseI32ArrayAttr({numInputs, numOutputs});
    state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

/// @brief Prints an `EmbedOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// Mirrors `InstantiateOp::print`: emits the optional offload annotation,
/// the region symbol name, optional `inputs`/`outputs` operand lists, and
/// the trailing function type.
void EmbedOp::print(OpAsmPrinter &p)
{
    // Has offloaded notation
    if (auto offloaded = getOffloadedAttr())
        p << " offloaded=" << stringifyOffloadHardware(offloaded.getValue());

    // Actor name
    p << ' ';
    p.printAttributeWithoutType(getActorAttr());

    // Print inputs and output if exists
    if (!getInputs().empty()) p << " inputs(" << getInputs() << ")";
    if (!getOutputs().empty()) p << " outputs(" << getOutputs() << ")";

    // Print function type
    p << " : " << getFunctionType();
}

/// @brief Verifies that the embedded region exists and has a matching
///        signature.
///
/// Mirrors `InstantiateOp::verify`: looks up the embedded `RegionOp` by
/// symbol and confirms that its base function type matches the one declared
/// by this `EmbedOp`.  A missing region or a type mismatch are both hard
/// errors.
LogicalResult EmbedOp::verify()
{
    auto embeddedOp = getEmbeddedOperation();
    if (!embeddedOp)
        return emitOpError("Cannot find embedded region operation.");

    auto baseFuncTy = getBaseFunctionType();
    auto embedBaseFuncTy =
        dyn_cast<NodeInterface>(embeddedOp).getBaseFunctionType();
    if (baseFuncTy != embedBaseFuncTy)
        return emitOpError(
            "Expected same signature of embedded region and this operation.");

    return success();
}

//===----------------------------------------------------------------------===//
// DFG structure operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

/// @brief Verifies that the `OutputOp` operands match the enclosing
///        `OperatorOp`'s output port types.
///
/// Checks that the number of operands equals the number of output ports on
/// the parent `OperatorOp` and that each operand type matches the
/// corresponding port type in order.  Mismatches are hard errors because
/// they represent a type-unsafe connection between the operator body and its
/// output channels.
LogicalResult OutputOp::verify()
{
    auto operatorOutputTypes = getParentOp().getOutputPortTypes();
    auto outputTypes = getOperandTypes();

    if (outputTypes.size() != operatorOutputTypes.size())
        return emitOpError("Number of outputs must match operator's outputs.");

    for (auto [operatorOutType, thisOutType] :
         llvm::zip(operatorOutputTypes, outputTypes)) {
        if (operatorOutType != thisOutType)
            return emitOpError(
                "Output types must match operator's output types.");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// PullOp
//===----------------------------------------------------------------------===//

/// @brief Assigns a pretty name to the token result of a `dfg::PullOp`.
///
/// The result is named `%tokenN` where N is the zero-based count of
/// `PullOp`s that appear before this one inside the enclosing DFG node.
/// The walk is interrupted as soon as the current operation is reached,
/// keeping the traversal proportional to the number of pull operations in
/// the node rather than the total size of the body.
void PullOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* parent = self->getParentOfType<NodeInterface>();

    int count = 0;
    parent->walk([&](PullOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getResult(), "token" + std::to_string(count));
}

/// @brief Builds a `dfg::PullOp` that reads from a shaped or scalar port.
///
/// The result type is inferred as the element type of the `dfg::OutputType`
/// port operand.
///
/// @param readPort  The `dfg::OutputType` port to read from.
/// @param indices   Element indices for shaped ports; must be empty for
///                  scalar ports.
void PullOp::build(
    OpBuilder &,
    OperationState &state,
    Value readPort,
    ValueRange indices)
{
    state.addOperands(readPort);
    if (!indices.empty()) state.addOperands(indices);
    auto readPortTy = dyn_cast<OutputType>(readPort.getType()).getElementType();
    state.addTypes(readPortTy);
}

/// @brief Parses a `dfg::PullOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.pull` %readPort (`[` index-list `]`)? `:` port-type
/// @endcode
///
/// The port type must be a `dfg::OutputType`.  Index operands, if present,
/// must be of `index` type and address a specific element when pulling from
/// a shaped port.  The result type is derived as the element type of the
/// port and is not written explicitly.
ParseResult PullOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse read port
    OpAsmParser::UnresolvedOperand readPort;
    if (parser.parseOperand(readPort)) return failure();

    // Indices if port is shaped
    SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
    if (succeeded(parser.parseOptionalLSquare())) {
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand idx;
                if (parser.parseOperand(idx)) return failure();
                indices.push_back(idx);
                return success();
            })
            || parser.parseRSquare())
            return failure();
    }

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();
    state.addTypes(cast<OutputType>(portType).getElementType());

    // Resolve operands
    if (parser.resolveOperand(readPort, portType, state.operands))
        return failure();
    if (!indices.empty()) {
        for (auto idx : indices)
            if (parser.resolveOperand(
                    idx,
                    IndexType::get(parser.getContext()),
                    state.operands))
                return failure();
    }

    return success();
}

/// @brief Prints a `dfg::PullOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// Emits `%readPort[idx, ...] : portType`, omitting the bracket list when
/// no indices are present.
void PullOp::print(OpAsmPrinter &p)
{
    p << " " << getReadPort();
    auto indices = getIndices();
    if (!indices.empty()) {
        p << "[";
        for (size_t i = 0; i < indices.size(); ++i) {
            p << indices[i];
            if (i < indices.size() - 1) p << ", ";
        }
        p << "]";
    }
    p << " : " << getReadPort().getType();
}

/// @brief Verifies that the index list is consistent with the port shape.
///
/// For scalar ports, no indices may be provided.  For shaped ports, the
/// number of index operands must exactly match the port rank.  Either
/// mismatch is a hard error because it would lead to out-of-bounds or
/// under-specified element access at runtime.
LogicalResult PullOp::verify()
{
    auto portType = cast<OutputType>(getReadPort().getType());
    auto shape = portType.getShape();
    auto indices = getIndices();

    if (shape.empty() && !indices.empty())
        return emitOpError("Indices provided but read port has no shape");

    if (!shape.empty() && indices.size() != shape.size())
        return emitOpError("Expected ") << shape.size()
                                        << " index operand(s) to match the "
                                           "rank of the read port, but got "
                                        << indices.size();

    return success();
}

//===----------------------------------------------------------------------===//
// PullAsTensorOp
//===----------------------------------------------------------------------===//

/// @brief Assigns a pretty name to the tensor result of a
///        `dfg::PullAsTensorOp`.
///
/// The result is named `%token_tensorN` where N is the count of
/// `PullAsTensorOp`s preceding this one inside the enclosing DFG node,
/// following the same walk-and-count strategy used by `PullOp`.
void PullAsTensorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* parent = self->getParentOfType<NodeInterface>();

    int count = 0;
    parent->walk([&](PullAsTensorOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getResult(), "token_tensor" + std::to_string(count));
}

/// @brief Builds a `dfg::PullAsTensorOp` that reads an entire shaped port
///        as a ranked tensor.
///
/// The result type is a `RankedTensorType` whose shape and element type are
/// taken from the `dfg::OutputType` of @p readPort.
///
/// @param readPort  The shaped `dfg::OutputType` port to read; scalar ports
///                  are rejected by `verify`.
void PullAsTensorOp::build(OpBuilder &, OperationState &state, Value readPort)
{
    state.addOperands(readPort);
    auto portType = cast<OutputType>(readPort.getType());
    state.addTypes(
        RankedTensorType::get(portType.getShape(), portType.getElementType()));
}

/// @brief Parses a `dfg::PullAsTensorOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.pull_as_tensor` %readPort `:` port-type
/// @endcode
///
/// No index list is accepted because this operation always reads the full
/// shaped extent of the port.  The result type (a `RankedTensorType`) is
/// derived from the port type and is not written explicitly.
ParseResult PullAsTensorOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse read port
    OpAsmParser::UnresolvedOperand readPort;
    if (parser.parseOperand(readPort)) return failure();

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();
    auto portDFGType = cast<OutputType>(portType);
    state.addTypes(
        RankedTensorType::get(
            portDFGType.getShape(),
            portDFGType.getElementType()));

    // Resolve operands
    if (parser.resolveOperand(readPort, portType, state.operands))
        return failure();

    return success();
}

/// @brief Prints a `dfg::PullAsTensorOp` in the custom assembly syntax
///        consumed by `parse`.
///
/// Emits `%readPort : portType`; the tensor result type is implicit.
void PullAsTensorOp::print(OpAsmPrinter &p)
{ p << " " << getReadPort() << " : " << getReadPort().getType(); }

/// @brief Verifies that the source port is shaped.
///
/// Pulling a scalar port as a tensor is ill-typed, so it is rejected as a
/// hard error.
LogicalResult PullAsTensorOp::verify()
{
    // Check if read port is shaped
    if (getReadPort().getType().getShape().empty())
        return emitOpError("Cannot pull as tensor from a scalar read port.");
    return success();
}

//===----------------------------------------------------------------------===//
// PullAsMemRefOp
//===----------------------------------------------------------------------===//

/// @brief Assigns a pretty name to the memref result of a
///        `dfg::PullAsMemRefOp`.
///
/// The result is named `%token_memrefN` where N is the count of
/// `PullAsMemRefOp`s preceding this one inside the enclosing DFG node,
/// following the same walk-and-count strategy used by `PullOp`.
void PullAsMemRefOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* parent = self->getParentOfType<NodeInterface>();

    int count = 0;
    parent->walk([&](PullAsMemRefOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getResult(), "token_memref" + std::to_string(count));
}

/// @brief Builds a `dfg::PullAsMemRefOp` that reads an entire shaped port
///        as a ranked memref.
///
/// The result type is a `MemRefType` whose shape and element type are taken
/// from the `dfg::OutputType` of @p readPort.
///
/// @param readPort  The shaped `dfg::OutputType` port to read; scalar ports
///                  are rejected by `verify`.
void PullAsMemRefOp::build(OpBuilder &, OperationState &state, Value readPort)
{
    state.addOperands(readPort);
    auto portType = cast<OutputType>(readPort.getType());
    state.addTypes(
        MemRefType::get(portType.getShape(), portType.getElementType()));
}

/// @brief Parses a `dfg::PullAsMemRefOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.pull_as_memref` %readPort `:` port-type
/// @endcode
///
/// Mirrors `PullAsTensorOp::parse`; the memref result type is derived from
/// the port type and is not written explicitly.
ParseResult PullAsMemRefOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse read port
    OpAsmParser::UnresolvedOperand readPort;
    if (parser.parseOperand(readPort)) return failure();

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();
    auto portDFGType = cast<OutputType>(portType);
    state.addTypes(
        MemRefType::get(portDFGType.getShape(), portDFGType.getElementType()));

    // Resolve operands
    if (parser.resolveOperand(readPort, portType, state.operands))
        return failure();

    return success();
}

/// @brief Prints a `dfg::PullAsMemRefOp` in the custom assembly syntax
///        consumed by `parse`.
///
/// Emits `%readPort : portType`; the memref result type is implicit.
void PullAsMemRefOp::print(OpAsmPrinter &p)
{ p << " " << getReadPort() << " : " << getReadPort().getType(); }

/// @brief Verifies that the source port is shaped.
///
/// Mirrors `PullAsTensorOp::verify`; pulling a scalar port as a memref is
/// rejected as a hard error.
LogicalResult PullAsMemRefOp::verify()
{
    // Check if read port is shaped
    if (getReadPort().getType().getShape().empty())
        return emitOpError("Cannot pull as tensor from a scalar read port.");
    return success();
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `dfg::PushOp` that writes a token to a shaped or scalar
///        port.
///
/// @param token      The value to write; must match the port's element type.
/// @param writePort  The `dfg::InputType` port to write to.
/// @param indices    Element indices for shaped ports; must be empty for
///                   scalar ports.
void PushOp::build(
    OpBuilder &,
    OperationState &state,
    Value token,
    Value writePort,
    ValueRange indices)
{
    state.addOperands(token);
    state.addOperands(writePort);
    if (!indices.empty()) state.addOperands(indices);
}

/// @brief Parses a `dfg::PushOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.push` %token `to` %writePort (`[` index-list `]`)? `:` port-type
/// @endcode
///
/// The `to` keyword separates the written value from the destination port.
/// Index operands, if present, address a specific element of a shaped port.
/// The token type is resolved as the element type of the port and is not
/// written explicitly.
ParseResult PushOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse token
    OpAsmParser::UnresolvedOperand token;
    if (parser.parseOperand(token)) return failure();

    // Parse write port
    OpAsmParser::UnresolvedOperand writePort;
    if (parser.parseKeyword("to") || parser.parseOperand(writePort))
        return failure();

    // Indices if port is shaped
    SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
    if (succeeded(parser.parseOptionalLSquare())) {
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand idx;
                if (parser.parseOperand(idx)) return failure();
                indices.push_back(idx);
                return success();
            })
            || parser.parseRSquare())
            return failure();
    }

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();

    // Resolve operands
    if (parser.resolveOperand(
            token,
            cast<InputType>(portType).getElementType(),
            state.operands))
        return failure();
    if (parser.resolveOperand(writePort, portType, state.operands))
        return failure();
    if (!indices.empty()) {
        for (auto idx : indices)
            if (parser.resolveOperand(
                    idx,
                    IndexType::get(parser.getContext()),
                    state.operands))
                return failure();
    }

    return success();
}

/// @brief Prints a `dfg::PushOp` in the custom assembly syntax consumed by
///        `parse`.
///
/// Emits `%token to %writePort[idx, ...] : portType`, omitting the bracket
/// list when no indices are present.
void PushOp::print(OpAsmPrinter &p)
{
    p << " " << getToken() << " to " << getWritePort();
    auto indices = getIndices();
    if (!indices.empty()) {
        p << "[";
        for (size_t i = 0; i < indices.size(); ++i) {
            p << indices[i];
            if (i < indices.size() - 1) p << ", ";
        }
        p << "]";
    }
    p << " : " << getWritePort().getType();
}

/// @brief Verifies that the index list is consistent with the port shape.
///
/// Mirrors `PullOp::verify`: scalar ports must have no indices, and shaped
/// ports must have exactly as many indices as the port rank.
LogicalResult PushOp::verify()
{
    auto portType = cast<InputType>(getWritePort().getType());
    auto shape = portType.getShape();
    auto indices = getIndices();

    if (shape.empty() && !indices.empty())
        return emitOpError("Indices provided but read port has no shape");

    if (!shape.empty() && indices.size() != shape.size())
        return emitOpError("Expected ") << shape.size()
                                        << " index operand(s) to match the "
                                           "rank of the read port, but got "
                                        << indices.size();

    return success();
}

//===----------------------------------------------------------------------===//
// PushTensorOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `dfg::PushTensorOp` that writes a ranked tensor to a
///        shaped port.
///
/// @param tokenTensor  The ranked-tensor value to write.
/// @param writePort    The shaped `dfg::InputType` port to write to.
void PushTensorOp::build(
    OpBuilder &,
    OperationState &state,
    Value tokenTensor,
    Value writePort)
{
    state.addOperands(tokenTensor);
    state.addOperands(writePort);
}

/// @brief Parses a `dfg::PushTensorOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.push_tensor` %tokenTensor `to` %writePort `:` port-type
/// @endcode
///
/// The tensor token type is resolved as a `RankedTensorType` derived from
/// the port's shape and element type and is not written explicitly.
ParseResult PushTensorOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse token
    OpAsmParser::UnresolvedOperand token;
    if (parser.parseOperand(token)) return failure();

    // Parse write port
    OpAsmParser::UnresolvedOperand writePort;
    if (parser.parseKeyword("to") || parser.parseOperand(writePort))
        return failure();

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();

    // Resolve operands
    auto portDFGType = cast<InputType>(portType);
    if (parser.resolveOperand(
            token,
            RankedTensorType::get(
                portDFGType.getShape(),
                portDFGType.getElementType()),
            state.operands))
        return failure();
    if (parser.resolveOperand(writePort, portType, state.operands))
        return failure();

    return success();
}

/// @brief Prints a `dfg::PushTensorOp` in the custom assembly syntax
///        consumed by `parse`.
///
/// Emits `%tokenTensor to %writePort : portType`.
void PushTensorOp::print(OpAsmPrinter &p)
{
    p << " " << getTokenTensor() << " to " << getWritePort() << " : "
      << getWritePort().getType();
}

//===----------------------------------------------------------------------===//
// PushMemRefOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `dfg::PushMemRefOp` that writes a ranked memref to a
///        shaped port.
///
/// @param tokenMemref  The ranked-memref value to write.
/// @param writePort    The shaped `dfg::InputType` port to write to.
void PushMemRefOp::build(
    OpBuilder &,
    OperationState &state,
    Value tokenMemref,
    Value writePort)
{
    state.addOperands(tokenMemref);
    state.addOperands(writePort);
}

/// @brief Parses a `dfg::PushMemRefOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `dfg.push_memref` %tokenMemref `to` %writePort `:` port-type
/// @endcode
///
/// The memref token type is resolved as a `MemRefType` derived from the
/// port's shape and element type and is not written explicitly.
ParseResult PushMemRefOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse token
    OpAsmParser::UnresolvedOperand token;
    if (parser.parseOperand(token)) return failure();

    // Parse write port
    OpAsmParser::UnresolvedOperand writePort;
    if (parser.parseKeyword("to") || parser.parseOperand(writePort))
        return failure();

    // Parse port type
    Type portType;
    if (parser.parseColon() || parser.parseType(portType)) return failure();

    // Resolve operands
    auto portDFGType = cast<InputType>(portType);
    if (parser.resolveOperand(
            token,
            MemRefType::get(
                portDFGType.getShape(),
                portDFGType.getElementType()),
            state.operands))
        return failure();
    if (parser.resolveOperand(writePort, portType, state.operands))
        return failure();

    return success();
}

/// @brief Prints a `dfg::PushMemRefOp` in the custom assembly syntax
///        consumed by `parse`.
///
/// Emits `%tokenMemref to %writePort : portType`.
void PushMemRefOp::print(OpAsmPrinter &p)
{
    p << " " << getTokenMemref() << " to " << getWritePort() << " : "
      << getWritePort().getType();
}

//===----------------------------------------------------------------------===//
// DFGDialect
//===----------------------------------------------------------------------===//

void DFGDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.cpp.inc"
        >();
}
