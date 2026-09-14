/// Implements the EmitHLS dialect ops.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"

#include "laksa-mlir/Dialect/EmitHLS/EmitHLSEnums.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::emithls;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.cpp.inc"

//===----------------------------------------------------------------------===//

// For multiple variadic attributes
constexpr char kOperandSegmentSizesAttr[] = "operandSegmentSizes";

//===----------------------------------------------------------------------===//
// Structure operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::IncludeOp` with the given header string.
///
/// @param header   The header file name (e.g. `"ap_int.h"`).
void IncludeOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef header)
{
    state.addAttribute(
        getHeaderAttrName(state.name),
        builder.getStringAttr(header));
}

/// @brief Parses `emithls::IncludeOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.include` `"`header`"`
/// @endcode
ParseResult IncludeOp::parse(OpAsmParser &parser, OperationState &state)
{
    StringAttr includeStr;
    OptionalParseResult includeParseResult = parser.parseOptionalAttribute(
        includeStr,
        getHeaderAttrName(state.name),
        state.attributes);
    if (!includeParseResult.has_value())
        return parser.emitError(parser.getNameLoc())
               << "expected string attribute";

    return success();
}

/// @brief Prints an `emithls::IncludeOp` in the custom assembly syntax consumed
/// by `parse`.
void IncludeOp::print(OpAsmPrinter &p) { p << " \"" << getHeader() << "\""; }

/// @brief Verifies an `emithls::IncludeOp` has only `ModuleOp` as parent.
LogicalResult IncludeOp::verify()
{
    auto parent = getOperation()->getParentOp();
    if (!isa<ModuleOp>(parent))
        return emitOpError(
            "An IncludeOp must only have ModuleOp as its parent.");
    return success();
}

namespace {
struct RemoveSameInclude final : public OpRewritePattern<IncludeOp> {
    using OpRewritePattern<IncludeOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(IncludeOp op, PatternRewriter &rewriter) const override
    {
        auto moduleOp = op->getParentOfType<ModuleOp>();
        for (auto &bodyOp : moduleOp.getBody()->getOperations()) {
            if (&bodyOp == op.getOperation()) return failure();
            if (auto prevInclude = dyn_cast<IncludeOp>(&bodyOp))
                if (prevInclude.getHeader() == op.getHeader()) {
                    rewriter.eraseOp(op);
                    return success();
                }
        }
        return failure();
    }
};
struct HoistIncludeToTop final : public OpRewritePattern<IncludeOp> {
    using OpRewritePattern<IncludeOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(IncludeOp op, PatternRewriter &rewriter) const override
    {
        auto moduleOp = op->getParentOfType<ModuleOp>();
        Block* moduleBody = moduleOp.getBody();

        Operation* firstNonInclude = nullptr;
        for (auto &opInModule : moduleBody->getOperations()) {
            if (!isa<IncludeOp>(&opInModule)) {
                firstNonInclude = &opInModule;
                break;
            }
        }

        // If there is no other operation
        if (!firstNonInclude) return failure();

        for (auto &opInModule : moduleBody->getOperations()) {
            if (&opInModule == op.getOperation()) return failure();
            if (&opInModule == firstNonInclude) break;
        }

        rewriter.moveOpBefore(op, firstNonInclude);
        return success();
    }
};
struct ReorderIncludes final : public OpRewritePattern<IncludeOp> {
    using OpRewritePattern<IncludeOp>::OpRewritePattern;
    LogicalResult
    matchAndRewrite(IncludeOp op, PatternRewriter &rewriter) const override
    {
        auto moduleOp = op->getParentOfType<ModuleOp>();

        SmallVector<IncludeOp> includes(
            moduleOp.getBody()->getOps<IncludeOp>());

        if (includes.size() <= 1) return failure();

        auto currentIdx =
            std::distance(includes.begin(), llvm::find(includes, op));
        auto sortedIdx = llvm::count_if(includes, [&](IncludeOp includeOp) {
            return includeOp != op && includeOp.getHeader() < op.getHeader();
        });

        if (currentIdx == sortedIdx)
            return failure();
        else if (currentIdx < sortedIdx)
            rewriter.moveOpAfter(op, includes[sortedIdx]);
        else
            rewriter.moveOpBefore(op, includes[sortedIdx]);

        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for `IncludeOp`.
///
/// - RemoveSameInclude: removes the `IncludeOp` with the same header string as
/// one defined before.
/// - HoistIncludeToTop: moves all `IncludeOp` within `ModuleOp` to the top.
/// - ReorderIncludes: reorders all `IncludeOp` alphabetically in increasing
/// order.
void IncludeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<RemoveSameInclude>(context);
    patterns.add<HoistIncludeToTop>(context);
    patterns.add<ReorderIncludes>(context);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// @brief Creates a `FuncOp` with only @p name and @p functionType.
void FuncOp::build(
    OpBuilder &builder,
    OperationState &state,
    StringRef name,
    FunctionType functionType)
{ build(builder, state, name, functionType, ArrayAttr{}, ArrayAttr{}); }

/// @brief Parses an `emithls::FuncOp` from its assembly syntax.
///
/// Delegates to `function_interface_impl::parseFunctionOp`, building a
/// non-variadic `FunctionType` from the parsed argument and result types.
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &state)
{
    auto buildFuncType = [](Builder &builder,
                            ArrayRef<Type> argTypes,
                            ArrayRef<Type> results,
                            function_interface_impl::VariadicFlag,
                            std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    return function_interface_impl::parseFunctionOp(
        parser,
        state,
        /*allowVariadic=*/false,
        getFunctionTypeAttrName(state.name),
        buildFuncType,
        getArgAttrsAttrName(state.name),
        getResAttrsAttrName(state.name));
}

/// @brief Prints an `emithls::FuncOp` in the assembly syntax consumed by
/// `parse`.
void FuncOp::print(OpAsmPrinter &p)
{
    function_interface_impl::printFunctionOp(
        p,
        *this,
        /*isVariadic=*/false,
        getFunctionTypeAttrName(),
        getArgAttrsAttrName(),
        getResAttrsAttrName());
}

/// @brief Verifies an `emithls::FuncOp` has only supported argument types.
///
/// Each argument must be one of:
/// - `StreamType` or `PointerType` (passed directly), or
/// - `ArrayType` whose element type is not a `PointerType`.
LogicalResult FuncOp::verify()
{
    auto funcTy = getFunctionType();
    if (funcTy.getNumInputs() != 0) {
        for (auto type : funcTy.getInputs())
            if (auto arrayType = dyn_cast<ArrayType>(type)) {
                auto arrayElemType = arrayType.getElementType();
                if (isa<PointerType>(arrayElemType))
                    return emitOpError("Array of pointer type is not allowed.");
            } else if (!isa<StreamType, PointerType>(type)) {
                return emitOpError("Unsupported type ")
                       << type
                       << ", needs to be stream/pointer/array type for "
                          "function arguments.";
            }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::CallOp` targeting @p callee with @p operands.
///
/// @param callee    The `FuncOp` to call; its name is stored as a
///                  `SymbolRefAttr` and its result types become this op's
///                  result types.
/// @param operands  The arguments forwarded to the callee.
void CallOp::build(
    OpBuilder &,
    OperationState &state,
    FuncOp callee,
    ValueRange operands)
{
    state.addOperands(operands);
    state.addAttribute(
        getCalleeAttrName(state.name),
        SymbolRefAttr::get(callee));
    state.addTypes(callee.getFunctionType().getResults());
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

/// @brief Names the SSA result of a `VariableOp`.
///
/// Assigns a human-readable name of the form `<prefix><type>_<count>` to
/// the variable result so that printed IR reads, e.g., `%var_int32_0` or
/// `%const_array_1` instead of the default `%0`, `%1`, … scheme.  The
/// prefix is `"const"` for constant variables and `"var"` for mutable ones.
/// The count is the zero-based index among sibling variables of the same
/// type class and const-ness that appear before this op in the parent
/// function's entry block.
void VariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    auto thisOp = getOperation();

    // Count sibling variables within the nearest enclosing FuncOp when there
    // is one; otherwise fall back to the nearest ancestor op that sits
    // directly under the module.
    Operation* scopeOp = thisOp->getParentOfType<FuncOp>();
    if (!scopeOp) {
        scopeOp = thisOp;
        while (Operation* parent = scopeOp->getParentOp()) {
            if (isa<ModuleOp>(parent)) break;
            scopeOp = parent;
        }
        if (scopeOp == thisOp || scopeOp->getNumRegions() == 0) return;
    }

    auto resultTy = getType();
    unsigned count = 0;
    for (auto &opi : scopeOp->getRegion(0).front()) {
        if (auto varOp = dyn_cast<VariableOp>(opi)) {
            if (varOp.getOperation() == thisOp) break;
            if (((varOp.getType() == resultTy)
                 || (isa<ArrayType>(varOp.getType())
                     && isa<ArrayType>(resultTy)))
                && (varOp.getIsConst() == getIsConst()))
                ++count;
        }
    }

    std::string prefix = getIsConst() ? "const" : "var";
    std::string suffix;
    if (isa<IntegerType>(resultTy))
        suffix =
            "_int" + std::to_string(resultTy.getIntOrFloatBitWidth()) + "_";
    else if (isa<IndexType>(resultTy))
        suffix = "_index_";
    else if (isa<ArrayType>(resultTy))
        suffix = "_array_";
    else if (isa<StreamType>(resultTy))
        suffix = "_stream_";
    setNameFn(getVariable(), prefix + suffix + std::to_string(count));
}

/// @brief Builds a `VariableOp` whose initial value is a compile-time
/// attribute.
///
/// @param type        The type of the variable.
/// @param initNumber  Optional constant attribute used as the initial value;
///                    pass a null `Attribute` to leave the variable
///                    uninitialized.
/// @param is_const    Whether this variable is read-only (a HLS constant).
void VariableOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type type,
    Attribute initNumber,
    bool is_const)
{
    state.addTypes(type);
    if (initNumber)
        state.addAttribute(getInitNumberAttrName(state.name), initNumber);
    if (is_const) state.addAttribute("is_const", builder.getUnitAttr());
}

/// @brief Builds a `VariableOp` whose initial value is an SSA value.
///
/// @param type       The type of the variable.
/// @param initValue  Optional SSA value used to initialise the variable at
///                   runtime; pass a null `Value` to leave it uninitialized.
/// @param is_const   Whether this variable is read-only (a HLS constant).
void VariableOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type type,
    Value initValue,
    bool is_const)
{
    state.addTypes(type);
    if (initValue) state.addOperands(initValue);
    if (is_const) state.addAttribute("is_const", builder.getUnitAttr());
}

/// @brief Parses an `emithls::VariableOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.variable` `as` [`const`] type [`=` (ssa-value | attribute)]
/// @endcode
///
/// The `const` keyword marks read-only variables.  The optional `= <init>`
/// clause sets an initial value: either an SSA operand or a literal attribute.
/// A `const` variable without an initializer is rejected as ill-formed.
ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &state)
{
    if (failed(parser.parseKeyword("as")))
        return parser.emitError(
            parser.getCurrentLocation(),
            "expected keyword 'as'");

    // If it's const value
    bool isConst = false;
    if (succeeded(parser.parseOptionalKeyword("const"))) {
        state.addAttribute("is_const", parser.getBuilder().getUnitAttr());
        isConst = true;
    }

    // Parse type
    Type type;
    if (failed(parser.parseType(type)))
        return parser.emitError(parser.getCurrentLocation(), "expected type");
    state.addTypes(type);

    Attribute initNumber;
    OpAsmParser::UnresolvedOperand initValue;

    if (succeeded(parser.parseOptionalEqual())) {
        auto operandParseResult = parser.parseOptionalOperand(initValue);
        if (operandParseResult.has_value()) {
            // If there is an initial value
            if (failed(*operandParseResult)) return failure();
            if (failed(parser.resolveOperand(initValue, type, state.operands)))
                return failure();
        } else if (
            failed(parser.parseAttribute(
                initNumber,
                type,
                getInitNumberAttrName(state.name).data(),
                state.attributes))) {
            // If there is an initial attribute
            return parser.emitError(
                parser.getCurrentLocation(),
                "expected initial attribute");
        }
    } else {
        if (isConst)
            return parser.emitError(
                parser.getCurrentLocation(),
                "const value must have initial attribute");
    }

    if (failed(parser.parseOptionalAttrDict(state.attributes)))
        return failure();

    return success();
}

/// @brief Prints an `emithls::VariableOp` in the custom assembly syntax
///        consumed by `parse`.
void VariableOp::print(OpAsmPrinter &p)
{
    p << " as ";
    if (getIsConst()) p << "const ";
    p << getType();
    if (getInitNumber()) {
        p << " = ";
        p.printAttributeWithoutType(getInitNumberAttr());
    } else if (getInitValue()) {
        p << " = " << getInitValue();
    }
    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getIsConstAttrName(), getInitNumberAttrName()});
}

/// @brief Verifies an `emithls::VariableOp` has a supported type and
///        consistent initialization.
///
/// - The op must be nested (at any depth) inside an `emithls::FuncOp`, e.g.
///   directly in the function body or inside a `ForOp`/`IfOp`/`ExpressionOp`
///   region within it.
/// - Pointer types are rejected (dynamic memory is unsupported in HLS).
/// - A literal `initNumber` attribute is only allowed for integer, index,
///   float, and array types.
LogicalResult VariableOp::verify()
{
    auto resultType = getType();
    if (isa<PointerType>(resultType))
        return emitOpError("Dynamic memory usage is not supported in HLS.");
    if (getInitNumber()
        && !isa<IntegerType, IndexType, FloatType, ArrayType>(resultType))
        return emitOpError("Unsupported type to have init value.");
    return success();
}

namespace {
struct RemoveUnusedVariable final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &rewriter) const override
    {
        if (op.getVariable().use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

struct RemoveSameConstVariable final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &rewriter) const override
    {
        if (!op.getIsConst()) return failure();

        auto initAttr = op.getInitNumberAttr();
        auto constTy = op.getType();

        // Dedupe within the nearest enclosing FuncOp when there is one;
        // otherwise fall back to the nearest ancestor op that sits directly
        // under the module.
        Operation* scopeOp = op->getParentOfType<FuncOp>();
        if (!scopeOp) {
            scopeOp = op;
            while (Operation* parent = scopeOp->getParentOp()) {
                if (isa<ModuleOp>(parent)) break;
                scopeOp = parent;
            }
            if (scopeOp == op.getOperation()) return failure();
        }

        VariableOp candidate;
        scopeOp->walk([&](VariableOp varOp) -> WalkResult {
            if (varOp == op) return WalkResult::interrupt();
            if (varOp.getIsConst() && varOp->getParentOp() == scopeOp
                && varOp.getInitNumberAttr() == initAttr
                && varOp.getType() == constTy)
                candidate = varOp;
            return WalkResult::advance();
        });

        if (!candidate) return failure();
        op.getResult().replaceAllUsesWith(candidate.getResult());
        rewriter.eraseOp(op);
        return success();
    }
};

struct HoistToFunctionTop final : public OpRewritePattern<VariableOp> {
    using OpRewritePattern<VariableOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(VariableOp op, PatternRewriter &) const override
    {
        if (!op.getIsConst()) return failure();

        if (auto parentFunc = op->getParentOfType<FuncOp>()) {
            if (isa<FuncOp>(op->getParentOp())) return failure();
            auto &entryBlock = parentFunc.getRegion().front();
            op->moveBefore(&entryBlock, entryBlock.begin());
            return success();
        }

        // Hoist to the top of the nearest ancestor op that sits directly under
        // the module
        Operation* topLevelOp = op;
        while (Operation* parent = topLevelOp->getParentOp()) {
            if (isa<ModuleOp>(parent)) break;
            topLevelOp = parent;
        }
        if (topLevelOp == op.getOperation() || topLevelOp->getNumRegions() == 0)
            return failure();
        if (op->getParentOp() == topLevelOp) return failure();

        auto &entryBlock = topLevelOp->getRegion(0).front();
        op->moveBefore(&entryBlock, entryBlock.begin());
        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for `VariableOp`.
///
/// - RemoveUnusedVariable: erases any `VariableOp` whose result has no uses.
/// - RemoveSameConstVariable: deduplicates const variables with identical type
///   and init attribute, replacing later ones with the first occurrence.
/// - HoistToFunctionTop: moves const variables nested inside control-flow
///   blocks to the top of the enclosing `FuncOp` entry block.
void VariableOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{
    patterns.add<RemoveUnusedVariable>(context);
    patterns.add<RemoveSameConstVariable>(context);
    patterns.add<HoistToFunctionTop>(context);
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::UpdateOp`.
///
/// @param variable          The variable (or array) to update.
/// @param newValue          The value (or array) to write into @p variable.
/// @param indices           Index operands into @p variable; empty for scalar
///                          assignment.
/// @param newValueIndices   Index operands into @p newValue; empty when
///                          @p newValue is used as a whole.
void UpdateOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value variable,
    Value newValue,
    SmallVector<Value> indices,
    SmallVector<Value> newValueIndices)
{
    state.addOperands(variable);
    if (!indices.empty()) state.addOperands(indices);
    state.addOperands(newValue);
    if (!newValueIndices.empty()) state.addOperands(newValueIndices);
    auto operandSegmentSizes =
        builder.getDenseI32ArrayAttr({/*variable*/ 1,
                                      int32_t(indices.size()),
                                      /*newValue*/ 1,
                                      int32_t(newValueIndices.size())});
    state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);
}

/// @brief Parses an `emithls::UpdateOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.update` variable [`[` index-list `]`]
///     `with` new-value [`[` index-list `]`]
///     `:` variable-type `<-` new-value-type
/// @endcode
///
/// Both index lists are optional; when absent the assignment targets or
/// sources the variable/value as a whole.  The `operandSegmentSizes`
/// attribute is derived from the parsed operand counts and stored
/// automatically.
ParseResult UpdateOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand variable, newValue;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> indices, newValueIndices;

    if (parser.parseOperand(variable)) return failure();
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

    if (parser.parseKeyword("with") || parser.parseOperand(newValue))
        return failure();
    if (succeeded(parser.parseOptionalLSquare())) {
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand idx;
                if (parser.parseOperand(idx)) return failure();
                newValueIndices.push_back(idx);
                return success();
            })
            || parser.parseRSquare())
            return failure();
    }

    Type variableTy, newValueTy;
    if (parser.parseColon() || parser.parseType(variableTy)
        || parser.parseLess() || parser.parseMinus()
        || parser.parseType(newValueTy))
        return failure();

    if (parser.resolveOperand(variable, variableTy, state.operands))
        return failure();
    if (!indices.empty()) {
        for (auto idx : indices)
            if (parser.resolveOperand(
                    idx,
                    parser.getBuilder().getIndexType(),
                    state.operands))
                return failure();
    }
    if (parser.resolveOperand(newValue, newValueTy, state.operands))
        return failure();
    if (!newValueIndices.empty()) {
        for (auto idx : newValueIndices)
            if (parser.resolveOperand(
                    idx,
                    parser.getBuilder().getIndexType(),
                    state.operands))
                return failure();
    }

    auto operandSegmentSizes = parser.getBuilder().getDenseI32ArrayAttr(
        {/*variable*/ 1,
         int32_t(indices.size()),
         /*newValue*/ 1,
         int32_t(newValueIndices.size())});
    state.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

    return success();
}

/// @brief Prints an `emithls::UpdateOp` in the custom assembly syntax
///        consumed by `parse`.
void UpdateOp::print(OpAsmPrinter &p)
{
    p << " " << getVariable();
    auto indices = getIndices();
    if (!indices.empty()) {
        p << "[";
        for (size_t i = 0; i < indices.size(); ++i) {
            p << indices[i];
            if (i < indices.size() - 1) p << ", ";
        }
        p << "]";
    }
    p << " with " << getNewValue();
    auto newValueIndices = getNewValueIndices();
    if (!newValueIndices.empty()) {
        p << "[";
        for (size_t i = 0; i < newValueIndices.size(); ++i) {
            p << newValueIndices[i];
            if (i < newValueIndices.size() - 1) p << ", ";
        }
        p << "]";
    }
    p << " : " << getVariable().getType() << " <- " << getNewValue().getType();
}

/// @brief Verifies an `emithls::UpdateOp` has matching element types.
///
/// When indices are present the element type of the shaped variable (or new
/// value) is used for comparison; otherwise the operand type itself is used.
/// The resolved types of @p variable and @p newValue must be equal.
LogicalResult UpdateOp::verify()
{
    auto variable = getVariable();
    Type varTy =
        getIndices().empty()
            ? variable.getType()
            : dyn_cast<ShapedType>(variable.getType()).getElementType();
    auto newValue = getNewValue();
    Type newValTy =
        getNewValueIndices().empty()
            ? newValue.getType()
            : dyn_cast<ShapedType>(newValue.getType()).getElementType();
    if (varTy != newValTy)
        return emitOpError("update value and variable types don't match");
    return success();
}

//===----------------------------------------------------------------------===//
// ExpressionOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::ExpressionOp` with an inline body.
///
/// Creates a single-block region, invokes @p bodyBuilder to populate it, and
/// registers @p resultType as the op's result type.
///
/// @param resultType   The type produced by this expression.
/// @param bodyBuilder  Callback that emits ops into the expression body;
///                     receives the current builder and the op's location.
void ExpressionOp::build(
    OpBuilder &builder,
    OperationState &state,
    Type resultType,
    function_ref<void(OpBuilder &, Location)> bodyBuilder)
{
    OpBuilder::InsertionGuard guard(builder);
    Region* bodyRegion = state.addRegion();
    builder.createBlock(bodyRegion);
    bodyBuilder(builder, state.location);
    state.addTypes(resultType);
}

/// @brief Names the SSA result of an `emithls::ExpressionOp`.
///
/// Assigns a name of the form `expr<N>` (e.g. `%expr0`, `%expr2`) by counting
/// how many `ExpressionOp`s appear before this one in a pre-order walk of the
/// nearest enclosing non-module ancestor, so names are stable within a function
/// body regardless of nesting depth.
void ExpressionOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* topLevel = self;
    while (Operation* p = topLevel->getParentOp()) {
        if (isa<ModuleOp>(p)) break;
        topLevel = p;
    }

    unsigned count = 0;
    topLevel->walk([&](ExpressionOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getResult(), "expr" + std::to_string(count));
}

/// @brief Verifies an `emithls::ExpressionOp` has a consistent result type.
///
/// The type of the enclosing `ExpressionOp` must match the type of the value
/// produced by the terminating `YieldOp` in its body.
LogicalResult ExpressionOp::verify()
{
    auto yieldOp = cast<YieldOp>((&getBody().front())->getTerminator());
    if (getType() != yieldOp.getValue().getType())
        return emitOpError(
            "Expression return a different type from yield value.");
    return success();
}

namespace {
struct InlinePureNestedExpression final
        : public OpRewritePattern<ExpressionOp> {
    using OpRewritePattern<ExpressionOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ExpressionOp op, PatternRewriter &rewriter) const override
    {
        Block* block = &op.getBody().front();

        // Match: body must be exactly an ExpressionOp followed by a YieldOp.
        if (std::ranges::distance(block->getOperations()) != 2)
            return failure();
        auto innerExpr = dyn_cast<ExpressionOp>(block->front());
        if (!innerExpr) return failure();

        // Splice the inner expression's body (including its YieldOp) before it,
        // then erase the now-redundant wrapper and the outer YieldOp.
        rewriter.setInsertionPointToStart(block);
        IRMapping mapper;
        for (auto &opi : innerExpr.getBody().front())
            rewriter.clone(opi, mapper);
        rewriter.eraseOp(&block->back()); // outer YieldOp
        rewriter.eraseOp(innerExpr);
        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for `ExpressionOp`.
///
/// - InlinePureNestedExpression: flattens an `ExpressionOp` whose body
///   consists solely of another `ExpressionOp` followed by a `YieldOp` by
///   splicing the inner body into the outer block in place.
void ExpressionOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{ patterns.add<InlinePureNestedExpression>(context); }

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::YieldOp` that yields a single value.
///
/// @param value  The SSA value to return from the enclosing `ExpressionOp`.
void YieldOp::build(OpBuilder &, OperationState &state, Value value)
{ state.addOperands(value); }

/// @brief Builds an `emithls::YieldOp` with no operands.
///
/// Used by `SingleBlockImplicitTerminator` to auto-insert a terminator into
/// an `ExpressionOp` body when none is provided explicitly.
void YieldOp::build(OpBuilder &, OperationState &)
{
    // Do nothing
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::IfOp` with builder callbacks for its regions.
///
/// Creates the then-region unconditionally and, if @p elseBuilder is non-null,
/// also creates and populates the else-region.
///
/// @param condition    The `i1` condition value.
/// @param thenBuilder  Callback that populates the then-region body.
/// @param elseBuilder  Optional callback for the else-region; pass `nullptr`
///                     to omit the else branch.
void IfOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value condition,
    function_ref<void(OpBuilder &, Location)> thenBuilder,
    function_ref<void(OpBuilder &, Location)> elseBuilder)
{
    state.addOperands(condition);

    OpBuilder::InsertionGuard guard(builder);
    Region* thenRegion = state.addRegion();
    builder.createBlock(thenRegion);
    thenBuilder(builder, state.location);
    Region* elseRegion = state.addRegion();
    if (elseBuilder) {
        builder.createBlock(elseRegion);
        elseBuilder(builder, state.location);
    }
}

/// @brief Builds an `emithls::IfOp` with an optionally pre-allocated else
/// block.
///
/// Always creates the then-region block.  When @p ensureElseBlock is true an
/// empty else-region block is also created, ready to be populated later.
///
/// @param condition        The `i1` condition value.
/// @param ensureElseBlock  If true, an empty else block is pre-created.
void IfOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value condition,
    bool ensureElseBlock)
{
    state.addOperands(condition);

    OpBuilder::InsertionGuard guard(builder);
    Region* thenRegion = state.addRegion();
    builder.createBlock(thenRegion);
    Region* elseRegion = state.addRegion();
    if (ensureElseBlock) builder.createBlock(elseRegion);
}

/// @brief Parses an `emithls::IfOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.if` condition-value then-region [`else` else-region]
/// @endcode
///
/// The condition must be of `i1` type.  The `else` keyword and region are
/// optional; when absent the false path is a no-op.
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &state)
{
    state.regions.reserve(2);
    Region* thenRegion = state.addRegion();
    Region* elseRegion = state.addRegion();

    OpAsmParser::UnresolvedOperand condition;
    if (parser.parseOperand(condition)
        || parser.resolveOperand(
            condition,
            parser.getBuilder().getI1Type(),
            state.operands))
        return failure();

    if (parser.parseRegion(*thenRegion, {}, {})) return failure();
    if (succeeded(parser.parseOptionalKeyword("else")))
        if (parser.parseRegion(*elseRegion, {}, {})) return failure();

    return success();
}

/// @brief Prints an `emithls::IfOp` in the custom assembly syntax consumed
///        by `parse`.
void IfOp::print(OpAsmPrinter &p)
{
    p << " " << getCondition() << " ";
    p.printRegion(getThenRegion(), false, false);
    auto &elseRegion = getElseRegion();
    if (!elseRegion.empty()) {
        p << " else ";
        p.printRegion(elseRegion, false, false);
    }
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::ForOp` with integer-literal bounds and step.
///
/// Creates the body region with a single block argument for the induction
/// variable and invokes @p bodyBuilder (if provided) to populate the body.
///
/// @param lowerBound   Inclusive lower bound of the iteration range.
/// @param upperBound   Exclusive upper bound of the iteration range.
/// @param step         Loop stride; must be positive and non-zero.
/// @param bodyBuilder  Optional callback that emits ops into the loop body.
void ForOp::build(
    OpBuilder &builder,
    OperationState &state,
    int64_t lowerBound,
    int64_t upperBound,
    int64_t step,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder)
{
    state.addAttribute(
        getLowerBoundAttrName(state.name),
        builder.getIntegerAttr(builder.getIndexType(), lowerBound));
    state.addAttribute(
        getUpperBoundAttrName(state.name),
        builder.getIntegerAttr(builder.getIndexType(), upperBound));
    state.addAttribute(
        getStepAttrName(state.name),
        builder.getIntegerAttr(builder.getIndexType(), step));

    OpBuilder::InsertionGuard guard(builder);
    Region* bodyRegion = state.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(builder.getIndexType(), state.location);
    if (bodyBuilder)
        bodyBuilder(builder, state.location, bodyBlock->getArguments());
}

/// @brief Builds an `emithls::ForOp` from pre-constructed `Attribute` bounds.
///
/// Equivalent to the integer overload but accepts pre-built `IndexAttr` values,
/// useful when the attributes have already been created.
void ForOp::build(
    OpBuilder &builder,
    OperationState &state,
    Attribute lowerBound,
    Attribute upperBound,
    Attribute step,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder)
{
    state.addAttribute(getLowerBoundAttrName(state.name), lowerBound);
    state.addAttribute(getUpperBoundAttrName(state.name), upperBound);
    state.addAttribute(getStepAttrName(state.name), step);

    OpBuilder::InsertionGuard g(builder);
    Region* bodyRegion = state.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(builder.getIndexType(), state.location);
    if (bodyBuilder)
        bodyBuilder(builder, state.location, bodyBlock->getArguments());
}

/// @brief Names the induction-variable block argument.
///
/// Assigns a name of the form `idx<N>` where `N` is the zero-based nesting
/// depth, so the outermost loop uses `%idx0`, its immediate child `%idx1`,
/// and so on.
void ForOp::getAsmBlockArgumentNames(
    Region &region,
    OpAsmSetValueNameFn setNameFn)
{
    Operation* forOp = getOperation();
    unsigned nestedLevel = 0;

    while ((forOp = forOp->getParentOp()) && forOp)
        if (isa<ForOp>(forOp)) ++nestedLevel;

    setNameFn(region.getArgument(0), "idx" + std::to_string(nestedLevel));
}

/// @brief Parses an `emithls::ForOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.for` induction-var `=` lower-bound `to` upper-bound
///                                   `step` step body-region
/// @endcode
///
/// All bounds and the step are `index`-typed integer attributes.
/// The induction variable is added as a block argument to the body region.
ParseResult ForOp::parse(OpAsmParser &parser, OperationState &state)
{
    auto &builder = parser.getBuilder();
    Type indexType = builder.getIndexType();

    OpAsmParser::Argument inductionVariable;
    if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual())
        return failure();

    IntegerAttr lbAttr, ubAttr, stepAttr;
    if (parser.parseAttribute(
            lbAttr,
            indexType,
            getLowerBoundAttrName(state.name).data(),
            state.attributes)
        || parser.parseKeyword("to")
        || parser.parseAttribute(
            ubAttr,
            indexType,
            getUpperBoundAttrName(state.name).data(),
            state.attributes)
        || parser.parseKeyword("step")
        || parser.parseAttribute(
            stepAttr,
            indexType,
            getStepAttrName(state.name).data(),
            state.attributes))
        return failure();

    SmallVector<OpAsmParser::Argument> blkArgs;
    blkArgs.push_back(inductionVariable);
    blkArgs.front().type = indexType;

    Region* body = state.addRegion();
    if (parser.parseRegion(*body, blkArgs)) return failure();

    if (parser.parseOptionalAttrDict(state.attributes)) return failure();

    return success();
}

/// @brief Prints an `emithls::ForOp` in the custom assembly syntax consumed
///        by `parse`.
void ForOp::print(OpAsmPrinter &p)
{
    p << " " << getInductionVariable() << " = " << getLowerBound() << " to "
      << getUpperBound() << " step " << getStep();
    p << ' ';
    p.printRegion(
        getRegion(),
        /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getLowerBoundAttrName(), getUpperBoundAttrName(), getStepAttrName()});
}

/// @brief Verifies an `emithls::ForOp` has valid bounds and a non-empty body.
///
/// Rejects loops where:
/// - lower bound equals upper bound (zero iterations),
/// - step is zero,
/// - upper bound is unreachable from lower bound in at least one step, or
/// - the body region is empty.
LogicalResult ForOp::verify()
{
    if (getLowerBound() == getUpperBound())
        return emitOpError("expected loop iterate at least once");
    if (getStep() == 0) return emitOpError("expected non-zero step");
    if (getUpperBound().slt(getLowerBound() + getStep()))
        return emitOpError("unexpected index overflow");
    return success();
}

namespace {
struct RemoveTrivialLoop : public OpRewritePattern<ForOp> {
    using OpRewritePattern<ForOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ForOp op, PatternRewriter &rewriter) const override
    {
        if ((op.getLowerBound() + op.getStep()) != op.getUpperBound())
            return failure();

        Block &body = op.getRegion().front();
        bool hasPipeline = !body.empty() && isa<PragmaPipelineOp>(body.front());
        auto parent = dyn_cast<ForOp>(op->getParentOp());
        bool siblingHasLoop =
            parent
            && llvm::any_of(parent.getBody().front(), [&](Operation &sibling) {
                   return &sibling != op.getOperation()
                          && sibling
                                 .walk([](ForOp) {
                                     return WalkResult::interrupt();
                                 })
                                 .wasInterrupted();
               });
        if (hasPipeline && (!parent || siblingHasLoop)) return failure();

        if (hasPipeline)
            rewriter.moveOpBefore(
                &body.front(),
                &parent.getBody().front(),
                parent.getBody().front().begin());

        rewriter.setInsertionPoint(op);
        IRMapping mapper;
        auto constVarOp = VariableOp::create(
            rewriter,
            op.getLoc(),
            rewriter.getIndexType(),
            op.getLowerBoundAttr(),
            true);
        mapper.map(op.getInductionVariable(), constVarOp.getResult());
        for (auto &opFor : op.getRegion().getOps())
            rewriter.clone(opFor, mapper);
        rewriter.eraseOp(op);
        return success();
    }
};
struct RemoveEmptyLoop : public OpRewritePattern<ForOp> {
    using OpRewritePattern<ForOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ForOp op, PatternRewriter &rewriter) const override
    {
        // Preserve loops whose body is not empty
        if (!op.getRegion().getOps().empty()) return failure();
        rewriter.eraseOp(op);
        return success();
    }
};
struct CollapseUnusedOuterLoop : public OpRewritePattern<ForOp> {
    using OpRewritePattern<ForOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ForOp op, PatternRewriter &rewriter) const override
    {
        // Only safe when NEITHER induction variable is read
        if (!op.getInductionVariable().use_empty()) return failure();

        Block &body = op.getRegion().front();
        if (body.empty() || &body.front() != &body.back()) return failure();
        auto inner = dyn_cast<ForOp>(body.front());
        if (!inner || !inner.getInductionVariable().use_empty())
            return failure();

        rewriter.setInsertionPoint(op);
        auto merged = ForOp::create(
            rewriter,
            op.getLoc(),
            0,
            op.getTripCount() * inner.getTripCount());
        // Move inner's body into merged
        rewriter.mergeBlocks(
            &inner.getRegion().front(),
            &merged.getRegion().front(),
            merged.getInductionVariable());
        rewriter.eraseOp(op);
        return success();
    }
};
} // namespace

/// @brief Canonicalization patterns for `ForOp`.
///
/// - RemoveTrivialLoop: inlines the body of a single-iteration loop, replacing
///   the induction variable with a constant, then erases the loop.
/// - CollapseUnusedOuterLoop: folds a loop whose induction variable is unused
///   and whose body is nothing but another loop into that inner loop's trip
///   count, eliminating the pointless wrapping level of nesting.
void ForOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{ results.add<RemoveTrivialLoop, CollapseUnusedOuterLoop>(context); }

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArithFusedOp
//===----------------------------------------------------------------------===//

/// @brief Verifies an `emithls::ArithFusedOp` has the correct accumulator
/// value, which must be defined as a non-constant variable
LogicalResult ArithFusedOp::verify()
{
    if (auto varOp = dyn_cast<VariableOp>(getAcc().getDefiningOp())) {
        if (varOp.getIsConst())
            return emitOpError("Accumulator cannot be a constant variable.");
    } else {
        return emitOpError("Accumulator of the fused op must be a variable.");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ArithDataRangeOp
//===----------------------------------------------------------------------===//

/// @brief Parses an `emithls::ArithDataRangeOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.arith_data_range` data `(` high-bit `,` low-bit `)`
///     `:` data-type `->` result-type
/// @endcode
ParseResult ArithDataRangeOp ::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand data, hBit, lBit;
    Type dataTy, outTy;
    Type indexTy = parser.getBuilder().getIndexType();
    if (parser.parseOperand(data) || parser.parseLParen()
        || parser.parseOperand(hBit) || parser.parseComma()
        || parser.parseOperand(lBit) || parser.parseRParen())
        return failure();
    if (parser.parseColon() || parser.parseType(dataTy) || parser.parseArrow()
        || parser.parseType(outTy))
        return failure();
    if (parser.resolveOperand(data, dataTy, state.operands)
        || parser.resolveOperand(hBit, indexTy, state.operands)
        || parser.resolveOperand(lBit, indexTy, state.operands))
        return failure();
    state.addTypes(outTy);
    return success();
}

/// @brief Prints an `emithls::ArithDataRangeOp` in the custom assembly syntax
///        consumed by `parse`.
void ArithDataRangeOp::print(OpAsmPrinter &p)
{
    p << " " << getData() << "(" << getHighBit() << ", " << getLowBit() << ")";
    p << " : " << getData().getType() << " -> " << getType();
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ArrayReadOp
//===----------------------------------------------------------------------===//

/// @brief Verifies an `emithls::ArrayReadOp` has the correct number of indices.
///
/// The number of index operands must equal the rank of the array type.
LogicalResult ArrayReadOp::verify()
{
    if (static_cast<int64_t>(getIndices().size()) != getArrayType().getRank()) {
        return emitOpError(
                   "incorrect number of indices for array read, expected ")
               << getArrayType().getRank() << " but got "
               << getIndices().size();
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ArrayWriteOp
//===----------------------------------------------------------------------===//

/// @brief Verifies an `emithls::ArrayWriteOp` has the correct number of
///        indices.
///
/// The number of index operands must equal the rank of the array type.
LogicalResult ArrayWriteOp::verify()
{
    if (static_cast<int64_t>(getIndices().size()) != getArrayType().getRank())
        return emitOpError(
            "array write index operand count not equal to memref rank");
    return success();
}

//===----------------------------------------------------------------------===//
// Stream operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StreamReadOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::StreamReadOp`.
///
/// @param resultType  The element type produced by the read.
/// @param stream      The stream (or array of streams) to read from.
/// @param indices     Index operands when @p stream is an array of streams;
///                    empty for a plain stream.
void StreamReadOp::build(
    OpBuilder &,
    OperationState &state,
    Type resultType,
    Value stream,
    ValueRange indices)
{
    state.addOperands(stream);
    if (!indices.empty()) state.addOperands(indices);
    state.addTypes(resultType);
}

/// @brief Verifies an `emithls::StreamReadOp` has consistent stream and result
///        types.
///
/// When the operand is an `ArrayType`, its element must be a `StreamType` and
/// the number of indices must match the array rank.  For a plain `StreamType`
/// no indices are allowed.  In both cases the stream's element type must equal
/// the op's result type.
LogicalResult StreamReadOp::verify()
{
    auto streamType = getStream().getType();
    auto indices = getIndices();
    Type elemType;

    if (auto arrayType = dyn_cast<ArrayType>(streamType)) {
        // If the stream is an array of streams
        auto arrayShape = arrayType.getShape();
        auto arrayElemType = arrayType.getElementType();
        if (!isa<StreamType>(arrayElemType))
            return emitOpError("expect array of streams");
        elemType = cast<StreamType>(arrayElemType).getElementType();
        // Then, check the indices
        if (indices.empty()) return emitOpError("expect indices");
        if (indices.size() != arrayShape.size())
            return emitOpError("indices and array shapes mismatch");
    } else {
        // Otherwise, it must be a stream type
        elemType = cast<StreamType>(streamType).getElementType();
        // Expect no indices
        if (!indices.empty()) return emitOpError("expect no indices");
    }

    if (elemType != getType()) emitOpError("result and stream types mismatch");

    return success();
}

//===----------------------------------------------------------------------===//
// StreamWriteOp
//===----------------------------------------------------------------------===//

/// @brief Builds an `emithls::StreamWriteOp`.
///
/// @param data     The value to write into the stream.
/// @param stream   The stream (or array of streams) to write to.
/// @param indices  Index operands when @p stream is an array of streams;
///                 empty for a plain stream.
void StreamWriteOp::build(
    OpBuilder &,
    OperationState &state,
    Value data,
    Value stream,
    ValueRange indices)
{
    state.addOperands(data);
    state.addOperands(stream);
    if (!indices.empty()) state.addOperands(indices);
}

/// @brief Verifies an `emithls::StreamWriteOp` has consistent stream and data
///        types.
///
/// When the operand is an `ArrayType`, its element must be a `StreamType` and
/// the number of indices must match the array rank.  For a plain `StreamType`
/// no indices are allowed.  In both cases the stream's element type must equal
/// the type of the data operand.
LogicalResult StreamWriteOp::verify()
{
    auto streamType = getStream().getType();
    auto indices = getIndices();
    Type elemType;

    if (auto arrayType = dyn_cast<ArrayType>(streamType)) {
        // If the stream is an array of streams
        auto arrayShape = arrayType.getShape();
        auto arrayElemType = arrayType.getElementType();
        if (!isa<StreamType>(arrayElemType))
            return emitOpError("expect array of streams");
        elemType = cast<StreamType>(arrayElemType).getElementType();
        // Then, check the indices
        if (indices.empty()) return emitOpError("expect indices");
        if (indices.size() != arrayShape.size())
            return emitOpError("indices and array shapes mismatch");
    } else {
        // Otherwise, it must be a stream type
        elemType = cast<StreamType>(streamType).getElementType();
        // Expect no indices
        if (!indices.empty()) return emitOpError("expect no indices");
    }

    if (elemType != getData().getType())
        return emitOpError("data and stream types mismatch");
    return success();
}

//===----------------------------------------------------------------------===//
// Pragma operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PragmaArrayPartitionOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaArrayPartitionOp`.
///
/// @param variable    The array variable to partition.
/// @param partType    The partition strategy (`cyclic`, `block`, or
/// `complete`).
/// @param partFactor  Partition factor; required for `cyclic` and `block`.
/// @param partDim     Dimension to partition; required for `cyclic` and
/// `block`.
void PragmaArrayPartitionOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value variable,
    ArrayPartitionType partType,
    std::optional<int32_t> partFactor,
    std::optional<int32_t> partDim)
{
    state.addOperands(variable);
    state.addAttribute(
        getPartTypeAttrName(state.name),
        ArrayPartitionTypeAttr::get(builder.getContext(), partType));
    if (partFactor.has_value()) {
        state.addAttribute(
            getPartFactorAttrName(state.name),
            builder.getI32IntegerAttr(partFactor.value()));
    }
    if (partDim.has_value()) {
        state.addAttribute(
            getPartDimAttrName(state.name),
            builder.getI32IntegerAttr(partDim.value()));
    }
}

/// @brief Parses a `emithls::PragmaArrayPartitionOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_array_partition` `variable=` variable `(` type `)`
///     `type=` partition-type [`factor=` integer] [`dim=` integer]
/// @endcode
ParseResult
PragmaArrayPartitionOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand variable;
    Type variableType;
    if (parser.parseKeyword("variable") || parser.parseEqual()
        || parser.parseOperand(variable) || parser.parseLParen()
        || parser.parseType(variableType) || parser.parseRParen())
        return failure();
    if (parser.resolveOperand(variable, variableType, state.operands))
        return failure();

    StringRef partTypeStr;
    if (parser.parseKeyword("type") || parser.parseEqual()
        || parser.parseKeyword(&partTypeStr))
        return failure();
    auto partType = symbolizeArrayPartitionType(partTypeStr);
    if (!partType.has_value())
        return parser.emitError(
            parser.getNameLoc(),
            "unknown array partition type");
    state.addAttribute(
        getPartTypeAttrName(state.name),
        ArrayPartitionTypeAttr::get(parser.getContext(), partType.value()));

    if (succeeded(parser.parseOptionalKeyword("factor"))) {
        int32_t partFactor;
        if (parser.parseEqual() || parser.parseInteger(partFactor))
            return failure();
        state.addAttribute(
            getPartFactorAttrName(state.name),
            parser.getBuilder().getI32IntegerAttr(partFactor));
    }

    if (succeeded(parser.parseOptionalKeyword("dim"))) {
        int32_t partDim;
        if (parser.parseEqual() || parser.parseInteger(partDim))
            return failure();
        state.addAttribute(
            getPartDimAttrName(state.name),
            parser.getBuilder().getI32IntegerAttr(partDim));
    }

    return success();
}

/// @brief Prints a `emithls::PragmaArrayPartitionOp` in the custom assembly
///        syntax consumed by `parse`.
void PragmaArrayPartitionOp::print(OpAsmPrinter &p)
{
    p << " variable=" << getVariable() << "(" << getVariableType() << ")";
    p << " type=" << stringifyArrayPartitionType(getPartType());
    if (getPartFactor()) p << " factor=" << getPartFactor();
    if (getPartDim()) p << " dim=" << getPartDim();
}

/// @brief Verifies a `emithls::PragmaArrayPartitionOp` is well-formed.
///
/// - `cyclic` and `block` partitions require both a factor and a dimension.
/// - `complete` partition must not specify a factor.
LogicalResult PragmaArrayPartitionOp::verify()
{
    auto partType = getPartType();
    auto partFactor = getPartFactor();
    auto partDim = getPartDim();

    if (partType == ArrayPartitionType::cyclic
        || partType == ArrayPartitionType::block) {
        if (!partFactor.has_value())
            return emitOpError("cyclic and block partition requires a factor");
        if (!partDim.has_value())
            return emitOpError("cyclic and block partition requires a dim");
    } else {
        if (partFactor.has_value())
            return emitOpError("complete partition needs no factor");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// PragmaBindStorageOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaBindStorageOp`.
///
/// @param variable     The variable to bind storage for.
/// @param storageType  The storage type (e.g. `ram_1p`, `ram_2p`, `fifo`).
/// @param storageImpl  The implementation resource (e.g. `bram`, `uram`,
///                     `lutram`).
void PragmaBindStorageOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value variable,
    BindStorageType storageType,
    BindStorageImpl storageImpl)
{
    state.addOperands(variable);
    state.addAttribute(
        getStorageTypeAttrName(state.name),
        BindStorageTypeAttr::get(builder.getContext(), storageType));
    state.addAttribute(
        getStorageImplAttrName(state.name),
        BindStorageImplAttr::get(builder.getContext(), storageImpl));
}

/// @brief Parses a `emithls::PragmaBindStorageOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_bind_storage` `variable=` variable `(` type `)`
///     `type=` storage-type `impl=` storage-impl
/// @endcode
ParseResult
PragmaBindStorageOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand variable;
    Type variableType;
    if (parser.parseKeyword("variable") || parser.parseEqual()
        || parser.parseOperand(variable) || parser.parseLParen()
        || parser.parseType(variableType) || parser.parseRParen())
        return failure();
    if (parser.resolveOperand(variable, variableType, state.operands))
        return failure();

    StringRef storageTypeStr;
    if (parser.parseKeyword("type") || parser.parseEqual()
        || parser.parseKeyword(&storageTypeStr))
        return failure();
    auto storageType = symbolizeBindStorageType(storageTypeStr);
    if (!storageType.has_value())
        return parser.emitError(
            parser.getNameLoc(),
            "unknown bind storage type");
    state.addAttribute(
        getStorageTypeAttrName(state.name),
        BindStorageTypeAttr::get(
            parser.getBuilder().getContext(),
            storageType.value()));

    StringRef storageImplStr;
    if (parser.parseKeyword("impl") || parser.parseEqual()
        || parser.parseKeyword(&storageImplStr))
        return failure();
    auto storageImpl = symbolizeBindStorageImpl(storageImplStr);
    if (!storageImpl.has_value())
        return parser.emitError(
            parser.getNameLoc(),
            "unknown bind storage impl");
    state.addAttribute(
        getStorageImplAttrName(state.name),
        BindStorageImplAttr::get(
            parser.getBuilder().getContext(),
            storageImpl.value()));

    return success();
}

/// @brief Prints a `emithls::PragmaBindStorageOp` in the custom assembly
///        syntax consumed by `parse`.
void PragmaBindStorageOp::print(OpAsmPrinter &p)
{
    p << " variable=" << getVariable() << "(" << getVariableType() << ")";
    p << " type=" << stringifyBindStorageType(getStorageType());
    p << " impl=" << stringifyBindStorageImpl(getStorageImpl());
}

//===----------------------------------------------------------------------===//
// PragmaDataflowOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaDataflowOp` with an inline body.
///
/// Creates a single-block region and invokes @p bodyBuilder (if provided) to
/// populate it with the dataflow task graph.
///
/// @param bodyBuilder  Optional callback that emits ops into the dataflow body;
///                     pass `nullptr` to create an empty region.
void PragmaDataflowOp::build(
    OpBuilder &builder,
    OperationState &state,
    function_ref<void(OpBuilder &, Location)> bodyBuilder)
{
    OpBuilder::InsertionGuard guard(builder);
    Region* dataflowRegion = state.addRegion();
    if (bodyBuilder) {
        builder.createBlock(dataflowRegion);
        bodyBuilder(builder, state.location);
    }
}

//===----------------------------------------------------------------------===//
// PragmaInlineOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaInlineOp`.
///
/// @param off  If true, marks the enclosing function as `inline off`
///             (inhibits inlining); if false, requests inlining.
void PragmaInlineOp::build(OpBuilder &builder, OperationState &state, bool off)
{
    if (off)
        state.addAttribute(getOffAttrName(state.name), builder.getUnitAttr());
}

/// @brief Parses a `emithls::PragmaInlineOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_inline` [`off`]
/// @endcode
ParseResult PragmaInlineOp::parse(OpAsmParser &parser, OperationState &state)
{
    if (succeeded(parser.parseOptionalKeyword("off")))
        state.addAttribute(
            getOffAttrName(state.name),
            parser.getBuilder().getUnitAttr());
    return success();
}

/// @brief Prints a `emithls::PragmaInlineOp` in the custom assembly syntax
///        consumed by `parse`.
void PragmaInlineOp::print(OpAsmPrinter &p)
{
    if (getOff()) p << " off";
}

namespace {
struct HoistPragmaInlinePattern : public OpRewritePattern<PragmaInlineOp> {
    using OpRewritePattern<PragmaInlineOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(PragmaInlineOp op, PatternRewriter &) const override
    {
        Block* block = op->getBlock();
        if (&block->front() == op.getOperation()) return failure();
        op->moveBefore(&block->front());
        return success();
    }
    // Ensure this has higher priority over constant hoist
    PatternBenefit getBenefit() const { return 2; }
};
} // namespace

/// @brief Canonicalization patterns for `PragmaInlineOp`.
///
/// - HoistPragmaInlinePattern: moves a `PragmaInlineOp` to the front of its
///   block so it appears before other ops (e.g. hoisted constants).  Has
///   higher priority than constant hoisting to preserve correct ordering.
void PragmaInlineOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns,
    MLIRContext* context)
{ patterns.add<HoistPragmaInlinePattern>(context, 100); }

//===----------------------------------------------------------------------===//
// PragmaPipelineOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaPipelineOp` with the given interval and
/// pipeline style.
///
/// @param interval  The initiation interval (II); must be greater than 0.
/// @param style     The HLS pipeline style (e.g. `PipelineStyle::flp`).
void PragmaPipelineOp::build(
    OpBuilder &builder,
    OperationState &result,
    int32_t interval,
    emithls::PipelineStyle style)
{
    result.addAttribute(
        getIntervalAttrName(result.name),
        builder.getI32IntegerAttr(interval));
    result.addAttribute(
        getStyleAttrName(result.name),
        emithls::PipelineStyleAttr::get(builder.getContext(), style));
}

/// @brief Parses a `emithls::PragmaPipelineOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_pipeline` `II=`integer [`style=`style]
/// @endcode
///
/// If `style` is omitted, defaults to `flp`.
ParseResult PragmaPipelineOp::parse(OpAsmParser &parser, OperationState &state)
{
    // Parse initial interval as integer
    int32_t interval;
    if (parser.parseKeyword("II") || parser.parseEqual()
        || parser.parseInteger(interval))
        return failure();
    state.addAttribute(
        getIntervalAttrName(state.name),
        parser.getBuilder().getI32IntegerAttr(interval));

    // Parse style string, if none, use flp by default
    PipelineStyle style = PipelineStyle::flp;
    if (succeeded(parser.parseOptionalKeyword("style"))) {
        if (parser.parseEqual()) return failure();
        StringRef styleStr;
        if (parser.parseKeyword(&styleStr)) return failure();
        auto styleValue = symbolizePipelineStyle(styleStr);
        if (!styleValue)
            return parser.emitError(
                parser.getNameLoc(),
                "unknown pipeline style");
        style = *styleValue;
    }
    state.addAttribute(
        getStyleAttrName(state.name),
        PipelineStyleAttr::get(parser.getContext(), style));

    return success();
}

/// @brief Prints a `emithls::PragmaPipelineOp` in the custom assembly
/// syntax consumed by `parse`.
void PragmaPipelineOp::print(OpAsmPrinter &p)
{
    p << " II=" << getInterval()
      << " style=" << stringifyPipelineStyle(getStyle());
}

/// @brief Verifies that a `emithls::PragmaPipelineOp` is well-formed.
///
/// Checks that:
/// - the op is directly contained in a `ForOp`, and
/// - the initiation interval (II) is greater than 0.
LogicalResult PragmaPipelineOp::verify()
{
    if (!isa<ForOp>(getOperation()->getParentOp()))
        return emitOpError("pipeline pragma must be in a for-loop");
    if (getInterval() <= 0) return emitOpError("II must be greater than 0");
    return success();
}

namespace {
struct RemoveNestedPipeline final : public OpRewritePattern<PragmaPipelineOp> {
    using OpRewritePattern<PragmaPipelineOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        PragmaPipelineOp op,
        PatternRewriter &rewriter) const override
    {
        auto hasPrevPipeline = [](Operation* o) {
            auto* prev = o->getPrevNode();
            return prev && isa<PragmaPipelineOp>(prev);
        };
        // Check if there is pipeline before
        if (hasPrevPipeline(op)) {
            rewriter.eraseOp(op);
            return success();
        }
        // Check if the parent loop is already pipelined
        for (Operation* parent = op->getParentOp(); parent;
             parent = parent->getParentOp()) {
            if (isa<emithls::ForOp>(parent) && hasPrevPipeline(parent)) {
                rewriter.eraseOp(op);
                return success();
            }
        }
        return failure();
    }
};

} // namespace

/// @brief Canonicalization patterns for `PragmaPipelineOp`.
///
/// - RemoveNestedPipeline: removes a `PragmaPipelineOp` that is redundant
///   because a sibling or ancestor `ForOp` is already pipelined.
void PragmaPipelineOp::getCanonicalizationPatterns(
    RewritePatternSet &results,
    MLIRContext* context)
{ results.add<RemoveNestedPipeline>(context); }

//===----------------------------------------------------------------------===//
// PragmaStreamOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaStreamOp`.
///
/// @param variable  The stream variable (or array of streams) to configure.
/// @param depth     The FIFO depth for the stream channel; must be positive.
void PragmaStreamOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value variable,
    int32_t depth)
{
    state.addOperands(variable);
    state.addAttribute(
        getDepthAttrName(state.name),
        builder.getI32IntegerAttr(depth));
}

/// @brief Parses a `emithls::PragmaStreamOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_stream` `variable=` variable `(` type `)` `depth=` integer
/// @endcode
ParseResult PragmaStreamOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand variable;
    Type variableType;
    if (parser.parseKeyword("variable") || parser.parseEqual()
        || parser.parseOperand(variable) || parser.parseLParen()
        || parser.parseType(variableType) || parser.parseRParen())
        return failure();
    if (parser.resolveOperand(variable, variableType, state.operands))
        return failure();

    int32_t depth;
    if (parser.parseKeyword("depth") || parser.parseEqual()
        || parser.parseInteger(depth))
        return failure();
    state.addAttribute(
        getDepthAttrName(state.name),
        parser.getBuilder().getI32IntegerAttr(depth));

    return success();
}

/// @brief Prints a `emithls::PragmaStreamOp` in the custom assembly syntax
///        consumed by `parse`.
void PragmaStreamOp::print(OpAsmPrinter &p)
{
    p << " variable=" << getVariable() << "(" << getVariableType() << ")";
    p << " depth=" << getDepth();
}

/// @brief Verifies a `emithls::PragmaStreamOp` is well-formed.
///
/// - The depth must be a positive integer.
/// - If the variable is an `ArrayType`, its element type must be a
/// `StreamType`.
LogicalResult PragmaStreamOp::verify()
{
    if (getDepth() <= 0) return emitOpError("expect positive integer depth");
    if (auto arrayType = dyn_cast<ArrayType>(getVariableType())) {
        if (!isa<StreamType>(arrayType.getElementType()))
            return emitOpError("expect array of streams");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// PragmaUnrollOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::PragmaUnrollOp`.
///
/// @param factor  Optional unroll factor; when absent the loop is fully
///                unrolled.
void PragmaUnrollOp::build(
    OpBuilder &builder,
    OperationState &state,
    std::optional<int32_t> factor)
{
    if (factor.has_value())
        state.addAttribute(
            getFactorAttrName(state.name),
            builder.getI32IntegerAttr(factor.value()));
}

/// @brief Parses a `emithls::PragmaUnrollOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.pragma_unroll` [`factor=` integer]
/// @endcode
///
/// When `factor` is omitted the loop is fully unrolled.
ParseResult PragmaUnrollOp::parse(OpAsmParser &parser, OperationState &state)
{
    if (succeeded(parser.parseOptionalKeyword("factor"))) {
        int32_t factor;
        if (parser.parseEqual() || parser.parseInteger(factor))
            return failure();
        state.addAttribute(
            getFactorAttrName(state.name),
            parser.getBuilder().getI32IntegerAttr(factor));
    }
    return success();
}

/// @brief Prints a `emithls::PragmaUnrollOp` in the custom assembly syntax
///        consumed by `parse`.
void PragmaUnrollOp::print(OpAsmPrinter &p)
{
    if (getFactor()) p << " factor=" << getFactor();
}

//===----------------------------------------------------------------------===//
// Helper Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HelperLineBufferOp
//===----------------------------------------------------------------------===//

/// @brief Names the SSA result of a `emithls::HelperLineBufferOp`.
///
/// Assigns a name of the form `linebuf_<N>` where `N` is the zero-based index
/// among all `HelperLineBufferOp`s that appear before this one in a pre-order
/// walk of the nearest enclosing non-module ancestor.
void HelperLineBufferOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* topLevel = self;
    while (Operation* p = topLevel->getParentOp()) {
        if (isa<ModuleOp>(p)) break;
        topLevel = p;
    }
    unsigned count = 0;
    topLevel->walk([&](HelperLineBufferOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getBufRef(), "linebuf_" + std::to_string(count));
}

/// @brief Builds a `emithls::HelperLineBufferOp`.
///
/// @param tokenRef     The shift-register token memref/array that backs the
///                     line buffer storage.
/// @param index        The current pixel index selecting which line to retain.
/// @param linebufType  The type of the resulting line buffer memref/array.
/// @param numChan      Per-dimension channel counts stored per pixel position.
/// @param numLine      Number of lines held simultaneously in the buffer.
void HelperLineBufferOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value tokenRef,
    Value index,
    Type linebufType,
    ArrayRef<int32_t> numChan,
    int32_t numLine)
{
    state.addOperands(tokenRef);
    if (index) state.addOperands(index);
    state.addAttribute(
        getNumChanAttrName(state.name),
        builder.getI32ArrayAttr(numChan));
    state.addAttribute(
        getNumLineAttrName(state.name),
        builder.getI32IntegerAttr(numLine));
    state.addTypes({linebufType});
}

/// @brief Parses a `emithls::HelperLineBufferOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.helper_line_buffer` token-memref
///     `num_chan` `[` integer (`,` integer)* `]` `num_line` integer
///     `keep` index `:` token-type `->` linebuf-type
/// @endcode
ParseResult
HelperLineBufferOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand tokenRef, index;
    bool hasKeepIndex = false;
    int32_t numLine;
    SmallVector<int32_t> numChan;
    Type tokenType, linebufType;

    if (parser.parseOperand(tokenRef) || parser.parseKeyword("num_chan")
        || parser.parseLSquare())
        return failure();

    if (parser.parseOptionalRSquare().failed()) {
        do {
            int32_t val;
            if (parser.parseInteger(val)) return failure();
            numChan.push_back(val);
        } while (parser.parseOptionalComma().succeeded());
        if (parser.parseRSquare()) return failure();
    }

    if (parser.parseKeyword("num_line") || parser.parseInteger(numLine))
        return failure();
    if (succeeded(parser.parseOptionalKeyword("keep"))) {
        hasKeepIndex = true;
        if (parser.parseOperand(index)) return failure();
    }
    if (parser.parseColon() || parser.parseType(tokenType)
        || parser.parseArrow() || parser.parseType(linebufType))
        return failure();

    if (parser.resolveOperand(tokenRef, tokenType, state.operands))
        return failure();
    if (hasKeepIndex)
        if (parser.resolveOperand(
                index,
                parser.getBuilder().getIndexType(),
                state.operands))
            return failure();

    state.addAttribute(
        getNumChanAttrName(state.name),
        parser.getBuilder().getI32ArrayAttr(numChan));
    state.addAttribute(
        getNumLineAttrName(state.name),
        parser.getBuilder().getI32IntegerAttr(numLine));
    state.addTypes({linebufType});

    return success();
}

/// @brief Prints a `emithls::HelperLineBufferOp` in the custom assembly syntax
///        consumed by `parse`.
void HelperLineBufferOp::print(OpAsmPrinter &p)
{
    p << " " << getTokenRef() << " num_chan [";
    llvm::interleaveComma(getNumChan(), p, [&](Attribute attr) {
        p << llvm::cast<IntegerAttr>(attr).getInt();
    });
    p << "] num_line " << getNumLine();
    if (getIndex()) p << " keep " << getIndex();
    p << " : " << getTokenRef().getType() << " -> " << getBufRef().getType();
}

//===----------------------------------------------------------------------===//
// HelperWindowOp
//===----------------------------------------------------------------------===//

/// @brief Names the SSA result of a `emithls::HelperWindowOp`.
///
/// Assigns a name of the form `window_<N>` where `N` is the zero-based index
/// among all `HelperWindowOp`s that appear before this one in a pre-order
/// walk of the nearest enclosing non-module ancestor.
void HelperWindowOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn)
{
    Operation* self = getOperation();
    Operation* topLevel = self;
    while (Operation* p = topLevel->getParentOp()) {
        if (isa<ModuleOp>(p)) break;
        topLevel = p;
    }
    unsigned count = 0;
    topLevel->walk([&](HelperWindowOp op) -> WalkResult {
        if (op.getOperation() == self) return WalkResult::interrupt();
        ++count;
        return WalkResult::advance();
    });
    setNameFn(getWinRef(), "window_" + std::to_string(count));
}

/// @brief Builds a `emithls::HelperWindowOp`.
///
/// @param linebuf     The line buffer memref/array from which the window is
///                    extracted.
/// @param indices     Multi-dimensional index operands selecting the window
///                    origin within the line buffer.
/// @param windowType  The type of the resulting window memref/array.
void HelperWindowOp::build(
    OpBuilder &,
    OperationState &state,
    Value linebuf,
    ValueRange indices,
    Type windowType)
{
    state.addOperands(linebuf);
    state.addOperands(indices);
    state.addTypes(windowType);
}

/// @brief Parses a `emithls::HelperWindowOp` from its custom assembly syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.helper_window` linebuf `from` `[` index-list `]`
///     `:` buffer-type `->` window-type
/// @endcode
///
/// All index operands are resolved as `index` type.
ParseResult HelperWindowOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand linebuf;
    SmallVector<OpAsmParser::UnresolvedOperand> indices;
    Type bufferType, windowType;

    if (parser.parseOperand(linebuf) || parser.parseKeyword("from")
        || parser.parseLSquare()
        || parser.parseCommaSeparatedList([&]() -> ParseResult {
               OpAsmParser::UnresolvedOperand idx;
               if (parser.parseOperand(idx)) return failure();
               indices.push_back(idx);
               return success();
           })
        || parser.parseRSquare() || parser.parseColon()
        || parser.parseType(bufferType) || parser.parseArrow()
        || parser.parseType(windowType))
        return failure();

    if (parser.resolveOperand(linebuf, bufferType, state.operands))
        return failure();
    if (parser.resolveOperands(
            indices,
            parser.getBuilder().getIndexType(),
            state.operands))
        return failure();
    state.addTypes(windowType);

    return success();
}

/// @brief Prints a `emithls::HelperWindowOp` in the custom assembly syntax
///        consumed by `parse`.
void HelperWindowOp::print(OpAsmPrinter &p)
{
    p << " " << getBufRef() << " from [";
    for (auto [i, index] : llvm::enumerate(getIndices())) {
        p << index;
        if (i == getIndices().size() - 1) break;
        p << ", ";
    }
    p << "] : " << getBufRef().getType() << " -> " << getWinRef().getType();
}

//===----------------------------------------------------------------------===//
// HelperAccumulateOp
//===----------------------------------------------------------------------===//

/// @brief Builds a `emithls::HelperAccumulateOp`.
///
/// @param accuRef       The accumulation target memref/array.
/// @param indices       Index operands into @p accuRef selecting the element
///                      to accumulate into.
/// @param operatorCode  The reduction operator (e.g., add, mul) to apply.
/// @param value         The value to reduce into the element at @p indices.
void HelperAccumulateOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value accuRef,
    ValueRange indices,
    FusedOperator operatorCode,
    Value value)
{
    state.addOperands(accuRef);
    state.addOperands(value);
    state.addOperands(indices);
    state.addAttribute(
        getOpCodeAttrName(state.name),
        FusedOperatorAttr::get(builder.getContext(), operatorCode));
}

/// @brief Parses a `emithls::HelperAccumulateOp` from its custom assembly
/// syntax.
///
/// The expected syntax is:
/// @code
///   `emithls.helper_accumulate` accumulator-ref `at` `[` index-list `]`
///     operator-code value `:` memref-or-array-type (`cast` value-type)?
/// @endcode
///
/// The element type of @p memref-or-array-type is used to resolve @p value,
/// unless an explicit trailing `cast` value-type overrides it — needed when
/// the accumulate increment was computed at a different precision than the
/// accumulator. All index operands are resolved as `index` type.
ParseResult
HelperAccumulateOp::parse(OpAsmParser &parser, OperationState &state)
{
    OpAsmParser::UnresolvedOperand accuRef, value;
    SmallVector<OpAsmParser::UnresolvedOperand> indices;
    StringRef operatorCodeStr;
    Type type;

    if (parser.parseOperand(accuRef) || parser.parseKeyword("at")
        || parser.parseLSquare())
        return failure();
    if (failed(parser.parseOptionalRSquare())) {
        if (parser.parseCommaSeparatedList([&]() -> ParseResult {
                OpAsmParser::UnresolvedOperand idx;
                if (parser.parseOperand(idx)) return failure();
                indices.push_back(idx);
                return success();
            })
            || parser.parseRSquare())
            return failure();
    }
    if (parser.parseKeyword(&operatorCodeStr) || parser.parseOperand(value)
        || parser.parseColonType(type))
        return failure();

    auto operatorCode = symbolizeFusedOperator(operatorCodeStr);
    if (!operatorCode.has_value())
        return parser.emitError(
            parser.getNameLoc(),
            "unknown accumulate operator");
    state.addAttribute(
        getOpCodeAttrName(state.name),
        FusedOperatorAttr::get(parser.getContext(), operatorCode.value()));

    auto shapedType = cast<ShapedType>(type);
    Type valueType = shapedType.getElementType();
    if (succeeded(parser.parseOptionalKeyword("cast"))
        && parser.parseType(valueType))
        return failure();

    if (parser.resolveOperand(accuRef, shapedType, state.operands))
        return failure();
    if (parser.resolveOperand(value, valueType, state.operands))
        return failure();
    if (parser.resolveOperands(
            indices,
            parser.getBuilder().getIndexType(),
            parser.getCurrentLocation(),
            state.operands))
        return failure();

    return success();
}

/// @brief Prints a `emithls::HelperAccumulateOp` in the custom assembly syntax
///        consumed by `parse`.
void HelperAccumulateOp::print(OpAsmPrinter &p)
{
    p << " " << getAccuRef() << " at [";
    llvm::interleaveComma(getIndices(), p);
    p << "] " << stringifyFusedOperator(getOpCode()) << " " << getValue();
    p << " : " << getAccuRef().getType();
    Type valueType = getValue().getType();
    if (valueType != cast<ShapedType>(getAccuRef().getType()).getElementType())
        p << " cast " << valueType;
}

//===----------------------------------------------------------------------===//
// EmitHLSDialect
//===----------------------------------------------------------------------===//

void EmitHLSDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.cpp.inc"
        >();
}
