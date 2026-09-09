// Implementation of translation EmitHLSToCpp
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToCpp/HLSCppEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::emithls;
using llvm::formatv;

namespace {

// Calls eachFn on every element of c, printing a comma between elements,
// and stops at the first failure.
template<typename Container, typename UnaryFunctor>
static LogicalResult interleaveCommaWithError(
    const Container &c,
    raw_ostream &os,
    UnaryFunctor eachFn)
{
    auto it = c.begin();
    if (it == c.end()) return success();
    if (failed(eachFn(*it))) return failure();
    for (++it; it != c.end(); ++it) {
        os << ", ";
        if (failed(eachFn(*it))) return failure();
    }
    return success();
}

// Binding power used to decide whether a nested expression operand needs
// parentheses. Only operations that can appear inside an expression tree
// are covered here.
static FailureOr<int> getOperatorPrecedence(Operation* operation)
{
    return llvm::TypeSwitch<Operation*, FailureOr<int>>(operation)
        .Case<ArithAddOp>([](auto) { return 12; })
        .Case<ArithSubOp>([](auto) { return 12; })
        .Case<ArithMulOp>([](auto) { return 13; })
        .Case<ArithDivOp>([](auto) { return 13; })
        .Case<ArithRemOp>([](auto) { return 13; })
        .Case<ArithAndOp>([](auto) { return 7; })
        .Case<ArithOrOp>([](auto) { return 5; })
        .Case<ArithLogicalAndOp>([](auto) { return 4; })
        .Case<ArithLogicalOrOp>([](auto) { return 3; })
        .Case<ArithShlOp>([](auto) { return 11; })
        .Case<ArithShrOp>([](auto) { return 11; })
        .Case<ArithCastOp>([](auto) { return 15; })
        .Case<ArithDataRangeOp>([](auto) { return 1; })
        .Case<ArithCmpOp>([](auto op) -> FailureOr<int> {
            switch (op.getPredicate()) {
            case CmpPredicate::eq:
            case CmpPredicate::ne: return 8;
            case CmpPredicate::lt:
            case CmpPredicate::le:
            case CmpPredicate::gt:
            case CmpPredicate::ge: return 9;
            }
            return op->emitError("unsupported cmp predicate");
        })
        .Case<ArithSelectOp>([](auto) { return 2; })
        .Case<CallOp>([](auto) { return 16; })
        .Case<ExpressionOp>([](auto) { return 17; })
        .Default([](auto op) {
            return op->emitError("unsupported operation in expression");
        });
}

//===----------------------------------------------------------------------===//
// HLSCppEmitter
//===----------------------------------------------------------------------===//

// Emits a program in EmitHLS IR as HLS C++ source
class HLSCppEmitter {
public:
    explicit HLSCppEmitter(raw_ostream &os) : os(os) {}

    // Emits the given operation, dispatching to the operation-specific
    // printer. Appends a trailing semicolon/newline for statements when
    // requested.
    LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

    // Emitter helpers. Implemented alongside the operations that need them.
    LogicalResult emitType(Location loc, Type type);
    LogicalResult
    emitAttribute(Location loc, Attribute attr, bool printConstant);
    LogicalResult emitVariableDeclaration(OpResult result);
    LogicalResult emitAssignPrefix(Operation &op);
    LogicalResult emitExpression(ExpressionOp exprOp, bool isNested);
    LogicalResult emitOperand(Value value);

    raw_indented_ostream &getOS() { return os; }
    Operation* getCurrentOperation() { return currentOperation; }

    // Whether varOp should be emitted as a bare literal at its use sites
    // instead of being declared as a named C++ variable.
    bool shouldEmitConstant(VariableOp varOp)
    {
        if (!varOp.getIsConst()) return false;
        if (isa<ArrayType>(varOp.getType())) return false;
        return true;
    }

    //===------------------------------------------------------------------===//
    // Value naming
    //===------------------------------------------------------------------===//

    StringRef getOrCreateName(Value value);

    //===------------------------------------------------------------------===//
    // Expression precedence. Tracks the enclosing expression and its
    // operator precedence so nested expressions know when they need
    // parentheses.
    //===------------------------------------------------------------------===//

    ExpressionOp getEmittedExpression() { return emittedExpression; }
    bool isPartOfCurrentExpression(Value value)
    {
        if (!emittedExpression) return false;
        Operation* defOp = value.getDefiningOp();
        if (!defOp) return false;
        auto exprOp = dyn_cast<ExpressionOp>(defOp->getParentOp());
        return exprOp == emittedExpression;
    }
    void pushEmittedExpression(ExpressionOp exprOp)
    {
        expressionStack.push_back(emittedExpression);
        emittedExpression = exprOp;
    }
    void restoreEmittedExpression()
    {
        if (!expressionStack.empty()) {
            emittedExpression = expressionStack.back();
            expressionStack.pop_back();
        } else {
            emittedExpression = nullptr;
        }
    }
    void pushExpressionPrecedence(int precedence)
    { emittedExpressionPrecedence.push_back(precedence); }
    void popExpressionPrecedence() { emittedExpressionPrecedence.pop_back(); }
    static int lowestPrecedence() { return 0; }
    int getExpressionPrecedence()
    {
        if (emittedExpressionPrecedence.empty()) return lowestPrecedence();
        return emittedExpressionPrecedence.back();
    }

    // RAII helper that drops the value names introduced within a block
    // once it goes out of scope.
    struct BlockScope {
        BlockScope(HLSCppEmitter &emitter, Block*) : emitter(emitter)
        {
            oldValues =
                llvm::to_vector(llvm::make_first_range(emitter.valueNames));
        }
        ~BlockScope()
        {
            llvm::SmallPtrSet<Value, 16> oldValueSet(
                oldValues.begin(),
                oldValues.end());
            llvm::DenseMap<Value, std::string> newValueNames;
            for (const auto &pair : emitter.valueNames)
                if (oldValueSet.contains(pair.first))
                    newValueNames.insert(pair);
            emitter.valueNames = std::move(newValueNames);
        }

    private:
        HLSCppEmitter &emitter;
        SmallVector<Value> oldValues;
    };

private:
    // Picks the prefix used when generating a name for an operation's
    // result, keyed by the operation name without its dialect prefix.
    struct NameManager {
        NameManager() = default;
        std::string getPrefix(Operation* op)
        {
            if (!op) return "v";
            auto opName = op->getName().getStringRef();
            size_t firstDotPos = opName.find('.');
            if (firstDotPos != StringRef::npos)
                opName = opName.substr(firstDotPos + 1);
            auto it = prefixMap.find(opName);
            if (it != prefixMap.end()) return it->second;
            return "v";
        }

    private:
        const llvm::StringMap<std::string> prefixMap = {
            // ArithOps
            {     "arith.add",  "sum"},
            {    "arith.cast", "cast"},
            {     "arith.cmp",  "cmp"},
            {     "arith.div", "quot"},
            {     "arith.mul", "prod"},
            {      "arith.or",   "or"},
            {     "arith.rem",  "rem"},
            {  "arith.select",  "mux"},
            {     "arith.sub", "diff"},
            {     "arith.shr",  "shr"},
            {     "arith.shl",  "shl"},
            {     "arith.max",  "max"},
            {     "arith.min",  "min"},
            // ArrayOps
            {    "array.read", "elem"},
            {"array.ptr_read", "elem"},
            // StreamOps
            {   "stream.read", "data"},
        };
    };

    StringRef addNameForValue(Value value, const std::string &baseName)
    {
        unsigned suffix = 0;
        std::string nameStr;
        do {
            nameStr = formatv("{0}{1}", baseName, suffix++);
        } while (llvm::any_of(valueNames, [&](const auto &entry) {
            return entry.second == nameStr;
        }));
        valueNames[value] = nameStr;
        return valueNames[value];
    }

    // Operations
    Operation* currentOperation = nullptr;
    ExpressionOp emittedExpression = nullptr;
    SmallVector<ExpressionOp> expressionStack;
    SmallVector<int> emittedExpressionPrecedence;

    // Printer
    raw_indented_ostream os;

    // Helper for name generation
    llvm::DenseMap<Value, std::string> valueNames;
    NameManager nameManager;
};

} // namespace

//===----------------------------------------------------------------------===//
// Print different operations
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(HLSCppEmitter &emitter, ModuleOp moduleOp)
{
    for (Operation &op : moduleOp.getBody()->getOperations())
        if (failed(emitter.emitOperation(op, true))) return failure();
    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, IncludeOp includeOp)
{
    raw_indented_ostream &os = emitter.getOS();
    os << "#include \"" << includeOp.getHeader() << "\"";
    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, VariableOp variableOp)
{
    raw_indented_ostream &os = emitter.getOS();

    bool hasInitNumber = static_cast<bool>(variableOp.getInitNumber());
    Value initValue = variableOp.getInitValue();

    if (hasInitNumber || initValue) {
        if (variableOp.getIsConst()) {
            if (emitter.shouldEmitConstant(variableOp)) return success();
            os << "const ";
        }
        if (failed(emitter.emitAssignPrefix(*variableOp.getOperation())))
            return failure();
        if (hasInitNumber
            && failed(emitter.emitAttribute(
                variableOp->getLoc(),
                variableOp.getInitNumberAttr(),
                false)))
            return failure();
        if (initValue && failed(emitter.emitOperand(initValue)))
            return failure();
    } else if (
        failed(emitter.emitVariableDeclaration(
            variableOp.getOperation()->getResult(0)))) {
        return failure();
    }
    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, UpdateOp updateOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitOperand(updateOp.getVariable()))) return failure();
    for (auto idx : updateOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    os << " = ";
    if (failed(emitter.emitOperand(updateOp.getNewValue()))) return failure();
    for (auto idx : updateOp.getNewValueIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, FuncOp funcOp)
{
    raw_indented_ostream &os = emitter.getOS();
    HLSCppEmitter::BlockScope scope(emitter, &funcOp.getBody().front());

    os << "void " << funcOp.getSymName() << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(),
            os,
            [&](BlockArgument arg) -> LogicalResult {
                auto argTy = arg.getType();
                if (failed(emitter.emitType(funcOp.getLoc(), argTy)))
                    return failure();
                os << " ";
                if (isa<StreamType>(argTy))
                    os << "&";
                else if (isa<PointerType>(argTy))
                    os << "*";
                os << emitter.getOrCreateName(arg);
                if (auto arrTy = dyn_cast<ArrayType>(argTy))
                    for (auto dim : arrTy.getShape()) os << "[" << dim << "]";
                return success();
            })))
        return failure();
    os << ")\n{\n";
    os.indent();
    for (Operation &opi : funcOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, true))) return failure();
    os.unindent() << "}";

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, CallOp callOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << callOp.getCallee() << "(";
    if (failed(interleaveCommaWithError(
            callOp.getArgOperands(),
            os,
            [&](Value arg) -> LogicalResult {
                return emitter.emitOperand(arg);
            })))
        return failure();
    os << ")";

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, IfOp ifOp)
{
    raw_indented_ostream &os = emitter.getOS();
    HLSCppEmitter::BlockScope scope(emitter, &ifOp.getThenRegion().front());

    os << "if (";
    if (failed(emitter.emitOperand(ifOp.getCondition()))) return failure();
    os << ") {\n";
    os.indent();
    for (Operation &opi : ifOp.getThenRegion().getOps())
        if (failed(emitter.emitOperation(opi, true))) return failure();
    os.unindent();
    os << "}";
    if (!ifOp.getElseRegion().empty()) {
        os << " else {\n";
        os.indent();
        for (Operation &opi : ifOp.getElseRegion().getOps())
            if (failed(emitter.emitOperation(opi, true))) return failure();
        os.unindent();
        os << "}";
    }
    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, ForOp forOp)
{
    raw_indented_ostream &os = emitter.getOS();
    HLSCppEmitter::BlockScope scope(emitter, &forOp.getBody().front());

    Value iVar = forOp.getInductionVariable();

    os << "for (";
    if (failed(emitter.emitType(forOp.getLoc(), iVar.getType())))
        return failure();
    os << " " << emitter.getOrCreateName(iVar) << " = " << forOp.getLowerBound()
       << ";";
    os << " " << emitter.getOrCreateName(iVar) << " < " << forOp.getUpperBound()
       << ";";
    os << " " << emitter.getOrCreateName(iVar) << " += " << forOp.getStep()
       << ") {\n";
    os.indent();
    for (Operation &opi : forOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, true))) return failure();
    os.unindent();
    os << "}";

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &, ExpressionOp)
{
    // The contents of an expression are never visited generically. They get
    // emitted inline, at the point of use, by emitOperand and emitExpression.
    return success();
}

//===----------------------------------------------------------------------===//
// ArithOps
//===----------------------------------------------------------------------===//

static LogicalResult printBinaryOperation(
    HLSCppEmitter &emitter,
    Operation* operation,
    StringRef binaryOperator)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*operation))) return failure();
    if (failed(emitter.emitOperand(operation->getOperand(0)))) return failure();
    os << " " << binaryOperator << " ";
    if (failed(emitter.emitOperand(operation->getOperand(1)))) return failure();

    return success();
}

static LogicalResult printLogicalOperation(
    HLSCppEmitter &emitter,
    Operation* operation,
    StringRef logicalOperator)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*operation))) return failure();
    SmallVector<Value> operands = operation->getOperands();
    for (Value operand : operands) {
        if (failed(emitter.emitOperand(operand))) return failure();
        if (operand != operands.back()) os << " " << logicalOperator << " ";
    }

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithAddOp addOp)
{ return printBinaryOperation(emitter, addOp.getOperation(), "+"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithSubOp subOp)
{ return printBinaryOperation(emitter, subOp.getOperation(), "-"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithMulOp mulOp)
{ return printBinaryOperation(emitter, mulOp.getOperation(), "*"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithDivOp divOp)
{ return printBinaryOperation(emitter, divOp.getOperation(), "/"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithRemOp remOp)
{ return printBinaryOperation(emitter, remOp.getOperation(), "%"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithAndOp andOp)
{ return printBinaryOperation(emitter, andOp.getOperation(), "&"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithOrOp orOp)
{ return printBinaryOperation(emitter, orOp.getOperation(), "|"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithShlOp shlOp)
{ return printBinaryOperation(emitter, shlOp.getOperation(), "<<"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithShrOp shrOp)
{ return printBinaryOperation(emitter, shrOp.getOperation(), ">>"); }

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithMaxOp maxOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*maxOp.getOperation())))
        return failure();

    os << "std::max(";
    if (failed(emitter.emitOperand(maxOp.getLhs()))) return failure();
    os << ", ";
    if (failed(emitter.emitOperand(maxOp.getRhs()))) return failure();
    os << ")";

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithMinOp minOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*minOp.getOperation())))
        return failure();

    os << "std::min(";
    if (failed(emitter.emitOperand(minOp.getLhs()))) return failure();
    os << ", ";
    if (failed(emitter.emitOperand(minOp.getRhs()))) return failure();
    os << ")";

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArithLogicalAndOp andOp)
{ return printLogicalOperation(emitter, andOp.getOperation(), "&&"); }

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArithLogicalOrOp orOp)
{ return printLogicalOperation(emitter, orOp.getOperation(), "||"); }

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArithFusedOp fusedOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitOperand(fusedOp.getAcc()))) return failure();
    os << " ";
    switch (fusedOp.getOpCode()) {
    case FusedOperator::add: os << "+"; break;
    case FusedOperator::sub: os << "-"; break;
    case FusedOperator::mul: os << "*"; break;
    }
    os << "= ";
    if (failed(emitter.emitOperand(fusedOp.getVal()))) return failure();

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithCastOp castOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*castOp.getOperation())))
        return failure();

    os << "(";
    if (failed(emitter.emitType(castOp.getLoc(), castOp.getType())))
        return failure();
    os << ")";
    if (failed(emitter.emitOperand(castOp.getFrom()))) return failure();

    return success();
}

static LogicalResult printOperation(HLSCppEmitter &emitter, ArithCmpOp cmpOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*cmpOp.getOperation())))
        return failure();

    if (failed(emitter.emitOperand(cmpOp.getLhs()))) return failure();
    switch (cmpOp.getPredicate()) {
    case CmpPredicate::eq: os << " == "; break;
    case CmpPredicate::ne: os << " != "; break;
    case CmpPredicate::lt: os << " < "; break;
    case CmpPredicate::le: os << " <= "; break;
    case CmpPredicate::gt: os << " > "; break;
    case CmpPredicate::ge: os << " >= "; break;
    }
    if (failed(emitter.emitOperand(cmpOp.getRhs()))) return failure();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArithSelectOp selectOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*selectOp.getOperation())))
        return failure();

    if (failed(emitter.emitOperand(selectOp.getCondition()))) return failure();
    os << " ? ";
    if (failed(emitter.emitOperand(selectOp.getTrueValue()))) return failure();
    os << " : ";
    if (failed(emitter.emitOperand(selectOp.getFalseValue()))) return failure();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArithDataRangeOp rangeOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*rangeOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(rangeOp.getData());
    os << ".range(";
    if (failed(emitter.emitOperand(rangeOp.getHighBit()))) return failure();
    os << ", ";
    if (failed(emitter.emitOperand(rangeOp.getLowBit()))) return failure();
    os << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// ArrayOps
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(HLSCppEmitter &emitter, ArrayReadOp readOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(readOp.getArray());
    for (auto idx : readOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArrayWriteOp writeOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << emitter.getOrCreateName(writeOp.getArray());
    for (auto idx : writeOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    os << " = ";
    if (failed(emitter.emitOperand(writeOp.getValue()))) return failure();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArrayPointerReadOp readOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(readOp.getPointer());
    for (auto idx : readOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, ArrayPointerWriteOp writeOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << emitter.getOrCreateName(writeOp.getPointer());
    for (auto idx : writeOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    os << " = ";
    if (failed(emitter.emitOperand(writeOp.getValue()))) return failure();

    return success();
}

//===----------------------------------------------------------------------===//
// StreamOps
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(HLSCppEmitter &emitter, StreamReadOp readOp)
{
    raw_indented_ostream &os = emitter.getOS();

    if (failed(emitter.emitAssignPrefix(*readOp.getOperation())))
        return failure();

    os << emitter.getOrCreateName(readOp.getStream());
    for (auto idx : readOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    os << ".read()";

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, StreamWriteOp writeOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << emitter.getOrCreateName(writeOp.getStream());
    for (auto idx : writeOp.getIndices()) {
        os << "[";
        if (failed(emitter.emitOperand(idx))) return failure();
        os << "]";
    }
    os << ".write(";
    if (failed(emitter.emitOperand(writeOp.getData()))) return failure();
    os << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// PragmaOps
//===----------------------------------------------------------------------===//

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaArrayPartitionOp arrayPartOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS ARRAY_PARTITION variable="
       << emitter.getOrCreateName(arrayPartOp.getVariable());
    os << " type=" << arrayPartOp.getPartType();
    if (auto factor = arrayPartOp.getPartFactor()) os << " factor=" << *factor;
    if (auto dim = arrayPartOp.getPartDim()) os << " dim=" << *dim;

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaBindStorageOp bindStorageOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS BIND_STORAGE variable="
       << emitter.getOrCreateName(bindStorageOp.getVariable());
    os << " type=" << bindStorageOp.getStorageType()
       << " impl=" << bindStorageOp.getStorageImpl();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaDataflowOp dataflowOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "\n#pragma HLS DATAFLOW\n";
    for (Operation &opi : dataflowOp.getBody().getOps())
        if (failed(emitter.emitOperation(opi, true))) return failure();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaInlineOp inlineOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS INLINE";
    if (inlineOp.getOff()) os << " off";

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaPipelineOp pipelineOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS PIPELINE II=" << pipelineOp.getInterval();
    os << " style=" << pipelineOp.getStyle();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaStreamOp streamOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS STREAM variable="
       << emitter.getOrCreateName(streamOp.getVariable());
    os << " depth=" << streamOp.getDepth();

    return success();
}

static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaUnrollOp unrollOp)
{
    raw_indented_ostream &os = emitter.getOS();

    os << "#pragma HLS UNROLL";
    if (auto factor = unrollOp.getFactor()) os << " factor=" << *factor;

    return success();
}

// This op only marks where the interface pragmas belong. Function arguments
// are always named arg0, arg1, ... in order, so the port names can be
// reconstructed just by counting the enclosing FuncOp's arguments.
static LogicalResult
printOperation(HLSCppEmitter &emitter, PragmaTopInterfaceOp topInterfaceOp)
{
    raw_indented_ostream &os = emitter.getOS();

    auto funcOp = topInterfaceOp->getParentOfType<FuncOp>();
    if (!funcOp)
        return topInterfaceOp->emitOpError(
            "expected to be nested inside a FuncOp");

    for (unsigned i : llvm::seq(funcOp.getNumArguments())) {
        os << "#pragma HLS INTERFACE mode=m_axi port=arg" << i
           << " offset=slave bundle=gmem_arg" << i << "\n";
        os << "#pragma HLS INTERFACE mode=s_axilite port=arg" << i
           << " bundle=control\n";
    }
    os << "#pragma HLS INTERFACE mode=s_axilite port=return bundle=control";

    return success();
}

//===----------------------------------------------------------------------===//
// Emitter helper methods
//===----------------------------------------------------------------------===//

LogicalResult HLSCppEmitter::emitType(Location loc, Type type)
{
    raw_indented_ostream &os = getOS();
    if (auto intTy = dyn_cast<IntegerType>(type)) {
        auto width = intTy.getWidth();
        if (width == 1) {
            os << "bool";
            return success();
        }
        if (intTy.getSignedness() == IntegerType::Unsigned)
            os << "ap_uint<" << width << ">";
        else
            os << "ap_int<" << width << ">";
        return success();
    }
    if (isa<IndexType>(type)) {
        os << "size_t";
        return success();
    }
    if (auto floatTy = dyn_cast<FloatType>(type)) {
        switch (floatTy.getIntOrFloatBitWidth()) {
        case 16: os << "half"; break;
        case 32: os << "float"; break;
        case 64: os << "double"; break;
        default: return emitError(loc, "unsupported float width for HLS C++");
        }
        return success();
    }
    if (auto arrayTy = dyn_cast<ArrayType>(type))
        return emitType(loc, arrayTy.getElementType());
    if (auto streamTy = dyn_cast<StreamType>(type)) {
        os << "hls::stream<";
        if (failed(emitType(loc, streamTy.getElementType()))) return failure();
        os << ">";
        return success();
    }
    if (auto ptrTy = dyn_cast<PointerType>(type))
        return emitType(loc, ptrTy.getElementType());

    return emitError(loc, "cannot emit type ") << type;
}

LogicalResult
HLSCppEmitter::emitAttribute(Location loc, Attribute attr, bool printConstant)
{
    raw_indented_ostream &os = getOS();
    auto printInt = [&](const APInt &val, bool isUnsigned) {
        if (val.getBitWidth() == 1) {
            os << (val.getBoolValue() ? "true" : "false");
        } else {
            SmallString<128> strValue;
            val.toString(strValue, 10, !isUnsigned, false);
            bool shouldPrintParen = printConstant && val.isNegative();
            if (shouldPrintParen) os << "(";
            os << strValue;
            if (shouldPrintParen) os << ")";
        }
    };
    auto printFloat = [&](const APFloat &val) {
        if (val.isFinite()) {
            SmallString<128> strValue;
            val.toString(strValue);
            os << strValue;
        } else if (val.isNaN()) {
            os << "NAN";
        } else if (val.isInfinity()) {
            if (val.isNegative()) os << "-";
            os << "INFINITY";
        }
    };
    // Print integer attributes.
    if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
        if (auto iTy = dyn_cast<IntegerType>(iAttr.getType())) {
            printInt(
                iAttr.getValue(),
                iTy.getSignedness() == IntegerType::Unsigned);
            return success();
        }
        if (isa<IndexType>(iAttr.getType())) {
            printInt(iAttr.getValue(), false);
            return success();
        }
    }
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
        if (auto iTy = dyn_cast<IntegerType>(dense.getElementType())) {
            os << '{';
            bool isUnsigned = iTy.getSignedness() == IntegerType::Unsigned;
            if (dense.isSplat())
                printInt(*dense.getValues<APInt>().begin(), isUnsigned);
            else
                llvm::interleaveComma(
                    dense.getValues<APInt>(),
                    os,
                    [&](const APInt &val) { printInt(val, isUnsigned); });
            os << '}';
            return success();
        }
        if (isa<IndexType>(dense.getElementType())) {
            os << '{';
            if (dense.isSplat())
                printInt(*dense.getValues<APInt>().begin(), false);
            else
                llvm::interleaveComma(
                    dense.getValues<APInt>(),
                    os,
                    [&](const APInt &val) { printInt(val, false); });
            os << '}';
            return success();
        }
    }

    // Print floating point attributes.
    if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
        if (!isa<Float16Type, Float32Type, Float64Type>(fAttr.getType()))
            return emitError(
                loc,
                "expected floating point attribute to be f16, f32 or f64");
        printFloat(fAttr.getValue());
        return success();
    }
    if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
        if (!isa<Float16Type, Float32Type, Float64Type>(dense.getElementType()))
            return emitError(
                loc,
                "expected floating point attribute to be f16, f32 or f64");
        os << '{';
        if (dense.isSplat())
            printFloat(*dense.getValues<APFloat>().begin());
        else
            llvm::interleaveComma(
                dense.getValues<APFloat>(),
                os,
                [&](const APFloat &val) { printFloat(val); });
        os << '}';
        return success();
    }

    // Print symbolic reference attributes.
    if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
        if (sAttr.getNestedReferences().size() > 1)
            return emitError(loc, "attribute has more than 1 nested reference");
        os << sAttr.getRootReference().getValue();
        return success();
    }

    // Print type attributes.
    if (auto typeAttr = dyn_cast<TypeAttr>(attr))
        return emitType(loc, typeAttr.getValue());

    return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult HLSCppEmitter::emitVariableDeclaration(OpResult result)
{
    raw_indented_ostream &os = getOS();
    if (valueNames.contains(result))
        return result.getDefiningOp()->emitError(
            "result variable for the operation already declared");
    if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
        return failure();
    os << " " << getOrCreateName(result);
    if (auto arrayTy = dyn_cast<ArrayType>(result.getType()))
        for (auto dim : arrayTy.getShape()) os << "[" << dim << "]";
    return success();
}

LogicalResult HLSCppEmitter::emitAssignPrefix(Operation &op)
{
    if (getEmittedExpression()) return success();
    if (op.getNumResults() != 1) return failure();
    if (failed(emitVariableDeclaration(op.getResult(0)))) return failure();
    getOS() << " = ";
    return success();
}

LogicalResult HLSCppEmitter::emitExpression(ExpressionOp exprOp, bool isNested)
{
    if (!isNested)
        assert(
            emittedExpressionPrecedence.empty()
            && "expect precedence stack to be empty");

    Operation* rootOp = exprOp.getRootOp();

    pushEmittedExpression(exprOp);

    FailureOr<int> precedence = getOperatorPrecedence(rootOp);
    if (failed(precedence)) return failure();

    pushExpressionPrecedence(*precedence);
    if (failed(emitOperation(*rootOp, false))) return failure();
    popExpressionPrecedence();

    if (!isNested)
        assert(
            emittedExpressionPrecedence.empty()
            && "expected precedence stack to be empty");

    restoreEmittedExpression();

    return success();
}

LogicalResult HLSCppEmitter::emitOperand(Value value)
{
    raw_indented_ostream &os = getOS();
    if (isPartOfCurrentExpression(value)) {
        Operation* defOp = value.getDefiningOp();
        assert(defOp && "value must be defined by an operation");
        FailureOr<int> precedence = getOperatorPrecedence(defOp);
        if (failed(precedence)) return failure();

        if (auto nestedExpr = dyn_cast<ExpressionOp>(defOp))
            return emitExpression(nestedExpr, true);

        bool encloseInParenthesis = *precedence <= getExpressionPrecedence();
        if (encloseInParenthesis) os << "(";
        pushExpressionPrecedence(*precedence);
        if (failed(emitOperation(*defOp, false))) return failure();
        if (encloseInParenthesis) os << ")";
        popExpressionPrecedence();
        return success();
    }

    if (!isa<BlockArgument>(value)) {
        Operation* defOp = value.getDefiningOp();
        if (auto exprOp = dyn_cast<ExpressionOp>(defOp))
            return emitExpression(exprOp, false);
        if (auto varOp = dyn_cast<VariableOp>(defOp)) {
            if (shouldEmitConstant(varOp)) {
                if (isa<ArithMaxOp, ArithMinOp>(getCurrentOperation())) {
                    os << "(";
                    if (failed(emitType(varOp.getLoc(), varOp.getType())))
                        return failure();
                    os << ")";
                }
                return emitAttribute(
                    varOp->getLoc(),
                    varOp.getInitNumberAttr(),
                    true);
            }
        }
    }

    os << getOrCreateName(value);
    return success();
}

StringRef HLSCppEmitter::getOrCreateName(Value value)
{
    auto it = valueNames.find(value);
    if (it != valueNames.end()) return it->second;

    Operation* definingOp = value.getDefiningOp();

    // Treat FuncOp and ForOp block arguments differently. If there is no
    // defining operation, it's a block argument.
    if (!definingOp) {
        if (auto blkArg = dyn_cast<BlockArgument>(value)) {
            Block* parentBlock = blkArg.getParentBlock();
            if (Operation* parentOp = parentBlock->getParentOp()) {
                if (isa<FuncOp>(parentOp)) return addNameForValue(value, "arg");
                if (isa<ForOp>(parentOp)) return addNameForValue(value, "idx");
                return addNameForValue(value, "v");
            }
        }
        return "";
    }

    // Otherwise, it's defined by another operation.
    std::string prefix;
    if (auto varOp = dyn_cast<VariableOp>(definingOp)) {
        if (varOp.getIsConst())
            prefix = "cst";
        else if (auto arrTy = dyn_cast<ArrayType>(varOp.getType()))
            prefix =
                isa<StreamType>(arrTy.getElementType()) ? "stream" : "array";
        else if (isa<StreamType>(varOp.getType()))
            prefix = "stream";
        else
            prefix = "var";
    } else {
        prefix = nameManager.getPrefix(definingOp);
    }
    if (!isa<FuncOp>(value.getParentBlock()->getParentOp())) prefix += "_tmp";
    return addNameForValue(value, prefix);
}

//===----------------------------------------------------------------------===//
// HLSCppEmitter::emitOperation
//===----------------------------------------------------------------------===//

LogicalResult
HLSCppEmitter::emitOperation(Operation &op, bool trailingSemicolon)
{
    currentOperation = &op;

    LogicalResult status =
        llvm::TypeSwitch<Operation*, LogicalResult>(&op)
            .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
            .Case<
                IncludeOp,
                FuncOp,
                CallOp,
                VariableOp,
                UpdateOp,
                IfOp,
                ForOp,
                ExpressionOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<YieldOp>([](auto op) {
                return op->emitError(
                    "YieldOp should never be emitted directly");
            })
            .Case<
                ArithAddOp,
                ArithSubOp,
                ArithMulOp,
                ArithDivOp,
                ArithRemOp,
                ArithAndOp,
                ArithOrOp,
                ArithShlOp,
                ArithShrOp,
                ArithMinOp,
                ArithMaxOp,
                ArithLogicalAndOp,
                ArithLogicalOrOp,
                ArithFusedOp,
                ArithCastOp,
                ArithCmpOp,
                ArithSelectOp,
                ArithDataRangeOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                ArrayReadOp,
                ArrayWriteOp,
                ArrayPointerReadOp,
                ArrayPointerWriteOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<StreamReadOp, StreamWriteOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<
                PragmaArrayPartitionOp,
                PragmaBindStorageOp,
                PragmaDataflowOp,
                PragmaPipelineOp,
                PragmaInlineOp,
                PragmaStreamOp,
                PragmaTopInterfaceOp,
                PragmaUnrollOp>(
                [&](auto op) { return printOperation(*this, op); })
            .Case<HelperLineBufferOp, HelperWindowOp, HelperAccumulateOp>(
                [](auto op) {
                    return op->emitError(
                        "expect no helper operations before translation");
                })
            .Default([](Operation* op) {
                return op->emitError("unsupported operation '")
                       << op->getName() << "'";
            });

    if (failed(status)) return failure();

    raw_indented_ostream &resultOS = getOS();
    trailingSemicolon &= !isa<
        ModuleOp,
        ExpressionOp,
        ForOp,
        FuncOp,
        IfOp,
        IncludeOp,
        PragmaInterface>(op);
    bool trailingNewLine =
        !isa<ExpressionOp>(op) && !op.getParentOfType<ExpressionOp>();
    if (auto varOp = dyn_cast<VariableOp>(op))
        if (shouldEmitConstant(varOp)) {
            trailingSemicolon = false;
            trailingNewLine = false;
        }
    resultOS << (trailingSemicolon ? ";" : "");
    resultOS << (trailingNewLine ? "\n" : "");

    if (isa<ArithMulOp>(op) && trailingNewLine)
        resultOS << "#pragma HLS BIND_OP variable="
                 << getOrCreateName(op.getResult(0)) << " op=mul impl=dsp\n";

    return success();
}

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult emithls::translateEmitHLSToCpp(Operation* op, raw_ostream &os)
{
    HLSCppEmitter emitter(os);
    return emitter.emitOperation(*op, false);
}
