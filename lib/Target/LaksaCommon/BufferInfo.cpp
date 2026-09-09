// Implementation of the buffer analysis shared by the laksa-hls-kria-driver
// translations
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Target/LaksaCommon/BufferInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::emithls;

static LogicalResult getElementBitWidth(Location loc, Type type, unsigned &bits)
{
    if (auto intTy = dyn_cast<IntegerType>(type)) {
        bits = intTy.getWidth();
        return success();
    }
    if (auto floatTy = dyn_cast<FloatType>(type)) {
        bits = floatTy.getIntOrFloatBitWidth();
        return success();
    }
    return emitError(
               loc,
               "unsupported pointer element type for buffer "
               "sizing: ")
           << type;
}

// Returns the number of times `op` executes, i.e. the product of the trip
// counts of every emithls.for enclosing it up to (but not including)
// `boundary`.
static int64_t countEnclosingIterations(Operation* op, Operation* boundary)
{
    int64_t count = 1;
    for (Operation* parent = op->getParentOp(); parent && parent != boundary;
         parent = parent->getParentOp())
        if (auto forOp = dyn_cast<ForOp>(parent)) count *= forOp.getTripCount();
    return count;
}

// Follows `pointer` through every array.ptr_read/array.ptr_write that
// touches it directly, and through every emithls.call that forwards it into
// a callee's argument, accumulating direction and size facts into `info`.
static LogicalResult analyzeBuffer(
    Value pointer,
    LaksaBufferInfo &info,
    llvm::SmallPtrSetImpl<Value> &visited)
{
    if (!visited.insert(pointer).second) return success();

    for (Operation* user : pointer.getUsers()) {
        if (auto readOp = dyn_cast<ArrayPointerReadOp>(user)) {
            if (readOp.getPointer() != pointer) continue;
            unsigned bits;
            if (failed(getElementBitWidth(
                    readOp.getLoc(),
                    readOp.getResult().getType(),
                    bits)))
                return failure();
            info.isReadByKernel = true;
            info.elementBits = bits;
            FuncOp parentFunc = readOp->getParentOfType<FuncOp>();
            info.numElements = std::max(
                info.numElements,
                countEnclosingIterations(readOp, parentFunc));
        } else if (auto writeOp = dyn_cast<ArrayPointerWriteOp>(user)) {
            if (writeOp.getPointer() != pointer) continue;
            unsigned bits;
            if (failed(getElementBitWidth(
                    writeOp.getLoc(),
                    writeOp.getValue().getType(),
                    bits)))
                return failure();
            info.isWrittenByKernel = true;
            info.elementBits = bits;
            FuncOp parentFunc = writeOp->getParentOfType<FuncOp>();
            info.numElements = std::max(
                info.numElements,
                countEnclosingIterations(writeOp, parentFunc));
        } else if (auto callOp = dyn_cast<CallOp>(user)) {
            auto moduleOp = callOp->getParentOfType<ModuleOp>();
            auto callee = moduleOp.lookupSymbol<FuncOp>(callOp.getCalleeAttr());
            if (!callee) continue;
            for (auto [idx, operand] :
                 llvm::enumerate(callOp.getArgOperands())) {
                if (operand != pointer) continue;
                if (failed(
                        analyzeBuffer(callee.getArgument(idx), info, visited)))
                    return failure();
            }
        }
    }
    return success();
}

FailureOr<FuncOp> emithls::findLaksaTopFunc(ModuleOp moduleOp)
{
    FuncOp topFunc = nullptr;
    auto result = moduleOp.walk([&](PragmaTopInterfaceOp topInterfaceOp) {
        FuncOp funcOp = topInterfaceOp->getParentOfType<FuncOp>();
        if (!funcOp || topFunc) return WalkResult::interrupt();
        topFunc = funcOp;
        return WalkResult::advance();
    });
    if (result.wasInterrupted() || !topFunc)
        return moduleOp->emitError(
            "expected exactly one function containing a top_interface "
            "pragma");
    return topFunc;
}

LogicalResult emithls::collectLaksaBufferInfos(
    FuncOp topFunc,
    SmallVectorImpl<LaksaBufferInfo> &buffers)
{
    buffers.assign(topFunc.getNumArguments(), LaksaBufferInfo{});
    for (auto [idx, info] : llvm::enumerate(buffers)) {
        llvm::SmallPtrSet<Value, 8> visited;
        if (failed(analyzeBuffer(topFunc.getArgument(idx), info, visited)))
            return failure();
        if (info.numElements == 0)
            return topFunc.emitError(
                       "could not determine buffer size for argument ")
                   << idx
                   << "; expected an array.ptr_read or array.ptr_write "
                      "reachable through it";
    }
    return success();
}

std::string emithls::getLaksaMacroPrefix(StringRef designName)
{
    std::string prefix = designName.str();
    for (char &c : prefix) c = llvm::toUpper(c);
    return prefix;
}
