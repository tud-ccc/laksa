// Implementation of translation EmitHLSToLaksaHeader
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Target/EmitHLSToLaksaHeader/LaksaHeaderEmitter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::emithls;

namespace {

// Facts about one top-level m_axi argument, gathered by following it through
// every array.ptr_read/array.ptr_write that reads or writes through it --
// directly, or through emithls.call operands forwarded into dataflow node
// functions.
struct BufferInfo {
    bool isReadByKernel = false;    // host must write the buffer before launch
    bool isWrittenByKernel = false; // host may read the buffer after launch
    int64_t numElements = 0;
    unsigned elementBits = 0;
};

} // namespace

// Finds the function marked as the design's top function, i.e. the one whose
// body contains a PragmaTopInterfaceOp. Its symbol name doubles as the
// generated header's macro prefix and include guard.
static FailureOr<FuncOp> findTopFunc(ModuleOp moduleOp)
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
    BufferInfo &info,
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

static std::string toUpper(StringRef s)
{
    std::string result = s.str();
    for (char &c : result) c = llvm::toUpper(c);
    return result;
}

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult
emithls::translateEmitHLSToLaksaHeader(ModuleOp op, raw_ostream &os)
{
    FailureOr<FuncOp> topFunc = findTopFunc(op);
    if (failed(topFunc)) return failure();
    StringRef designName = topFunc->getSymName();
    unsigned numArgs = topFunc->getNumArguments();
    std::string prefix = toUpper(designName);

    SmallVector<BufferInfo> buffers(numArgs);
    for (unsigned i = 0; i < numArgs; ++i) {
        llvm::SmallPtrSet<Value, 8> visited;
        if (failed(analyzeBuffer(topFunc->getArgument(i), buffers[i], visited)))
            return failure();
        if (buffers[i].numElements == 0)
            return topFunc->emitError(
                       "could not determine buffer size for argument ")
                   << i
                   << "; expected an array.ptr_read or array.ptr_write "
                      "reachable through it";
    }

    std::string guard = prefix + "_LAKSA_CONFIG_H";

    os << "// Generated by ladle -t emithls-to-laksa-header.\n"
       << "// Buffer sizes and AXI-Lite register offsets for '" << designName
       << "', matching the\n"
       << "// argument layout expected by laksa.h's IOCTL_START_KERNEL.\n"
       << "#ifndef " << guard << "\n"
       << "#define " << guard << "\n\n"
       << "#define " << prefix << "_NUM_ARGS " << numArgs << "\n\n"
       << "#define " << prefix << "_CTRL_OFFSET 0x00 /* ap_ctrl */\n\n";

    // Register offsets follow Vitis HLS's fixed AXI-Lite control layout for
    // m_axi arguments: ap_ctrl/gie/ier/isr occupy 0x00-0x0c, then each
    // 64-bit pointer argument gets a 0xc-byte slot starting at 0x10 (low
    // word, high word, one reserved word).
    for (unsigned i = 0; i < numArgs; ++i) {
        unsigned lowOffset = 0x10 + i * 0xc;
        unsigned highOffset = lowOffset + 0x4;
        const BufferInfo &info = buffers[i];
        unsigned elementBytes = (info.elementBits + 7) / 8;
        int64_t sizeBytes = info.numElements * elementBytes;
        StringRef direction = info.isReadByKernel && info.isWrittenByKernel
                                  ? "input/output"
                              : info.isReadByKernel ? "input"
                                                    : "output";

        os << "/* arg" << i << ": " << direction << " buffer, "
           << info.numElements << " x " << info.elementBits
           << "-bit elements */\n"
           << "#define " << prefix << "_ARG" << i << "_LOW_OFFSET  0x";
        os.write_hex(lowOffset);
        os << "\n#define " << prefix << "_ARG" << i << "_HIGH_OFFSET 0x";
        os.write_hex(highOffset);
        os << "\n#define " << prefix << "_ARG" << i << "_SIZE        "
           << sizeBytes << " /* bytes */\n\n";
    }

    os << "#endif /* " << guard << " */\n";

    return success();
}
