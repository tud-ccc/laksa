// Implementation of translation EmitCToLaksaRef
//
// @file
// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Target/EmitCToLaksaRef/LaksaRefEmitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::laksa;

namespace {

llvm::cl::opt<std::string> clLaksaRefHeader(
    "laksa-ref-header",
    llvm::cl::desc(
        "Header the generated program takes the reference kernel from, as "
        "produced by the mlir-to-cpp translation"),
    llvm::cl::init("ref.h"));

llvm::cl::opt<unsigned> clLaksaRefMaxReported(
    "laksa-ref-max-reported",
    llvm::cl::desc(
        "How many differing elements the generated program spells out before "
        "going quiet"),
    llvm::cl::init(16));

// One argument of the reference kernel.
struct RefArg {
    emitc::ArrayType type;
    std::string elementType;
    std::string format; // printf conversion for one element
    std::string cast;   // what to cast an element to before printing it
    bool isRead = false;
    bool isWritten = false;
    std::string name;
    std::string boardName; // holds output<n>.bin, only when written
    std::string file;      // input<n>.bin, or output<n>.bin when write-only
    std::string fileMacro;
    std::string boardFile; // output<n>.bin when both read and written
    std::string boardFileMacro;
};

// The C spelling the mlir-to-cpp translation gives an element type.
FailureOr<std::string> getCElementType(Type type)
{
    if (auto intTy = dyn_cast<IntegerType>(type)) {
        unsigned width = intTy.getWidth();
        if (width == 1) return std::string("bool");
        if (width == 8 || width == 16 || width == 32 || width == 64)
            return (intTy.isUnsigned() ? std::string("uint")
                                       : std::string("int"))
                   + std::to_string(width) + "_t";
    }
    if (isa<Float32Type>(type)) return std::string("float");
    if (isa<Float64Type>(type)) return std::string("double");
    return failure();
}

// The only emitc.func with a body, which is the reference kernel.
FailureOr<emitc::FuncOp> findRefFunc(ModuleOp moduleOp)
{
    emitc::FuncOp found;
    for (auto funcOp : moduleOp.getOps<emitc::FuncOp>()) {
        if (funcOp.isExternal()) continue;
        if (found)
            return moduleOp.emitError(
                "expected exactly one emitc.func to check against");
        found = funcOp;
    }
    if (!found)
        return moduleOp.emitError("expected an emitc.func to check against");
    return found;
}

// Whether the kernel loads from arg, stores to it, or both.
void classifyArg(BlockArgument arg, RefArg &refArg)
{
    for (Operation* user : arg.getUsers()) {
        auto subscriptOp = dyn_cast<emitc::SubscriptOp>(user);
        if (!subscriptOp || subscriptOp->getOperand(0) != arg) continue;
        Value element = subscriptOp->getResult(0);
        for (Operation* access : element.getUsers()) {
            if (isa<emitc::LoadOp>(access)) refArg.isRead = true;
            if (auto assignOp = dyn_cast<emitc::AssignOp>(access))
                if (assignOp.getVar() == element) refArg.isWritten = true;
        }
    }
}

void emitArrayDecl(raw_ostream &os, const RefArg &arg, StringRef name)
{
    os << "static " << arg.elementType << " " << name;
    for (int64_t extent : arg.type.getShape()) os << "[" << extent << "]";
    os << ";\n";
}

void emitReadAll(raw_ostream &os)
{
    os << "static int read_all(const char* path, void* data, size_t size)\n"
          "{\n"
          "    FILE* file = fopen(path, \"rb\");\n"
          "    if (!file) {\n"
          "        fprintf(stderr, \"cannot open %s for reading\\n\", path);\n"
          "        return -1;\n"
          "    }\n"
          "    size_t got = fread(data, 1, size, file);\n"
          "    int trailing = fgetc(file) != EOF;\n"
          "    fclose(file);\n"
          "    if (got != size) {\n"
          "        fprintf(stderr, \"%s: expected %zu bytes, got %zu\\n\", "
          "path, size, got);\n"
          "        return -1;\n"
          "    }\n"
          "    if (trailing)\n"
          "        fprintf(stderr, \"%s: ignoring bytes past %zu\\n\", path, "
          "size);\n"
          "    return 0;\n"
          "}\n";
}

void emitCompare(raw_ostream &os, const RefArg &arg)
{
    ArrayRef<int64_t> shape = arg.type.getShape();
    int64_t numElements = 1;
    for (int64_t extent : shape) numElements *= extent;

    os << "static long compare_" << arg.name << "(void)\n{\n"
       << "    long mismatches = 0;\n";

    std::string indent = "    ";
    for (auto [dim, extent] : llvm::enumerate(shape)) {
        os << indent << "for (size_t i" << dim << " = 0; i" << dim << " < "
           << extent << "; ++i" << dim << ") {\n";
        indent += "    ";
    }

    std::string subscript;
    for (size_t dim = 0; dim < shape.size(); ++dim)
        subscript += "[i" + std::to_string(dim) + "]";

    os << indent << arg.cast << " want = " << arg.name << subscript << ";\n"
       << indent << arg.cast << " have = " << arg.boardName << subscript
       << ";\n"
       << indent << "if (want == have) continue;\n"
       << indent << "if (mismatches < MAX_REPORTED)\n"
       << indent << "    printf(\"" << arg.name;
    for (size_t dim = 0; dim < shape.size(); ++dim) os << "[%zu]";
    os << ": expected " << arg.format << ", got " << arg.format << "\\n\"";
    for (size_t dim = 0; dim < shape.size(); ++dim) os << ", i" << dim;
    os << ", want, have);\n"
       << indent << "else if (mismatches == MAX_REPORTED)\n"
       << indent << "    printf(\"...\\n\");\n"
       << indent << "++mismatches;\n";

    for (size_t dim = shape.size(); dim > 0; --dim) {
        indent.resize(indent.size() - 4);
        os << indent << "}\n";
    }

    os << "\n"
       << "    if (mismatches)\n"
       << "        printf(" << arg.boardFileMacro << " \": %ld of "
       << numElements << " elements differ from the reference\\n\", "
       << "mismatches);\n"
       << "    else\n"
       << "        printf(" << arg.boardFileMacro << " \": all " << numElements
       << " elements match the reference\\n\");\n"
       << "    return mismatches;\n"
          "}\n";
}

} // namespace

//===----------------------------------------------------------------------===//
// Translation entry point
//===----------------------------------------------------------------------===//

LogicalResult laksa::translateEmitCToLaksaRef(ModuleOp op, raw_ostream &os)
{
    FailureOr<emitc::FuncOp> refFunc = findRefFunc(op);
    if (failed(refFunc)) return failure();

    Block &body = refFunc->getFunctionBody().front();
    SmallVector<RefArg, 4> args(body.getNumArguments());
    unsigned numInputs = 0;
    unsigned numOutputs = 0;
    for (auto [idx, blockArg] : llvm::enumerate(body.getArguments())) {
        RefArg &arg = args[idx];
        arg.type = dyn_cast<emitc::ArrayType>(blockArg.getType());
        if (!arg.type)
            return refFunc->emitError("argument ")
                   << idx << " is not an emitc.array";

        FailureOr<std::string> elementType =
            getCElementType(arg.type.getElementType());
        if (failed(elementType))
            return refFunc->emitError("argument ")
                   << idx << " has an element type without a C spelling";
        arg.elementType = *elementType;
        bool isFloat = isa<FloatType>(arg.type.getElementType());
        arg.format = isFloat ? "%g" : "%ld";
        arg.cast = isFloat ? "double" : "long";

        classifyArg(blockArg, arg);
        if (!arg.isRead && !arg.isWritten)
            return refFunc->emitError("argument ")
                   << idx << " is neither read nor written by the reference";

        arg.name = "arg" + std::to_string(idx);
        std::string argMacro = "ARG" + std::to_string(idx);
        if (arg.isRead) {
            arg.file = "input" + std::to_string(numInputs++) + ".bin";
            arg.fileMacro = argMacro + "_IN_FILE";
        }
        if (arg.isWritten) {
            arg.boardName = arg.name + "_board";
            arg.boardFile = "output" + std::to_string(numOutputs++) + ".bin";
            arg.boardFileMacro = argMacro + "_OUT_FILE";
            if (!arg.isRead) {
                arg.file = arg.boardFile;
                arg.fileMacro = arg.boardFileMacro;
            }
        }
    }

    StringRef kernelName = refFunc->getSymName();
    bool renameKernel = kernelName == "main";
    std::string kernelCall = renameKernel ? "ref" : kernelName.str();

    os << "// Generated by ladle -t emitc-to-laksa-ref.\n"
          "// Checks what the design wrote on the board against the scalar "
          "reference in\n"
          "// \""
       << clLaksaRefHeader
       << "\", generated by\n"
          "//   ladle <input>.mlir -t mlir-to-cpp -o "
       << clLaksaRefHeader
       << "\n"
          "//\n"
          "// Build, with \""
       << clLaksaRefHeader
       << "\" next to this file:\n"
          "//   gcc -O2 -I. -o ref ref.c\n"
          "//\n"
          "// Run it where app.c left its .bin files:\n"
          "//   ./ref\n"
          "\n"
          "#include <stdbool.h>\n"
          "#include <stddef.h>\n"
          "#include <stdint.h>\n"
          "#include <stdio.h>\n"
          "#include <stdlib.h>\n"
          "#include <string.h>\n"
          "\n";

    if (renameKernel)
        os << "// \"" << clLaksaRefHeader
           << "\" names the kernel 'main', the name it carries in the input\n"
              "// program, so rename it on the way in.\n"
              "#define main "
           << kernelCall << "\n";
    os << "#include \"" << clLaksaRefHeader << "\"\n";
    if (renameKernel) os << "#undef main\n";
    os << "\n";

    for (const RefArg &arg : args) {
        if (!arg.fileMacro.empty())
            os << "#define " << arg.fileMacro << " \"" << arg.file << "\"\n";
        if (arg.isRead && arg.isWritten)
            os << "#define " << arg.boardFileMacro << " \"" << arg.boardFile
               << "\"\n";
    }

    os << "\n"
          "// How many differing elements to spell out before going quiet.\n"
          "#define MAX_REPORTED "
       << clLaksaRefMaxReported << "\n\n";

    for (const RefArg &arg : args) {
        emitArrayDecl(os, arg, arg.name);
        if (arg.isWritten) emitArrayDecl(os, arg, arg.boardName);
    }

    os << "\n";
    emitReadAll(os);

    for (const RefArg &arg : args) {
        if (!arg.isWritten) continue;
        os << "\n";
        emitCompare(os, arg);
    }

    os << "\nint main(void)\n{\n";
    for (const RefArg &arg : args) {
        if (arg.isRead)
            os << "    if (read_all(" << arg.fileMacro << ", " << arg.name
               << ", sizeof(" << arg.name << ")) != 0) return EXIT_FAILURE;\n";
        if (arg.isWritten)
            os << "    if (read_all(" << arg.boardFileMacro << ", "
               << arg.boardName << ", sizeof(" << arg.boardName
               << ")) != 0) return EXIT_FAILURE;\n";
    }

    os << "\n    " << kernelCall << "(";
    llvm::interleaveComma(args, os, [&](const RefArg &arg) { os << arg.name; });
    os << ");\n\n"
          "    long mismatches = 0;\n";
    for (const RefArg &arg : args)
        if (arg.isWritten)
            os << "    mismatches += compare_" << arg.name << "();\n";

    os << "    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;\n"
          "}\n";

    return success();
}
