/// Central compiler driver for LAKSA.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

const char* const usage =
    "ladle - LAKSA compiler driver\n\n"
    "Dispatches to laksa-opt and laksa-translate to turn an input file into an "
    "output file through an optional pass pipeline and/or translation.\n\n"
    "Usage: ladle [options] <input file>\n\n"
    "Options:\n"
    "  -o, --output=<file>    Output filename (default: null)\n"
    "  -p, --passes=<value>   Passes to run through laksa-opt\n"
    "  -t, --translation=<name>\n"
    "                         Translation for laksa-translate to run\n"
    "  -v, --verbose          Increase laksa-opt's debug verbosity\n"
    "                         (repeatable, e.g. -vvv); has no effect on\n"
    "                         laksa-translate. Each level adds on top of\n"
    "                         the previous one:\n"
    "                           1: print each stage's command line, and\n"
    "                              pass --mlir-print-ir-after-all\n"
    "                           2: also enable every LAKSA pass's own\n"
    "                              debug output (-debug-only=<pass>)\n"
    "                           3: also enable MLIR's dialect-conversion\n"
    "                              debug output\n"
    "  -h, --help             Print this help\n";

// Map each pass to its own defined debug type
struct PassDebugType {
    const char* flag;
    const char* debugType;
};
const PassDebugType laksaPassDebugTypes[] = {
    {          "dfg-operator-to-process",       "dfg-operator-to-process"},
    {          "dfg-inline-embed-region",       "dfg-inline-embed-region"},
    {             "dfg-io-normalization",          "dfg-io-normalization"},
    {     "dfg-channel-fanout-expansion",  "dfg-channel-fanout-expansion"},
    {           "dfg-collapse-unit-dims",        "dfg-collapse-unit-dims"},
    {         "emithls-add-io-functions",      "emithls-add-io-functions"},
    {          "emithls-fold-expression",       "emithls-fold-expression"},
    {            "emithls-fuse-operator",         "emithls-fuse-operator"},
    {          "emithls-insert-includes",       "emithls-insert-includes"},
    {              "emithls-loop-fusion",           "emithls-loop-fusion"},
    {         "emithls-merge-cast-chain",      "emithls-merge-cast-chain"},
    {               "emithls-pragma-dse",            "emithls-pragma-dse"},
    {         "emithls-pragma-insertion",      "emithls-pragma-insertion"},
    {          "emithls-resolve-helpers",       "emithls-resolve-helpers"},
    {    "func-outline-computation-leaf", "func-outline-computation-leaf"},
    {            "linalg-map-to-generic",         "linalg-map-to-generic"},
    {            "linalg-soft-transpose",         "linalg-soft-transpose"},
    {     "linalg-scalarize-splat-dense",  "linalg-scalarize-splat-dense"},
    {        "convert-affine-to-emithls",             "affine-to-emithls"},
    {         "convert-arith-to-emithls",              "arith-to-emithls"},
    {           "convert-dfg-to-emithls",                "dfg-to-emithls"},
    {              "convert-func-to-dfg",                   "func-to-dfg"},
    {         "convert-index-to-emithls",              "index-to-emithls"},
    {    "convert-linalg-to-laksa-loops",         "linalg-to-laksa-loops"},
    {"convert-memref-pad-to-laksa-loops",     "memref-pad-to-laksa-loops"},
};

struct Options {
    std::string inputFilename = "-";
    std::string outputFilename = "-";
    std::string passes;
    std::string translation;
    unsigned verbosity = 0;
};

/// Parses argv into `opts`, exiting the process on `--help` or a usage
/// error.
Options parseArgs(int argc, char** argv)
{
    Options opts;
    bool haveInput = false;
    bool haveTranslation = false;

    auto takeValue =
        [&](StringRef flag, StringRef inlineValue, int &i) -> std::string {
        if (!inlineValue.empty()) return inlineValue.str();
        if (i + 1 >= argc) {
            errs() << "ladle: option '" << flag << "' requires a value\n";
            exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        StringRef arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            outs() << usage;
            exit(0);
        }
        if (arg == "--verbose") {
            ++opts.verbosity;
            continue;
        }
        if (arg.size() >= 2 && arg[0] == '-' && arg[1] != '-'
            && arg.drop_front().find_first_not_of('v') == StringRef::npos) {
            // -v, -vv, -vvv, ... each 'v' bumps the verbosity level by one.
            opts.verbosity += arg.size() - 1;
            continue;
        }

        StringRef name = arg;
        StringRef inlineValue;
        if (arg.starts_with("--")) {
            size_t eq = arg.find('=');
            if (eq != StringRef::npos) {
                name = arg.substr(0, eq);
                inlineValue = arg.substr(eq + 1);
            }
        }

        if (name == "-o" || name == "--output") {
            opts.outputFilename = takeValue(name, inlineValue, i);
        } else if (name == "-p" || name == "--passes") {
            opts.passes = takeValue(name, inlineValue, i);
        } else if (name == "-t" || name == "--translation") {
            if (haveTranslation) {
                errs() << "ladle: '" << name
                       << "' may only be given once; laksa-translate can "
                          "only run a single translation\n";
                exit(1);
            }
            opts.translation = takeValue(name, inlineValue, i);
            haveTranslation = true;
        } else if (arg.starts_with("-") && arg != "-") {
            errs() << "ladle: unknown option '" << arg << "'\n";
            exit(1);
        } else if (!haveInput) {
            opts.inputFilename = arg.str();
            haveInput = true;
        } else {
            errs() << "ladle: unexpected extra argument '" << arg << "'\n";
            exit(1);
        }
    }

    return opts;
}

/// Locates `name` next to the running `ladle` binary, falling back to PATH.
std::string findTool(StringRef name, StringRef selfDir)
{
    SmallString<128> candidate(selfDir);
    sys::path::append(candidate, name);
    if (sys::fs::can_execute(candidate)) return std::string(candidate);

    if (auto found = sys::findProgramByName(name)) return *found;

    errs() << "ladle: could not find '" << name
           << "' next to ladle or on "
              "PATH\n";
    exit(1);
}

/// Normalizes a user-provided pass/translation name into a `--flag`.
std::string asFlag(StringRef name)
{
    if (name.starts_with("-")) return name.str();
    return ("--" + name).str();
}

/// Runs `program` with `args` (args[0] is conventionally the program
/// itself), aborting ladle with a diagnostic if it fails.
void run(const Options &opts, StringRef program, ArrayRef<std::string> args)
{
    if (opts.verbosity >= 1) {
        errs() << "+";
        for (auto &a : args) errs() << ' ' << a;
        errs() << '\n';
    }

    SmallVector<StringRef> refs(args.begin(), args.end());
    std::string errMsg;
    int rc = sys::ExecuteAndWait(
        program,
        refs,
        /*Env=*/std::nullopt,
        /*Redirects=*/{},
        /*SecondsToWait=*/0,
        /*MemoryLimit=*/0,
        &errMsg);
    if (rc != 0) {
        errs() << "ladle: '" << program << "' failed";
        if (!errMsg.empty()) errs() << ": " << errMsg;
        errs() << " (exit code " << rc << ")\n";
        exit(1);
    }
}

} // namespace

int main(int argc, char** argv)
{
    Options opts = parseArgs(argc, argv);

    bool runOpt = !opts.passes.empty();
    bool runTranslate = !opts.translation.empty();

    if (!runOpt && !runTranslate) {
        errs() << "ladle: nothing to do; specify --passes and/or "
                  "--translation\n";
        return 1;
    }

    std::string mainExe = sys::fs::getMainExecutable(argv[0], (void*)&main);
    std::string selfDir = std::string(sys::path::parent_path(mainExe));

    SmallString<128> tempPath;
    bool haveTemp = false;
    scope_exit cleanup([&] {
        if (haveTemp) {
            if (auto ec = sys::fs::remove(tempPath))
                errs() << "ladle: warning: failed to remove temporary file '"
                       << tempPath << "': " << ec.message() << "\n";
        }
    });

    std::string stageInput = opts.inputFilename;

    if (runOpt) {
        std::string optPath = findTool("laksa-opt", selfDir);

        std::string stageOutput = opts.outputFilename;
        if (runTranslate) {
            int fd;
            if (auto ec = sys::fs::createTemporaryFile(
                    "ladle",
                    "mlir",
                    fd,
                    tempPath)) {
                errs() << "ladle: failed to create temporary file: "
                       << ec.message() << "\n";
                return 1;
            }
            if (auto ec = sys::Process::SafelyCloseFileDescriptor(fd)) {
                errs()
                    << "ladle: failed to close temporary file: " << ec.message()
                    << "\n";
                return 1;
            }
            haveTemp = true;
            stageOutput = std::string(tempPath);
        }

        std::vector<std::string> args = {optPath, stageInput};
        if (StringRef(opts.passes).contains('(')) {
            // Looks like a full textual pass pipeline, e.g.
            // 'builtin.module(canonicalize,cse)'; forward verbatim.
            args.push_back("--pass-pipeline=" + opts.passes);
        } else {
            // A comma-separated list of individual pass names.
            SmallVector<StringRef> passNames;
            StringRef(opts.passes)
                .split(passNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
            for (StringRef pass : passNames)
                args.push_back(asFlag(pass.trim()));
        }

        if (opts.verbosity >= 1) args.push_back("--mlir-print-ir-after-all");
        if (opts.verbosity >= 2) {
            // Only enable debug output for the LAKSA passes actually
            // requested via --passes, not every pass LAKSA knows about.
            std::vector<StringRef> debugTypes;
            for (const auto &p : laksaPassDebugTypes)
                if (StringRef(opts.passes).contains(p.flag))
                    debugTypes.push_back(p.debugType);
            if (opts.verbosity >= 3) debugTypes.push_back("dialect-conversion");

            if (!debugTypes.empty()) {
                std::string debugOnly = "--debug-only=";
                for (size_t i = 0; i < debugTypes.size(); ++i) {
                    if (i != 0) debugOnly += ',';
                    debugOnly += debugTypes[i];
                }
                args.push_back(debugOnly);
            }
        }

        args.push_back("-o");
        args.push_back(stageOutput);

        run(opts, optPath, args);
        stageInput = stageOutput;
    }

    if (runTranslate) {
        std::string translatePath = findTool("laksa-translate", selfDir);

        std::vector<std::string> args = {
            translatePath,
            stageInput,
            asFlag(opts.translation),
            "-o",
            opts.outputFilename};

        run(opts, translatePath, args);
    }

    return 0;
}
