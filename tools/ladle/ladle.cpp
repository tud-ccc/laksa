/// Central compiler driver for LAKSA.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace llvm;

namespace {

const char* const usage =
    "ladle - LAKSA compiler driver\n\n"
    "Dispatches to laksa-opt and laksa-translate to turn an input file into an "
    "output file through an optional pass pipeline and/or translation.\n\n"
    "Usage: ladle [options] <input file>\n\n"
    "Options:\n"
    "  -o, --output=<file>    Output filename (default: null); with --hls,\n"
    "                         the directory to write the artifacts into\n"
    "                         (default: the current directory)\n"
    "  -p, --passes=<value>   Passes to run through laksa-opt\n"
    "  -t, --translation=<name>\n"
    "                         Translation for laksa-translate to run\n"
    "      --hls              Run the convert-to-emithls and\n"
    "                         convert-to-laksa-emitc pipelines on the input,\n"
    "                         then emit the whole HLS artifact set below the\n"
    "                         output directory:\n"
    "                           hls.mlir            the lowered IR\n"
    "                           ref.mlir            the input in emitc\n"
    "                           hw/main.cpp         Vitis HLS input\n"
    "                           hw/run_hls.tcl\n"
    "                           hw/run_vivado.tcl\n"
    "                           hw/build.sh         builds the bitstream\n"
    "                           app/app.h           board-side driver API\n"
    "                           app/app.c           board-side driver\n"
    "                           app/app.dtsi\n"
    "                           app/ref.h           scalar reference\n"
    "                           app/ref.c           output checker\n"
    "                           app/run.sh          loads and checks it\n"
    "                         Cannot be combined with --passes or\n"
    "                         --translation\n"
    "      --mocasin          Run the DFG extraction pipeline and translate\n"
    "                         the resulting graph to Mocasin YAML. Cannot be\n"
    "                         combined with --hls, --passes, or\n"
    "                         --translation\n"
    "      --num-bram=<n>     With --hls, the number of BRAMs the pragma DSE\n"
    "                         may use (default: 288)\n"
    "      --num-dsp=<n>      With --hls, the number of DSPs the pragma DSE\n"
    "                         may use (default: 1248)\n"
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

/// The pass pipeline --hls runs before translating anything.
const char* const hlsPipeline = "convert-to-emithls";
const char* const dfgPipeline = "convert-to-dfg";
const char* const mocasinTranslation = "dfg-to-mocasin";
/// The EmitHLS IR --hls leaves behind next to the generated artifacts.
const char* const hlsIRFilename = "hls.mlir";
/// The scalar reference implementation comes from the same input lowered to
/// emitc instead, so that it computes what the design is meant to compute
/// without inheriting any of its HLS-specific restructuring.
const char* const refPipeline = "convert-to-laksa-emitc";
const char* const refIRFilename = "ref.mlir";
/// Subdirectory holding everything the Vitis HLS and Vivado runs need. Both
/// generated scripts refer to their inputs relatively, so they are meant to be
/// sourced from here, with the C++ source sitting next to them.
const char* const hlsToolSubdir = "hw";
/// Subdirectory holding what gets deployed to the board, i.e. the device tree
/// overlay and the userspace program built against the laksa-hls-kria-driver.
const char* const hlsAppSubdir = "app";
/// One artifact produced by --hls. The file names are not free-form: the
/// generated run_hls.tcl feeds "main.cpp" to add_files (see the
/// `hls-source-file` option of emithls-to-hls-tcl), and the generated app.c
/// includes "app.h" (see the `laksa-app-header` option of
/// emithls-to-laksa-app).
struct HLSArtifact {
    const char* ir; // the lowered IR the artifact is translated from
    const char* translation;
    const char* subdir;
    const char* filename;
};
const HLSArtifact hlsArtifacts[] = {
    {hlsIRFilename,              "emithls-to-cpp", hlsToolSubdir,       "main.cpp"},
    {hlsIRFilename,          "emithls-to-hls-tcl", hlsToolSubdir,    "run_hls.tcl"},
    {hlsIRFilename,       "emithls-to-vivado-tcl", hlsToolSubdir, "run_vivado.tcl"},
    {hlsIRFilename, "emithls-to-hls-build-script", hlsToolSubdir,       "build.sh"},
    {hlsIRFilename,     "emithls-to-laksa-header",  hlsAppSubdir,          "app.h"},
    {hlsIRFilename,        "emithls-to-laksa-app",  hlsAppSubdir,          "app.c"},
    {hlsIRFilename,        "emithls-to-kria-dtsi",  hlsAppSubdir,       "app.dtsi"},
    {refIRFilename,                 "mlir-to-cpp",  hlsAppSubdir,          "ref.h"},
    {refIRFilename,          "emitc-to-laksa-ref",  hlsAppSubdir,          "ref.c"},
    {hlsIRFilename, "emithls-to-laksa-run-script",  hlsAppSubdir,         "run.sh"},
};

struct Options {
    std::string inputFilename = "-";
    std::string outputFilename = "-";
    std::string passes;
    std::string translation;
    bool hls = false;
    bool mocasin = false;
    std::optional<unsigned> numBRAM;
    std::optional<unsigned> numDSP;
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
    auto takeCount = [&](StringRef flag, StringRef inlineValue, int &i) {
        std::string value = takeValue(flag, inlineValue, i);
        unsigned count;
        if (StringRef(value).getAsInteger(10, count)) {
            errs()
                << "ladle: option '" << flag
                << "' requires a non-negative integer, got '" << value << "'\n";
            exit(1);
        }
        return count;
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
        if (arg == "--hls") {
            opts.hls = true;
            continue;
        }
        if (arg == "--mocasin") {
            opts.mocasin = true;
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
        } else if (name == "--num-bram") {
            opts.numBRAM = takeCount(name, inlineValue, i);
        } else if (name == "--num-dsp") {
            opts.numDSP = takeCount(name, inlineValue, i);
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

    if (opts.hls && (!opts.passes.empty() || !opts.translation.empty())) {
        errs() << "ladle: '--hls' brings its own pipeline and translations; "
                  "it cannot be combined with '--passes' or '--translation'\n";
        exit(1);
    }

    if (opts.mocasin
        && (opts.hls || !opts.passes.empty() || !opts.translation.empty())) {
        errs() << "ladle: '--mocasin' brings its own pipeline and translation; "
                  "it cannot be combined with '--hls', '--passes', or "
                  "'--translation'\n";
        exit(1);
    }

    if (opts.mocasin) {
        opts.passes = dfgPipeline;
        opts.translation = mocasinTranslation;
    }

    if (!opts.hls && (opts.numBRAM || opts.numDSP)) {
        errs() << "ladle: '--num-bram' and '--num-dsp' set the resource budget "
                  "of '--hls' and cannot be used without it\n";
        exit(1);
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

/// Assembles the laksa-opt command line that runs `passes` over `input` and
/// writes the result to `output`.
std::vector<std::string> buildOptArgs(
    const Options &opts,
    StringRef optPath,
    StringRef passes,
    StringRef input,
    StringRef output)
{
    std::vector<std::string> args = {optPath.str(), input.str()};

    if (passes.contains('(')) {
        // Looks like a full textual pass pipeline, e.g.
        // 'builtin.module(canonicalize,cse)'; forward verbatim.
        args.push_back(("--pass-pipeline=" + passes).str());
    } else {
        // A comma-separated list of individual pass names.
        SmallVector<StringRef> passNames;
        passes.split(passNames, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (StringRef pass : passNames) args.push_back(asFlag(pass.trim()));
    }

    if (opts.verbosity >= 1) args.push_back("--mlir-print-ir-after-all");
    if (opts.verbosity >= 2) {
        // Only enable debug output for the LAKSA passes actually requested,
        // not every pass LAKSA knows about.
        std::vector<StringRef> debugTypes;
        for (const auto &p : laksaPassDebugTypes)
            if (passes.contains(p.flag)) debugTypes.push_back(p.debugType);
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
    args.push_back(output.str());
    return args;
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

/// Lowers the input to EmitHLS once and turns that IR into every artifact in
/// `hlsArtifacts`, all inside `opts.outputFilename` as a directory.
int runHLSFlow(const Options &opts, StringRef selfDir)
{
    // Unlike everywhere else, --hls writes a whole set of files, so -o names a
    // directory rather than a file; without one, land in the current one.
    StringRef outputDir =
        opts.outputFilename == "-" ? StringRef(".") : opts.outputFilename;
    if (auto ec = sys::fs::create_directories(outputDir)) {
        errs() << "ladle: failed to create output directory '" << outputDir
               << "': " << ec.message() << "\n";
        return 1;
    }

    for (const char* subdir : {hlsToolSubdir, hlsAppSubdir}) {
        SmallString<128> subdirPath(outputDir);
        sys::path::append(subdirPath, subdir);
        if (auto ec = sys::fs::create_directories(subdirPath)) {
            errs() << "ladle: failed to create output directory '" << subdirPath
                   << "': " << ec.message() << "\n";
            return 1;
        }
    }

    std::string optPath = findTool("laksa-opt", selfDir);
    std::string translatePath = findTool("laksa-translate", selfDir);

    // The lowered IR stays at the top level: it is an intermediate, not
    // something either the toolchain or the board consumes.
    auto lower = [&](StringRef pipeline, const char* filename) {
        SmallString<128> irPath(outputDir);
        sys::path::append(irPath, filename);
        errs() << "INFO: Lowering " << opts.inputFilename << " to " << filename
               << " through " << pipeline << "...\n";
        run(opts,
            optPath,
            buildOptArgs(opts, optPath, pipeline, opts.inputFilename, irPath));
    };
    // The resource budget reaches the pipeline as its options, which only the
    // textual pipeline form can carry; buildOptArgs forwards that verbatim.
    SmallVector<std::string> hlsOptions;
    if (opts.numBRAM)
        hlsOptions.push_back("available-bram=" + std::to_string(*opts.numBRAM));
    if (opts.numDSP)
        hlsOptions.push_back("available-dsp=" + std::to_string(*opts.numDSP));
    std::string hlsPasses = hlsPipeline;
    if (!hlsOptions.empty())
        hlsPasses = std::string("builtin.module(") + hlsPipeline + "{"
                    + join(hlsOptions, " ") + "})";
    lower(hlsPasses, hlsIRFilename);
    lower(refPipeline, refIRFilename);

    for (const auto &artifact : hlsArtifacts) {
        SmallString<128> irPath(outputDir);
        sys::path::append(irPath, artifact.ir);
        SmallString<128> artifactPath(outputDir);
        sys::path::append(artifactPath, artifact.subdir, artifact.filename);
        errs() << "INFO: Writing " << artifact.subdir << "/"
               << artifact.filename << " from " << artifact.ir << " through "
               << artifact.translation << "...\n";
        run(opts,
            translatePath,
            {translatePath,
             std::string(irPath),
             asFlag(artifact.translation),
             "-o",
             std::string(artifactPath)});
        if (sys::path::extension(artifactPath) != ".sh") continue;
        if (auto ec = sys::fs::setPermissions(
                artifactPath,
                sys::fs::perms::all_read | sys::fs::perms::owner_write
                    | sys::fs::perms::all_exe)) {
            errs() << "ladle: failed to make '" << artifactPath
                   << "' executable: " << ec.message() << "\n";
            return 1;
        }
    }

    errs() << "INFO: Done, wrote "
           << sizeof(hlsArtifacts) / sizeof(hlsArtifacts[0])
           << " artifacts below '" << outputDir << "'\n";
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options opts = parseArgs(argc, argv);

    bool runOpt = !opts.passes.empty();
    bool runTranslate = !opts.translation.empty();

    if (!opts.hls && !runOpt && !runTranslate) {
        errs() << "ladle: nothing to do; specify --passes, --translation, "
                  "--hls, and/or --mocasin\n";
        return 1;
    }

    std::string mainExe = sys::fs::getMainExecutable(argv[0], (void*)&main);
    std::string selfDir = std::string(sys::path::parent_path(mainExe));

    if (opts.hls) return runHLSFlow(opts, selfDir);

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

        run(opts,
            optPath,
            buildOptArgs(opts, optPath, opts.passes, stageInput, stageOutput));
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
