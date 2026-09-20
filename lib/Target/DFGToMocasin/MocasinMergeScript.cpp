/// Implementation of the Mocasin profile-merging script emitter.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Target/DFGToMocasin/MocasinEmitter.h"

using namespace mlir;

LogicalResult dfg::emitMocasinMergeScript(raw_ostream &os)
{
    os << R"SCRIPT(#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

exec laksa-merge-profiles \
    "$script_dir/template.yaml" \
    --profiles-dir "$script_dir/../profiles" \
    --boundary CortexA53 \
    -o "$script_dir/application.yaml"
)SCRIPT";
    return success();
}
