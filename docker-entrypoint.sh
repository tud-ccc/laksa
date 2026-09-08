#!/bin/sh
set -eu

lic="${GRB_LICENSE_FILE:-/opt/gurobi/gurobi.lic}"

if [ ! -f "${lic}" ]; then
    echo "warning: no Gurobi license found at ${lic}." >&2
    echo "  mount one with:  -v /path/to/gurobi.lic:${lic}:ro" >&2
    echo "  Everything except Gurobi-backed passes still works." >&2
fi

exec "$@"
