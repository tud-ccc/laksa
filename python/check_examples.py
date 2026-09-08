#!/usr/bin/env python3
"""Checks that python/examples/*.py reproduce the same IR as the matching
test/*.mlir file run through the same laksa-opt pass pipeline.

Each examples/<relpath>.py is matched against test/<relpath>.mlir. The pass
flags are parsed out of that .mlir file's `// RUN: laksa-opt %s <flags> |
FileCheck %s` line. The reference output comes from running laksa-opt directly;
the actual output comes from the last `print(module)` call made by the
Python script (its final, fully up-to-date module state).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

RUN_RE = re.compile(r"//\s*RUN:\s*\S*laksa-opt\s+%s\s*(.*?)\s*\|\s*FileCheck\s+%s")


def normalize(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def extract_last_module(text: str) -> str:
    lines = text.splitlines()
    last_idx = None
    for i, line in enumerate(lines):
        if line.rstrip() == "module {":
            last_idx = i
    if last_idx is None:
        return text
    start = last_idx
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:])


def check_one(
    laksa_opt: str, python: str, py_file: Path, mlir_file: Path
) -> str | None:
    """Returns an error message on failure, None on success."""
    if not mlir_file.exists():
        return f"no matching test file {mlir_file}"

    match = RUN_RE.search(mlir_file.read_text())
    if not match:
        return f"could not parse RUN line in {mlir_file}"
    flags = match.group(1).split()

    ref = subprocess.run(
        [laksa_opt, str(mlir_file), *flags], capture_output=True, text=True
    )
    if ref.returncode != 0:
        return f"laksa-opt failed:\n{ref.stderr}"

    got = subprocess.run([python, str(py_file)], capture_output=True, text=True)
    if got.returncode != 0:
        return f"python script failed:\n{got.stderr}"

    py_module = extract_last_module(got.stdout)
    if normalize(py_module) != normalize(ref.stdout):
        return (
            "output mismatch\n"
            f"--- python ({py_file}) ---\n{py_module}\n"
            f"--- laksa-opt ({mlir_file} {' '.join(flags)}) ---\n{ref.stdout}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laksa-opt", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--examples-root", default=str(Path(__file__).parent / "examples")
    )
    parser.add_argument(
        "--test-root", default=str(Path(__file__).parent.parent / "test")
    )
    args = parser.parse_args()

    examples_root = Path(args.examples_root)
    test_root = Path(args.test_root)

    failures = []
    checked = 0
    for py_file in sorted(examples_root.rglob("*.py")):
        rel = py_file.relative_to(examples_root)
        mlir_file = test_root / rel.with_suffix(".mlir")
        error = check_one(args.laksa_opt, args.python, py_file, mlir_file)
        checked += 1
        if error is not None:
            failures.append(f"{rel}: {error}")
        else:
            print(f"PASS: {rel}")

    print(f"\nChecked {checked} example(s), {len(failures)} failure(s).")
    for failure in failures:
        print(f"\nFAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
