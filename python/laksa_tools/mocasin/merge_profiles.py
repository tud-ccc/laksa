# Merge LAKSA execution-profile fragments into a Mocasin application.
#
# @file
# @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

import argparse
import copy
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path
import sys
from typing import Any

import yaml


class ProfileMergeError(ValueError):
    """A profile fragment cannot be merged into the application template."""


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileMergeError(f"{description} must be a mapping")
    return value


def _profiles(document: Any, source: str) -> Mapping[str, Any]:
    root = _mapping(document, source)
    execution = _mapping(root.get("execution"), f"{source}: execution")
    processes = _mapping(
        execution.get("processes"), f"{source}: execution.processes"
    )
    return _mapping(
        processes.get("profiles"),
        f"{source}: execution.processes.profiles",
    )


def _validate_fragment_shape(fragment: Mapping[str, Any], source: str) -> None:
    if set(fragment) != {"execution"}:
        raise ProfileMergeError(
            f"{source} must contain only the 'execution' section"
        )
    execution = _mapping(fragment["execution"], f"{source}: execution")
    if set(execution) != {"processes"}:
        raise ProfileMergeError(
            f"{source}: execution must contain only 'processes'"
        )
    processes = _mapping(
        execution["processes"], f"{source}: execution.processes"
    )
    if set(processes) != {"profiles"}:
        raise ProfileMergeError(
            f"{source}: execution.processes must contain only 'profiles'"
        )


def merge_profiles(
    template: Mapping[str, Any],
    fragments: Iterable[tuple[str, Mapping[str, Any]]],
    boundary: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``template`` with profile fragments merged into it."""

    result = copy.deepcopy(template)
    target_profiles = _profiles(result, "template")

    if boundary is not None:
        processor_type, cycles = boundary
        boundary_fragment = {
            "execution": {
                "processes": {
                    "profiles": {
                        "boundary": {processor_type: {"cycles": cycles}}
                    }
                }
            }
        }
        fragments = chain(fragments, [("--boundary", boundary_fragment)])

    for source, fragment in fragments:
        _validate_fragment_shape(fragment, source)
        fragment_profiles = _profiles(fragment, source)
        for profile_name, processors_value in fragment_profiles.items():
            if profile_name not in target_profiles:
                raise ProfileMergeError(
                    f"{source} contains unknown profile '{profile_name}'"
                )
            processors = _mapping(
                processors_value, f"{source}: profile '{profile_name}'"
            )
            if not processors:
                raise ProfileMergeError(
                    f"{source}: profile '{profile_name}' is empty"
                )

            target = _mapping(
                target_profiles[profile_name],
                f"template: profile '{profile_name}'",
            )
            added_real_profile = False
            for processor_type, entry_value in processors.items():
                if not isinstance(processor_type, str) or not processor_type:
                    raise ProfileMergeError(
                        f"{source}: profile '{profile_name}' has an invalid "
                        "processor type"
                    )
                if processor_type == "UNKNOWN":
                    raise ProfileMergeError(
                        f"{source}: processor type 'UNKNOWN' is reserved for "
                        "template placeholders"
                    )
                entry = _mapping(
                    entry_value,
                    f"{source}: profile '{profile_name}.{processor_type}'",
                )
                cycles = entry.get("cycles")
                if type(cycles) is not int or cycles < 0:
                    raise ProfileMergeError(
                        f"{source}: profile '{profile_name}.{processor_type}' "
                        "must contain a non-negative integer 'cycles' value"
                    )
                if processor_type in target:
                    raise ProfileMergeError(
                        f"{source} duplicates profile "
                        f"'{profile_name}.{processor_type}'"
                    )
                target[processor_type] = copy.deepcopy(entry)
                added_real_profile = True

            if added_real_profile:
                target.pop("UNKNOWN", None)

    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return _mapping(document, str(path))


def _unresolved_profiles(document: Mapping[str, Any]) -> list[str]:
    profiles = _profiles(document, "merged application")
    return sorted(
        profile_name
        for profile_name, processors_value in profiles.items()
        if "UNKNOWN"
        in _mapping(
            processors_value,
            f"merged application: profile '{profile_name}'",
        )
    )


def _parse_boundary(value: str) -> tuple[str, int]:
    processor_type, separator, cycles_text = value.partition("=")
    if not processor_type or processor_type.strip() != processor_type:
        raise argparse.ArgumentTypeError("boundary must use PROCESSOR[=CYCLES]")
    if not separator:
        return processor_type, 1
    try:
        cycles = int(cycles_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "boundary cycles must be a non-negative integer"
        ) from error
    if cycles < 0:
        raise argparse.ArgumentTypeError(
            "boundary cycles must be a non-negative integer"
        )
    return processor_type, cycles


def _collect_profile_paths(
    explicit_paths: Iterable[Path], profiles_dir: Path | None
) -> list[Path]:
    paths = list(explicit_paths)
    if profiles_dir is not None:
        if not profiles_dir.is_dir():
            raise ProfileMergeError(
                f"profile directory '{profiles_dir}' does not exist"
            )
        discovered = sorted(profiles_dir.glob("profiles_*.yaml"))
        if not discovered:
            raise ProfileMergeError(
                f"profile directory '{profiles_dir}' contains no "
                "profiles_*.yaml files"
            )
        paths.extend(discovered)
    if not paths:
        raise ProfileMergeError(
            "no profile fragments specified; provide profile files or "
            "--profiles-dir"
        )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge CPU and FPGA profile fragments into a Mocasin application "
            "template."
        )
    )
    parser.add_argument("template", type=Path, help="Mocasin application template")
    parser.add_argument(
        "profiles", type=Path, nargs="*", help="profile fragment YAML files"
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        metavar="DIRECTORY",
        help="merge profiles_*.yaml files from DIRECTORY",
    )
    parser.add_argument(
        "--boundary",
        type=_parse_boundary,
        metavar="PROCESSOR[=CYCLES]",
        help=(
            "assign the synthetic boundary profile to PROCESSOR; CYCLES "
            "defaults to 1"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        template = _load_yaml(args.template)
        profile_paths = _collect_profile_paths(args.profiles, args.profiles_dir)
        fragments = [(str(path), _load_yaml(path)) for path in profile_paths]
        result = merge_profiles(template, fragments, boundary=args.boundary)
        with args.output.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                result,
                stream,
                sort_keys=False,
                default_flow_style=None,
                explicit_start=True,
                explicit_end=True,
            )
        unresolved = _unresolved_profiles(result)
        if unresolved:
            print(
                "warning: unresolved processor profiles remain: "
                + ", ".join(unresolved),
                file=sys.stderr,
            )
    except (OSError, ProfileMergeError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
