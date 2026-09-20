# Extract Mocasin execution profiles from Vitis HLS synthesis reports.
#
# @file
# @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

import argparse
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


class HLSProfileError(ValueError):
    """Vitis HLS reports cannot be converted into a profile fragment."""


_REPORT_NAME = re.compile(r"(main_node_(\d+))_csynth\.xml")


def _read_latency(path: Path, expected_name: str) -> int:
    try:
        report = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise HLSProfileError(f"cannot parse '{path}': {error}") from error

    model_name = report.findtext("./UserAssignments/TopModelName")
    if model_name != expected_name:
        raise HLSProfileError(
            f"'{path}' reports model '{model_name}', expected '{expected_name}'"
        )

    latency = report.findtext(
        "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"
    )
    try:
        cycles = int(latency) if latency is not None else -1
    except ValueError as error:
        raise HLSProfileError(
            f"'{path}' has invalid worst-case latency '{latency}'"
        ) from error
    if cycles < 0:
        raise HLSProfileError(
            f"'{path}' has invalid worst-case latency '{latency}'"
        )
    return cycles


def extract_hls_profile(
    report_root: Path, processor_type: str = "K26_PL"
) -> dict:
    """Return a Mocasin profile fragment from reports below ``report_root``."""

    reports = []
    for path in report_root.rglob("main_node_*_csynth.xml"):
        match = _REPORT_NAME.fullmatch(path.name)
        if match:
            reports.append((int(match.group(2)), match.group(1), path))
    reports.sort()
    if not reports:
        raise HLSProfileError(
            f"no main_node_N_csynth.xml reports found below '{report_root}'"
        )

    profiles = {}
    for _, process_name, path in reports:
        if process_name in profiles:
            raise HLSProfileError(
                f"multiple synthesis reports found for '{process_name}'"
            )
        profiles[process_name] = {
            processor_type: {"cycles": _read_latency(path, process_name)}
        }

    return {
        "metadata": {"source": "hls"},
        "execution": {"processes": {"profiles": profiles}},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract worst-case process latencies from Vitis HLS synthesis "
            "reports into a Mocasin profile fragment."
        )
    )
    parser.add_argument(
        "reports", type=Path, help="directory containing the Vitis HLS output"
    )
    parser.add_argument(
        "--processor-type", default="K26_PL", help="Mocasin processor type"
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = extract_hls_profile(args.reports, args.processor_type)
        with args.output.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                profile,
                stream,
                sort_keys=False,
                default_flow_style=None,
            )
    except (HLSProfileError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
