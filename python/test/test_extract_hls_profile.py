# Tests for extracting Mocasin profiles from Vitis HLS reports.
#
# @file
# @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

import tempfile
import unittest
from pathlib import Path

import yaml

from laksa_tools.mocasin.extract_hls_profile import extract_hls_profile
from laksa_tools.mocasin.extract_hls_profile import HLSProfileError
from laksa_tools.mocasin.extract_hls_profile import main


def report(model_name: str, worst_case_latency: str) -> str:
    return f"""\
<profile>
  <UserAssignments>
    <TopModelName>{model_name}</TopModelName>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Worst-caseLatency>{worst_case_latency}</Worst-caseLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
</profile>
"""


class ExtractHLSProfileTest(unittest.TestCase):
    def test_extracts_reports_in_process_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "hls_project" / "solution" / "syn" / "report"
            reports.mkdir(parents=True)
            (reports / "main_node_10_csynth.xml").write_text(
                report("main_node_10", "24361"), encoding="utf-8"
            )
            (reports / "main_node_2_csynth.xml").write_text(
                report("main_node_2", "1026"), encoding="utf-8"
            )
            (reports / "main_node_2_Pipeline_loop_csynth.xml").write_text(
                report("main_node_2_Pipeline_loop", "12"), encoding="utf-8"
            )

            result = extract_hls_profile(root)

            self.assertEqual(result["metadata"], {"source": "hls"})
            self.assertEqual(
                result["execution"]["processes"]["profiles"],
                {
                    "main_node_2": {"K26_PL": {"cycles": 1026}},
                    "main_node_10": {"K26_PL": {"cycles": 24361}},
                },
            )

    def test_command_writes_profile_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main_node_0_csynth.xml").write_text(
                report("main_node_0", "1034"), encoding="utf-8"
            )
            output = root / "profiles_K26_PL_hls.yaml"

            self.assertEqual(main([str(root), "-o", str(output)]), 0)
            result = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(
                result["execution"]["processes"]["profiles"]["main_node_0"],
                {"K26_PL": {"cycles": 1034}},
            )

    def test_rejects_missing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                HLSProfileError, "no main_node_N_csynth.xml reports"
            ):
                extract_hls_profile(Path(directory))


if __name__ == "__main__":
    unittest.main()
