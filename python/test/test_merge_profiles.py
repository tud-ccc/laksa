# Tests for the Mocasin profile merger.
#
# @file
# @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

import tempfile
import unittest
from pathlib import Path

import yaml

from laksa_tools.mocasin.merge_profiles import ProfileMergeError
from laksa_tools.mocasin.merge_profiles import main
from laksa_tools.mocasin.merge_profiles import merge_profiles


TEMPLATE = yaml.safe_load(
    """
name: main
execution:
  processes:
    profiles:
      node:
        UNKNOWN: {cycles: 0}
      boundary:
        UNKNOWN: {cycles: 0}
    instances:
      node:
        model: static
        profile: node
"""
)


def fragment(processor_type, cycles):
    return {
        "execution": {
            "processes": {
                "profiles": {"node": {processor_type: {"cycles": cycles}}}
            }
        }
    }


class MergeProfilesTest(unittest.TestCase):
    def test_merges_profiles_and_keeps_unresolved_boundary(self):
        result = merge_profiles(
            TEMPLATE,
            [
                ("cpu.yaml", fragment("CortexA53", 120)),
                ("fpga.yaml", fragment("K26_PL", 30)),
            ],
        )

        self.assertEqual(
            result["execution"]["processes"]["profiles"],
            {
                "node": {
                    "CortexA53": {"cycles": 120},
                    "K26_PL": {"cycles": 30},
                },
                "boundary": {"UNKNOWN": {"cycles": 0}},
            },
        )
        self.assertIn(
            "UNKNOWN",
            TEMPLATE["execution"]["processes"]["profiles"]["node"],
        )

    def test_rejects_duplicate_processor_profile(self):
        with self.assertRaisesRegex(
            ProfileMergeError, "duplicates profile 'node.CortexA53'"
        ):
            merge_profiles(
                TEMPLATE,
                [
                    ("first.yaml", fragment("CortexA53", 120)),
                    ("second.yaml", fragment("CortexA53", 130)),
                ],
            )

    def test_rejects_unknown_profile(self):
        unknown = fragment("CortexA53", 120)
        unknown["execution"]["processes"]["profiles"] = {
            "other": {"CortexA53": {"cycles": 120}}
        }
        with self.assertRaisesRegex(ProfileMergeError, "unknown profile 'other'"):
            merge_profiles(TEMPLATE, [("unknown.yaml", unknown)])

    def test_rejects_invalid_cycles(self):
        with self.assertRaisesRegex(
            ProfileMergeError, "non-negative integer 'cycles'"
        ):
            merge_profiles(
                TEMPLATE,
                [("invalid.yaml", fragment("CortexA53", -1))],
            )

    def test_command_writes_merged_application(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path = root / "app.template.yaml"
            profile_path = root / "profiles_CortexA53.yaml"
            output_path = root / "app.yaml"
            template_path.write_text(yaml.safe_dump(TEMPLATE), encoding="utf-8")
            profile_path.write_text(
                yaml.safe_dump(fragment("CortexA53", 120)), encoding="utf-8"
            )

            self.assertEqual(
                main(
                    [
                        str(template_path),
                        str(profile_path),
                        "-o",
                        str(output_path),
                    ]
                ),
                0,
            )
            result = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result["execution"]["processes"]["profiles"]["node"],
                {"CortexA53": {"cycles": 120}},
            )


if __name__ == "__main__":
    unittest.main()
