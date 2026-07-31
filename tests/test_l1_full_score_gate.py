import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_l1_full_score_gate import evaluate_l1_gate


def _manifest(*, tier="nominal", full_score=True):
    manifest = {
        "status": "complete",
        "task_index": 0,
        "level": "L1",
        "official_score": 10 if full_score else 5,
        "max_score": 10,
        "successful_grasp_events": 1,
        "required_grasp_events": 1,
        "collision_frames": 0,
        "final_target_distance_m": 0.2 if full_score else 2.0,
        "execution_result": {
            "success": bool(full_score),
            "failures": [] if full_score else [{"failure_stage": "verify"}],
        },
        "perturbation": {"tier": tier},
        "perturbation_application": {
            "valid": True,
            "nominal_noop": tier == "nominal",
        },
    }
    return manifest


def _passing_inputs():
    nominal = [_manifest() for _ in range(5)]
    perturbed = [_manifest(tier="small") for _ in range(18)]
    perturbed.extend(_manifest(tier="medium", full_score=False) for _ in range(2))
    return nominal, perturbed


class L1FullScoreGateTests(unittest.TestCase):
    def test_cli_loads_batch_directories_and_writes_passing_report(self):
        nominal, perturbed = _passing_inputs()
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nominal_dir = root / "nominal" / "manifests"
            perturbation_dir = root / "perturbed" / "manifests"
            nominal_dir.mkdir(parents=True)
            perturbation_dir.mkdir(parents=True)
            for index, manifest in enumerate(nominal):
                (nominal_dir / f"manifest-nominal-{index}.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
            for index, manifest in enumerate(perturbed):
                (perturbation_dir / f"manifest-perturbed-{index}.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
            output = root / "gate.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_l1_full_score_gate.py",
                    "--nominal-dir",
                    str(root / "nominal"),
                    "--perturbation-dir",
                    str(root / "perturbed"),
                    "--output",
                    str(output),
                ],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["gate_passed"])
            self.assertEqual(len(report["nominal_manifest_paths"]), 5)
            self.assertEqual(len(report["perturbation_manifest_paths"]), 20)

    def test_gate_accepts_five_nominal_and_eighteen_of_twenty_perturbed(self):
        nominal, perturbed = _passing_inputs()

        report = evaluate_l1_gate(nominal, perturbed)

        self.assertEqual(report["nominal_runs"], 5)
        self.assertEqual(report["nominal_full_score_runs"], 5)
        self.assertEqual(report["perturbation_runs"], 20)
        self.assertEqual(report["perturbation_full_score_runs"], 18)
        self.assertEqual(report["collision_runs"], 0)
        self.assertEqual(report["failure_stages"], {"verify": 2})
        self.assertTrue(report["gate_passed"])

    def test_gate_rejects_each_missing_requirement(self):
        mutations = {}

        nominal, perturbed = _passing_inputs()
        mutations["four nominal runs"] = (nominal[:4], perturbed)

        nominal, perturbed = _passing_inputs()
        perturbed[17] = _manifest(tier="small", full_score=False)
        mutations["only seventeen perturbation successes"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        perturbed[0]["collision_frames"] = 1
        mutations["one collision"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        perturbed[0]["perturbation_application"]["valid"] = False
        mutations["invalid perturbation audit"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        nominal[0]["successful_grasp_events"] = 0
        mutations["missing physical grasp evidence"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        nominal[0]["final_target_distance_m"] = 0.8
        mutations["target boundary is not strictly inside"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        mutations["nineteen perturbation runs"] = (nominal, perturbed[:19])

        nominal, perturbed = _passing_inputs()
        nominal[0]["collision_frames"] = None
        mutations["missing collision evidence"] = (nominal, perturbed)

        nominal, perturbed = _passing_inputs()
        nominal[0]["task_index"] = 1
        mutations["wrong task manifest"] = (nominal, perturbed)

        for label, (nominal, perturbed) in mutations.items():
            with self.subTest(label=label):
                report = evaluate_l1_gate(
                    copy.deepcopy(nominal),
                    copy.deepcopy(perturbed),
                )
                self.assertFalse(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
