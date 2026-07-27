import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "skills"
    / "competition_task.py"
)


class StubBaseSkill:
    def __init__(self, *, name, description, keywords):
        self.name = name
        self.description = description
        self.keywords = keywords


class StubSkillResult:
    def __init__(self, *, skill_name, success, message, payload=None):
        self.skill_name = skill_name
        self.success = success
        self.message = message
        self.payload = payload or {}


def load_module():
    robot_agent = types.ModuleType("robot_agent")
    core = types.ModuleType("robot_agent.core")
    core_types = types.ModuleType("robot_agent.core.types")
    core_types.ExecutionContext = SimpleNamespace
    core_types.SkillResult = StubSkillResult
    skills = types.ModuleType("robot_agent.skills")
    base = types.ModuleType("robot_agent.skills.base")
    base.BaseSkill = StubBaseSkill
    workflows = types.ModuleType("robot_agent.workflows")
    flow = types.ModuleType("robot_agent.workflows.competition_flow")
    flow.run_official_task = lambda **_kwargs: {"success": True}
    modules = {
        "robot_agent": robot_agent,
        "robot_agent.core": core,
        "robot_agent.core.types": core_types,
        "robot_agent.skills": skills,
        "robot_agent.skills.base": base,
        "robot_agent.workflows": workflows,
        "robot_agent.workflows.competition_flow": flow,
    }
    spec = importlib.util.spec_from_file_location("competition_task", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class CompetitionTaskTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def write_config(self, root: Path) -> Path:
        path = root / "task_config.json"
        path.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "level": "L1",
                            "source": "input_5",
                            "target": "output_4",
                            "object": ["box_near", "box_far"],
                            "max_score": 10,
                        },
                        {
                            "level": "L2",
                            "source": "input_6",
                            "target": "output_4",
                            "object": ["tote_upper", "tote_lower"],
                            "max_score": 15,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_task_index_accepts_top_level_and_scene_metadata(self):
        self.assertEqual(
            self.module.task_index_from_metadata({"task_index": 1}),
            1,
        )
        self.assertEqual(
            self.module.task_index_from_metadata(
                {"scene": {"task_index": "0"}}
            ),
            0,
        )

    def test_task_index_rejects_missing_bool_and_non_integer_values(self):
        for metadata in (
            {},
            {"task_index": True},
            {"task_index": "L1"},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    self.module.task_index_from_metadata(metadata)

    def test_load_official_task_returns_an_independent_task_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir))
            task = self.module.load_official_task(1, config)

            self.assertEqual(task["level"], "L2")
            task["level"] = "changed"
            self.assertEqual(
                self.module.load_official_task(1, config)["level"],
                "L2",
            )

    def test_load_official_task_rejects_invalid_config_and_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.write_config(root)
            for index in (-1, 2):
                with self.subTest(index=index):
                    with self.assertRaises(IndexError):
                        self.module.load_official_task(index, config)

            malformed = root / "malformed.json"
            malformed.write_text(
                json.dumps({"tasks": [{"level": "L1"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                self.module.load_official_task(0, malformed)

    def test_skill_handles_any_non_empty_scored_prompt(self):
        skill = self.module.CompetitionTaskSkill(
            backend=object(),
            scene_context=object(),
            grid=object(),
        )

        self.assertTrue(skill.can_handle("搬运物料"))
        self.assertTrue(skill.can_handle("execute official task"))
        self.assertFalse(skill.can_handle("   "))

    def test_skill_calls_official_workflow_and_preserves_result(self):
        captured = {}
        workflow = {
            "success": True,
            "states": {"box_near": "verified"},
            "history": [{"object_name": "box_near", "state": "verified"}],
        }

        def run_official_task(**kwargs):
            captured.update(kwargs)
            return workflow

        backend = object()
        scene_context = object()
        grid = object()
        skill = self.module.CompetitionTaskSkill(
            backend=backend,
            scene_context=scene_context,
            grid=grid,
        )
        task = {
            "level": "L1",
            "source": "input_5",
            "target": "output_4",
            "object": ["box_near"],
            "max_score": 10,
        }

        with (
            patch.object(self.module, "load_official_task", return_value=task),
            patch.object(
                self.module,
                "run_official_task",
                side_effect=run_official_task,
            ),
        ):
            result = skill.run(
                SimpleNamespace(
                    task="搬运物料",
                    metadata={"scene": {"task_index": 0}},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.payload["workflow"], workflow)
        self.assertEqual(result.payload["task_index"], 0)
        self.assertEqual(result.payload["level"], "L1")
        self.assertIs(captured["backend"], backend)
        self.assertIs(captured["scene_context"], scene_context)
        self.assertIs(captured["grid"], grid)
        self.assertEqual(captured["task"], task)
        self.assertEqual(captured["max_attempts"], 1)

    def test_skill_propagates_workflow_failure(self):
        skill = self.module.CompetitionTaskSkill(
            backend=object(),
            scene_context=object(),
            grid=object(),
        )
        task = {
            "level": "L2",
            "source": "input_6",
            "target": "output_4",
            "object": ["tote_lower"],
            "max_score": 15,
        }
        workflow = {
            "success": False,
            "states": {"tote_lower": "failed"},
            "failures": [{"failure_stage": "grasp"}],
        }
        with (
            patch.object(self.module, "load_official_task", return_value=task),
            patch.object(
                self.module,
                "run_official_task",
                return_value=workflow,
            ),
        ):
            result = skill.run(
                SimpleNamespace(task="task", metadata={"task_index": 1})
            )

        self.assertFalse(result.success)
        self.assertEqual(result.payload["error_code"], "workflow_failed")
        self.assertEqual(result.payload["workflow"], workflow)

    def test_skill_reports_context_and_workflow_exceptions_without_fallback(self):
        skill = self.module.CompetitionTaskSkill(
            backend=object(),
            scene_context=object(),
            grid=object(),
        )
        invalid = skill.run(SimpleNamespace(task="task", metadata={}))
        self.assertFalse(invalid.success)
        self.assertEqual(
            invalid.payload["error_code"],
            "task_resolution_failed",
        )

        task = {
            "level": "L1",
            "source": "input_5",
            "target": "output_4",
            "object": ["box_near"],
            "max_score": 10,
        }
        with (
            patch.object(self.module, "load_official_task", return_value=task),
            patch.object(
                self.module,
                "run_official_task",
                side_effect=RuntimeError("contact solver failed"),
            ),
        ):
            crashed = skill.run(
                SimpleNamespace(task="task", metadata={"task_index": 0})
            )

        self.assertFalse(crashed.success)
        self.assertEqual(
            crashed.payload["error_code"],
            "workflow_exception",
        )
        self.assertEqual(crashed.payload["error_type"], "RuntimeError")
        self.assertIn("contact solver failed", crashed.message)


if __name__ == "__main__":
    unittest.main()
