import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "skills"
    / "library.py"
)


def skill_class(name):
    class StubSkill:
        def __init__(self, **kwargs):
            self.name = name
            self.kwargs = kwargs

    return StubSkill


def load_module():
    modules = {
        "robot_agent": types.ModuleType("robot_agent"),
        "robot_agent.core": types.ModuleType("robot_agent.core"),
        "robot_agent.core.memory": types.ModuleType("robot_agent.core.memory"),
        "robot_agent.core.scene_context": types.ModuleType(
            "robot_agent.core.scene_context"
        ),
        "robot_agent.environments": types.ModuleType(
            "robot_agent.environments"
        ),
        "robot_agent.environments.base": types.ModuleType(
            "robot_agent.environments.base"
        ),
        "robot_agent.skills": types.ModuleType("robot_agent.skills"),
        "robot_agent.skills.base": types.ModuleType("robot_agent.skills.base"),
    }
    modules["robot_agent.core.memory"].InMemoryStore = object
    modules["robot_agent.core.scene_context"].SceneContext = object
    modules["robot_agent.environments.base"].EnvBackend = object
    modules["robot_agent.skills.base"].BaseSkill = object

    class_names = {
        "move": "MoveSkill",
        "pick_up": "PickUpSkill",
        "place_down": "PlaceDownSkill",
        "record_trajectory": "RecordTrajectorySkill",
        "analyze_supply": "AnalyzeSupplySkill",
        "knowledge_mgr": "KnowledgeMgrSkill",
        "memory_mgr": "MemoryMgrSkill",
        "read_document": "ReadDocumentSkill",
        "competition_task": "CompetitionTaskSkill",
    }
    registered_names = {
        "move": "move",
        "pick_up": "pick_up",
        "place_down": "place_down",
        "record_trajectory": "record_trajectory",
        "analyze_supply": "analyze_supply",
        "knowledge_mgr": "knowledge_mgr",
        "memory_mgr": "memory_mgr",
        "read_document": "read_document",
        "competition_task": "competition_task",
    }
    for module_name, class_name in class_names.items():
        full_name = f"robot_agent.skills.{module_name}"
        stub_module = types.ModuleType(full_name)
        setattr(
            stub_module,
            class_name,
            skill_class(registered_names[module_name]),
        )
        modules[full_name] = stub_module

    spec = importlib.util.spec_from_file_location(
        "competition_library",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    environment = {
        "GATE_PLANNER": "true",
        "VLM_BASE_URL": "",
        "VLM_API_KEY": "",
        "VLM_MODEL": "",
        "OPENAI_API_KEY": "",
    }
    with (
        patch.dict(sys.modules, modules),
        patch.dict(os.environ, environment, clear=False),
    ):
        spec.loader.exec_module(module)
        planner_gate = os.environ.get("GATE_PLANNER")
    return module, planner_gate


class CompetitionEntrypointTests(unittest.TestCase):
    def test_library_forces_planner_gate_off(self):
        _module, planner_gate = load_module()

        self.assertEqual(planner_gate, "false")

    def test_competition_skill_is_registered_first(self):
        module, _planner_gate = load_module()
        backend = object()
        scene = object()
        grid = np.zeros((2, 2))

        skills = module.wired_skills(
            backend,
            scene_context=scene,
            grid=grid,
        )

        self.assertEqual(skills[0].name, "competition_task")
        self.assertIs(skills[0].kwargs["backend"], backend)
        self.assertIs(skills[0].kwargs["scene_context"], scene)
        self.assertIs(skills[0].kwargs["grid"], grid)

    def test_official_and_optional_memory_skills_remain_registered(self):
        module, _planner_gate = load_module()
        memory = object()

        skills = module.wired_skills(
            object(),
            scene_context=object(),
            grid=np.zeros((2, 2)),
            memory_store=memory,
        )

        self.assertEqual(
            [skill.name for skill in skills],
            [
                "competition_task",
                "move",
                "pick_up",
                "place_down",
                "analyze_supply",
                "record_trajectory",
                "knowledge_mgr",
                "read_document",
                "memory_mgr",
            ],
        )
        self.assertIs(skills[-1].kwargs["store"], memory)


if __name__ == "__main__":
    unittest.main()
