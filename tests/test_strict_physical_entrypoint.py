import ast
from pathlib import Path


FLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "submission"
    / "JCIIOT"
    / "src"
    / "robot_agent"
    / "workflows"
    / "competition_flow.py"
)


def _function_source(name: str) -> str:
    tree = ast.parse(FLOW_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(
                FLOW_PATH.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError(f"missing function {name}")


def test_official_entrypoint_has_no_floor_push_or_attachment_calls():
    source = _function_source("run_official_task")
    forbidden = (
        "l1_floor_push",
        "run_physical_floor_route",
        "capture_transport_attachment",
        "transport_attachment",
    )
    assert not any(token in source for token in forbidden), source


def test_official_entrypoint_selects_physical_carry_literal():
    source = _function_source("run_official_task")
    assert 'transport_mode="physical_carry"' in source
