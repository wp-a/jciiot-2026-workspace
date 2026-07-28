#!/usr/bin/env python3
"""AST audit for simulator-state shortcuts in the submitted scored path."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_PRIVATE_BACKEND_MEMBERS = {
    "_mark_trajectory_event",
    "_record_trajectory_frame",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    rule: str
    excerpt: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_backend_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "backend" or node.id.endswith("_backend")
    if isinstance(node, ast.Attribute):
        if node.attr == "backend" or node.attr.endswith("_backend"):
            return True
        return _is_backend_reference(node.value)
    if isinstance(node, ast.Subscript):
        return _is_backend_reference(node.value)
    return False


def _subscript_key(node: ast.Subscript):
    value = node.slice
    if isinstance(value, ast.Constant):
        return value.value
    return None


def _target_has_terminal_attribute(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Subscript):
        return _target_has_terminal_attribute(node.value, name)
    return isinstance(node, ast.Attribute) and node.attr == name


def _iter_assignment_targets(node: ast.AST) -> Iterable[ast.AST]:
    if isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _iter_assignment_targets(item)
        return
    yield node


class ScoredPathVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path, source: str) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.violations: list[Violation] = []

    def _add(self, node: ast.AST, rule: str) -> None:
        line = int(getattr(node, "lineno", 0))
        excerpt = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.violations.append(
            Violation(
                path=self.path.as_posix(),
                line=line,
                column=int(getattr(node, "col_offset", 0)) + 1,
                rule=rule,
                excerpt=excerpt,
            )
        )

    def _check_target(self, target: ast.AST) -> None:
        for item in _iter_assignment_targets(target):
            if _target_has_terminal_attribute(item, "qpos"):
                self._add(item, "direct_qpos_write")
            if (
                isinstance(item, ast.Subscript)
                and _subscript_key(item) == "relative_xy"
            ) or (
                isinstance(item, ast.Attribute)
                and item.attr == "relative_xy"
            ):
                self._add(item, "attachment_relative_write")
            if (
                isinstance(item, ast.Attribute)
                and item.attr.startswith("_")
                and item.attr not in ALLOWED_PRIVATE_BACKEND_MEMBERS
                and _is_backend_reference(item.value)
            ):
                self._add(item, "private_backend_member")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.visit(node.value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if (
                module.endswith("transport_attachment")
                and alias.name == "sync_transport_attachment"
            ):
                self._add(node, "transport_sync_helper")
            if (
                module == "robot_agent.environments.robosuite_backend"
                and alias.name.startswith("_")
            ):
                self._add(node, "private_backend_import")

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if (
            isinstance(function, ast.Name)
            and function.id == "sync_transport_attachment"
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "sync_transport_attachment"
        ):
            self._add(node, "transport_sync_helper")
        if (
            isinstance(function, ast.Attribute)
            and function.attr.startswith("_")
            and function.attr not in ALLOWED_PRIVATE_BACKEND_MEMBERS
            and _is_backend_reference(function.value)
        ):
            self._add(node, "private_backend_member")
        self.generic_visit(node)


def scan_file(path: Path) -> list[Violation]:
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = ScoredPathVisitor(path=path, source=source)
    visitor.visit(tree)
    return visitor.violations


def scan_submission(root: Path) -> list[Violation]:
    root = Path(root)
    agent_root = root / "JCIIOT" / "src" / "robot_agent"
    paths = []
    for directory_name in ("skills", "workflows"):
        directory = agent_root / directory_name
        if directory.is_dir():
            paths.extend(directory.rglob("*.py"))
    violations = []
    for path in sorted(paths):
        violations.extend(scan_file(path))
    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    violations = scan_submission(args.root)
    report = {
        "root": str(args.root),
        "violation_count": len(violations),
        "violations": [item.as_dict() for item in violations],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
