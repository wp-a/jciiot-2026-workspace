#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
fixture_dir="${tmp_dir}/duplicate-source-workspace"

mkdir -p "${fixture_dir}"
git -C "${workspace_dir}" archive HEAD | tar -x -C "${fixture_dir}"
tail -n 1 "${fixture_dir}/research/source-ledger.csv" >> \
  "${fixture_dir}/research/source-ledger.csv"

if bash "${workspace_dir}/scripts/check_workspace.sh" \
  --workspace "${fixture_dir}" >"${tmp_dir}/duplicate.out" 2>&1; then
  printf 'expected duplicate source id check to fail\n' >&2
  exit 1
fi
rg -q 'duplicate source id' "${tmp_dir}/duplicate.out"

stale_dir="${tmp_dir}/stale-task-workspace"
mkdir -p "${stale_dir}/vendor"
git -C "${workspace_dir}" archive HEAD | tar -x -C "${stale_dir}"
ln -s "${workspace_dir}/vendor/JCIIOT2026" "${stale_dir}/vendor/JCIIOT2026"
python3 - "${stale_dir}/config/tasks.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
task = next(item for item in data["tasks"] if item["level"] == "L3")
task["source"] = "input_6"
task["source_center_xy"] = [11.937, 3.932]
path.write_text(json.dumps(data), encoding="utf-8")
PY

if bash "${workspace_dir}/scripts/check_workspace.sh" \
  --workspace "${stale_dir}" >"${tmp_dir}/stale-task.out" 2>&1; then
  printf 'expected stale official task check to fail\n' >&2
  exit 1
fi
rg -q 'L3 source' "${tmp_dir}/stale-task.out"

bash "${workspace_dir}/scripts/check_workspace.sh" \
  --workspace "${workspace_dir}"

printf 'workspace check tests passed\n'
