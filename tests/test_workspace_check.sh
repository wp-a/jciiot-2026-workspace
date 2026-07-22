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

bash "${workspace_dir}/scripts/check_workspace.sh" \
  --workspace "${workspace_dir}"

printf 'workspace check tests passed\n'
