#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tasks_file="${workspace_dir}/config/tasks.json"
vendor_dir="${workspace_dir}/vendor/JCIIOT2026"
expected_commit="f4ab8fd2158b919a41b2ce350432259cd1ee6a11"

jq empty "${tasks_file}"
test "$(jq '.tasks | length' "${tasks_file}")" -eq 5
test "$(jq '[.tasks[].max_score] | add' "${tasks_file}")" -eq 100

for required_file in \
  "README.md" \
  "docs/00-competition-brief.md" \
  "docs/01-official-baseline-audit.md" \
  "docs/02-technology-landscape.md" \
  "docs/03-winning-strategy.md" \
  "docs/04-source-index.md" \
  "docs/05-open-questions.md" \
  "docs/06-submission-compliance.md" \
  "research/rules-snapshot-2026-07-22.md"; do
  test -s "${workspace_dir}/${required_file}"
done

if test -d "${vendor_dir}/.git"; then
  actual_commit="$(git -C "${vendor_dir}" rev-parse HEAD)"
  if test "${actual_commit}" != "${expected_commit}"; then
    printf 'warning: vendor commit is %s, expected %s\n' "${actual_commit}" "${expected_commit}" >&2
    exit 1
  fi
else
  printf 'warning: official baseline snapshot is not present\n' >&2
  exit 1
fi

printf 'workspace checks passed\n'
