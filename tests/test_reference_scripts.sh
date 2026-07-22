#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

source_repo="${tmp_dir}/source-repo"
checkout_root="${tmp_dir}/checkouts"
manifest="${tmp_dir}/repositories.json"
bad_manifest="${tmp_dir}/repositories-bad.json"

git init --quiet --initial-branch=main "${source_repo}"
git -C "${source_repo}" config user.name "Reference Test"
git -C "${source_repo}" config user.email "reference-test@example.invalid"
git -C "${source_repo}" commit --quiet --allow-empty -m "fixture commit"
expected_commit="$(git -C "${source_repo}" rev-parse HEAD)"

jq -n \
  --arg url "${source_repo}" \
  --arg commit "${expected_commit}" \
  '{
    schema_version: 1,
    updated_at: "2026-07-22",
    repositories: [
      {
        id: "fixture",
        category: "test",
        url: $url,
        branch: "main",
        commit: $commit,
        license: "TEST-ONLY",
        competition_use: "test-only",
        sparse_paths: []
      }
    ]
  }' > "${manifest}"

bash "${workspace_dir}/scripts/fetch_references.sh" \
  --manifest "${manifest}" \
  --dest "${checkout_root}" >"${tmp_dir}/fetch.out" 2>&1

if rg -q 'warning:' "${tmp_dir}/fetch.out"; then
  cat "${tmp_dir}/fetch.out" >&2
  printf 'expected local reference fetch without Git warnings\n' >&2
  exit 1
fi

actual_commit="$(git -C "${checkout_root}/fixture" rev-parse HEAD)"
test "${actual_commit}" = "${expected_commit}"

bash "${workspace_dir}/scripts/check_references.sh" \
  --manifest "${manifest}" \
  --dest "${checkout_root}"

touch "${checkout_root}/fixture/untracked-file"
if bash "${workspace_dir}/scripts/fetch_references.sh" \
  --manifest "${manifest}" \
  --dest "${checkout_root}" >"${tmp_dir}/dirty.out" 2>&1; then
  printf 'expected dirty checkout fetch to fail\n' >&2
  exit 1
fi
rg -q 'dirty' "${tmp_dir}/dirty.out"
rm "${checkout_root}/fixture/untracked-file"

jq '.repositories[0].commit = "0000000000000000000000000000000000000000"' \
  "${manifest}" > "${bad_manifest}"
if bash "${workspace_dir}/scripts/check_references.sh" \
  --manifest "${bad_manifest}" \
  --dest "${checkout_root}" >"${tmp_dir}/mismatch.out" 2>&1; then
  printf 'expected commit mismatch check to fail\n' >&2
  exit 1
fi
rg -q 'commit mismatch' "${tmp_dir}/mismatch.out"

printf 'reference script tests passed\n'
