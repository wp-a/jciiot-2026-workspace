#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
require_private_remote=false

usage() {
  printf 'Usage: %s [--workspace DIR] [--require-private-remote]\n' "$0"
}

while test "$#" -gt 0; do
  case "$1" in
    --workspace)
      workspace_dir="$(cd "$2" && pwd)"
      shift 2
      ;;
    --require-private-remote)
      require_private_remote=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null
command -v jq >/dev/null
command -v python3 >/dev/null

required_files=(
  README.md
  STATUS.md
  CHANGELOG.md
  THIRD_PARTY_NOTICES.md
  config/tasks.json
  config/upstream-lock.json
  docs/00-competition-brief.md
  docs/01-official-baseline-audit.md
  docs/02-technology-landscape.md
  docs/03-winning-strategy.md
  docs/04-source-index.md
  docs/05-open-questions.md
  docs/06-submission-compliance.md
  docs/07-similar-projects.md
  docs/08-module-roadmap.md
  decisions/0001-hybrid-competition-architecture.md
  decisions/0002-reference-code-policy.md
  research/rules-snapshot-2026-07-22.md
  research/source-ledger.csv
  references/repositories.json
  experiments/experiment-log.csv
)

for required_file in "${required_files[@]}"; do
  if ! test -s "${workspace_dir}/${required_file}"; then
    printf 'missing or empty required file: %s\n' "${required_file}" >&2
    exit 1
  fi
done

tasks_file="${workspace_dir}/config/tasks.json"
jq -e '
  (.tasks | length) == 5 and
  ([.tasks[].max_score] | add) == 100 and
  ([.tasks[].level] | length == (unique | length)) and
  all(.tasks[];
    (.source_center_xy | length == 2) and
    (.target_center_xy | length == 2) and
    (.objects | length >= 1)
  )
' "${tasks_file}" >/dev/null

ledger_file="${workspace_dir}/research/source-ledger.csv"
python3 - "${ledger_file}" <<'PY'
import csv
import sys

ledger = sys.argv[1]
expected = [
    "source_id", "category", "title", "url", "version_or_commit",
    "accessed_at", "license", "evidence_level", "module",
    "adoption_status", "notes",
]
with open(ledger, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != expected:
        raise SystemExit("invalid source ledger header")
    seen = set()
    count = 0
    for row in reader:
        source_id = row["source_id"]
        if source_id in seen:
            raise SystemExit(f"duplicate source id: {source_id}")
        seen.add(source_id)
        count += 1
        if not row["url"].startswith("https://"):
            raise SystemExit(f"invalid source URL for {source_id}")
        if not row["accessed_at"]:
            raise SystemExit(f"missing access date for {source_id}")
    if count == 0:
        raise SystemExit("source ledger is empty")
print(f"source ledger valid: {count} sources")
PY

reference_manifest="${workspace_dir}/references/repositories.json"
jq -e '
  .schema_version == 1 and
  (.repositories | type == "array" and length > 0) and
  ([.repositories[].id] | length == (unique | length)) and
  all(.repositories[];
    (.commit | test("^[0-9a-f]{40}$")) and
    (.sparse_paths | type == "array")
  )
' "${reference_manifest}" >/dev/null

upstream_lock="${workspace_dir}/config/upstream-lock.json"
jq -e '
  .schema_version == 1 and
  (.repository.commit | test("^[0-9a-f]{40}$")) and
  (.tracked_files | length > 0) and
  all(.tracked_files[]; .sha256 | test("^[0-9a-f]{64}$"))
' "${upstream_lock}" >/dev/null

vendor_rel="$(jq -r '.repository.local_path' "${upstream_lock}")"
vendor_dir="${workspace_dir}/${vendor_rel}"
expected_commit="$(jq -r '.repository.commit' "${upstream_lock}")"
if ! test -d "${vendor_dir}/.git"; then
  printf 'official baseline checkout is missing: %s\n' "${vendor_dir}" >&2
  exit 1
fi
actual_commit="$(git -C "${vendor_dir}" rev-parse HEAD)"
if test "${actual_commit}" != "${expected_commit}"; then
  printf 'official baseline commit mismatch: %s != %s\n' \
    "${actual_commit}" "${expected_commit}" >&2
  exit 1
fi

sha256_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

while IFS=$'\t' read -r expected_hash file_rel; do
  actual_hash="$(sha256_file "${vendor_dir}/${file_rel}")"
  if test "${actual_hash}" != "${expected_hash}"; then
    printf 'upstream file hash mismatch: %s\n' "${file_rel}" >&2
    exit 1
  fi
done < <(jq -r '.tracked_files[] | [.sha256, .path] | @tsv' "${upstream_lock}")

bash "${workspace_dir}/scripts/check_references.sh" \
  --manifest "${reference_manifest}" \
  --dest "${workspace_dir}/references/repos" \
  --allow-missing

if git -C "${workspace_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for ignored_path in \
    .worktrees/example \
    references/repos \
    vendor/JCIIOT2026 \
    data/example.hdf5 \
    artifacts/example.mp4; do
    if ! git -C "${workspace_dir}" check-ignore -q "${ignored_path}"; then
      printf 'path should be ignored: %s\n' "${ignored_path}" >&2
      exit 1
    fi
  done

  tracked_forbidden="$(
    git -C "${workspace_dir}" ls-files | \
      grep -E '^(vendor/JCIIOT2026/|references/repos/|\.env$|\.env\.)|\.(pth|pt|ckpt|hdf5|h5|zip|tar|tar\.gz|mp4|mov|avi|onnx|safetensors|npy|npz|log)$' \
      || true
  )"
  if test -n "${tracked_forbidden}"; then
    printf 'forbidden tracked files:\n%s\n' "${tracked_forbidden}" >&2
    exit 1
  fi

  while IFS= read -r tracked_file; do
    test -n "${tracked_file}" || continue
    file_size="$(wc -c < "${workspace_dir}/${tracked_file}")"
    if test "${file_size}" -gt 10485760; then
      printf 'tracked file exceeds 10 MiB: %s\n' "${tracked_file}" >&2
      exit 1
    fi
  done < <(git -C "${workspace_dir}" ls-files)
fi

if test "${require_private_remote}" = true; then
  command -v gh >/dev/null
  origin_url="$(git -C "${workspace_dir}" remote get-url origin)"
  repo_name="${origin_url#https://github.com/}"
  repo_name="${repo_name#git@github.com:}"
  repo_name="${repo_name%.git}"
  visibility="$(gh repo view "${repo_name}" --json visibility --jq '.visibility')"
  if test "${visibility}" != "PRIVATE"; then
    printf 'GitHub repository is not private: %s (%s)\n' \
      "${repo_name}" "${visibility}" >&2
    exit 1
  fi
  printf 'private GitHub remote verified: %s\n' "${repo_name}"
fi

printf 'workspace checks passed at %s\n' "${workspace_dir}"
