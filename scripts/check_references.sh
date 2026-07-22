#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${workspace_dir}/references/repositories.json"
destination="${workspace_dir}/references/repos"
only_id=""
allow_missing=false

usage() {
  printf 'Usage: %s [--manifest FILE] [--dest DIR] [--only ID] [--allow-missing]\n' "$0"
}

while test "$#" -gt 0; do
  case "$1" in
    --manifest)
      manifest="$2"
      shift 2
      ;;
    --dest)
      destination="$2"
      shift 2
      ;;
    --only)
      only_id="$2"
      shift 2
      ;;
    --allow-missing)
      allow_missing=true
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
test -f "${manifest}"

jq -e '
  .schema_version == 1 and
  (.repositories | type == "array" and length > 0) and
  ([.repositories[].id] | length == (unique | length)) and
  all(.repositories[]; .commit | test("^[0-9a-f]{40}$"))
' "${manifest}" >/dev/null

selected=0
checked=0
missing=0
while IFS=$'\t' read -r repo_id repo_url repo_commit; do
  if test -n "${only_id}" && test "${repo_id}" != "${only_id}"; then
    continue
  fi
  selected=$((selected + 1))
  repo_dir="${destination}/${repo_id}"

  if ! test -d "${repo_dir}/.git"; then
    if test "${allow_missing}" = true; then
      printf 'missing %-19s %s\n' "${repo_id}" "${repo_dir}"
      missing=$((missing + 1))
      continue
    fi
    printf 'missing reference checkout: %s\n' "${repo_id}" >&2
    exit 1
  fi

  actual_origin="$(git -C "${repo_dir}" remote get-url origin)"
  if test "${actual_origin}" != "${repo_url}"; then
    printf 'origin mismatch for %s: %s != %s\n' \
      "${repo_id}" "${actual_origin}" "${repo_url}" >&2
    exit 1
  fi

  actual_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  if test "${actual_commit}" != "${repo_commit}"; then
    printf 'commit mismatch for %s: %s != %s\n' \
      "${repo_id}" "${actual_commit}" "${repo_commit}" >&2
    exit 1
  fi

  if test -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=all)"; then
    printf 'reference checkout is dirty: %s\n' "${repo_id}" >&2
    exit 1
  fi

  checked=$((checked + 1))
  printf 'ok      %-19s %s\n' "${repo_id}" "${actual_commit}"
done < <(
  jq -r '.repositories[] | [.id, .url, .commit] | @tsv' "${manifest}"
)

if test "${selected}" -eq 0; then
  printf 'repository id not found: %s\n' "${only_id}" >&2
  exit 1
fi

printf 'reference checks passed: %d checked, %d missing\n' "${checked}" "${missing}"
