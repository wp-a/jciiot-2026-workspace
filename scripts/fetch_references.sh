#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${workspace_dir}/references/repositories.json"
destination="${workspace_dir}/references/repos"
only_id=""

usage() {
  printf 'Usage: %s [--manifest FILE] [--dest DIR] [--only ID]\n' "$0"
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
  all(.repositories[];
    (.id | type == "string" and length > 0) and
    (.url | type == "string" and length > 0) and
    (.branch | type == "string" and length > 0) and
    (.commit | test("^[0-9a-f]{40}$")) and
    (.sparse_paths | type == "array")
  )
' "${manifest}" >/dev/null

mkdir -p "${destination}"
export GIT_LFS_SKIP_SMUDGE=1

selected=0
while IFS=$'\t' read -r repo_id repo_url repo_branch repo_commit sparse_paths; do
  if test -n "${only_id}" && test "${repo_id}" != "${only_id}"; then
    continue
  fi
  selected=$((selected + 1))
  repo_dir="${destination}/${repo_id}"
  use_partial_clone=false
  case "${repo_url}" in
    https://*|http://*|ssh://*|git@*) use_partial_clone=true ;;
  esac

  if test -e "${repo_dir}" && ! test -d "${repo_dir}/.git"; then
    printf '%s exists but is not a Git checkout\n' "${repo_dir}" >&2
    exit 1
  fi

  if test -d "${repo_dir}/.git"; then
    if test -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=all)"; then
      printf 'reference checkout is dirty: %s\n' "${repo_id}" >&2
      exit 1
    fi
    actual_origin="$(git -C "${repo_dir}" remote get-url origin)"
    if test "${actual_origin}" != "${repo_url}"; then
      printf 'origin mismatch for %s: %s != %s\n' \
        "${repo_id}" "${actual_origin}" "${repo_url}" >&2
      exit 1
    fi
  else
    if test "${use_partial_clone}" = true; then
      git clone \
        --filter=blob:none \
        --no-checkout \
        --depth 1 \
        --branch "${repo_branch}" \
        "${repo_url}" "${repo_dir}"
    else
      git clone --no-checkout --branch "${repo_branch}" \
        "${repo_url}" "${repo_dir}"
    fi
    git -C "${repo_dir}" config advice.detachedHead false
  fi

  if test -n "${sparse_paths}"; then
    git -C "${repo_dir}" sparse-checkout init --cone
    IFS='|' read -r -a sparse_array <<< "${sparse_paths}"
    git -C "${repo_dir}" sparse-checkout set "${sparse_array[@]}"
  else
    git -C "${repo_dir}" sparse-checkout disable 2>/dev/null || true
  fi

  if test "${use_partial_clone}" = true; then
    git -C "${repo_dir}" fetch --depth 1 origin "${repo_commit}"
  else
    git -C "${repo_dir}" fetch origin "${repo_commit}"
  fi
  git -C "${repo_dir}" checkout --detach --force "${repo_commit}"

  actual_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  if test "${actual_commit}" != "${repo_commit}"; then
    printf 'commit mismatch after fetch for %s: %s != %s\n' \
      "${repo_id}" "${actual_commit}" "${repo_commit}" >&2
    exit 1
  fi
  printf 'pinned %-20s %s\n' "${repo_id}" "${actual_commit}"
done < <(
  jq -r '.repositories[] |
    [.id, .url, .branch, .commit, (.sparse_paths | join("|"))] | @tsv' \
    "${manifest}"
)

if test "${selected}" -eq 0; then
  printf 'repository id not found: %s\n' "${only_id}" >&2
  exit 1
fi
