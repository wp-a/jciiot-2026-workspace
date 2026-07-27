#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_root=""
overlay_dir=""
output_dir=""

while (($#)); do
    case "$1" in
        --workspace)
            workspace_dir="$(cd "$2" && pwd)"
            shift 2
            ;;
        --official-root)
            official_root="$2"
            shift 2
            ;;
        --overlay)
            overlay_dir="$2"
            shift 2
            ;;
        --output)
            output_dir="$2"
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$official_root" || -z "$output_dir" ]]; then
    printf 'usage: %s --official-root PATH --output PATH [--workspace PATH] [--overlay PATH]\n' "$0" >&2
    exit 2
fi

official_root="$(cd "$official_root" && pwd)"
if [[ -z "$overlay_dir" ]]; then
    overlay_dir="$workspace_dir/submission"
fi

if [[ ! -d "$overlay_dir" ]]; then
    printf 'submission overlay not found: %s\n' "$overlay_dir" >&2
    exit 1
fi
if [[ -e "$output_dir" ]]; then
    printf 'output already exists: %s\n' "$output_dir" >&2
    exit 1
fi

expected_commit="$(python3 - "$workspace_dir/config/upstream-lock.json" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(lock["repository"]["commit"])
PY
)"
actual_commit="$(git -C "$official_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
    printf 'official commit mismatch: expected %s, got %s\n' "$expected_commit" "$actual_commit" >&2
    exit 1
fi

while IFS= read -r source_path; do
    relative_path="${source_path#"$overlay_dir"/}"
    case "$relative_path" in
        README.md|JCIIOT/knowledge/robot_params.json|JCIIOT/knowledge/generated_sop_l[1-5].md|JCIIOT/src/robot_agent/skills/*.py|JCIIOT/src/robot_agent/workflows/*.py)
            ;;
        *)
            printf 'forbidden overlay path: %s\n' "$relative_path" >&2
            exit 1
            ;;
    esac
done < <(find "$overlay_dir" -type f -print | sort)

cp -R "$official_root" "$output_dir"
while IFS= read -r source_path; do
    relative_path="${source_path#"$overlay_dir"/}"
    if [[ "$relative_path" == "README.md" ]]; then
        continue
    fi
    target_path="$output_dir/$relative_path"
    mkdir -p "$(dirname "$target_path")"
    cp "$source_path" "$target_path"
done < <(find "$overlay_dir" -type f -print | sort)

printf 'materialized submission at %s from official %s\n' "$output_dir" "$actual_commit"
