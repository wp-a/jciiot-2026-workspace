# JCIIOT 2026 Traceable Workspace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a private, version-controlled JCIIOT competition workspace whose research, external code references, decisions, experiments, and compliance status are reproducible and auditable.

**Architecture:** Keep first-party documentation, manifests, scripts, and competition code in the top-level Git repository. Keep the official baseline and external reference checkouts outside the top-level history, while pinning every checkout by URL and commit in machine-readable manifests.

**Tech Stack:** Git, GitHub CLI, Bash, JSON, CSV, Markdown, jq

---

### Task 1: Add Repository Governance Files

**Files:**
- Modify: `.gitignore`
- Create: `STATUS.md`
- Create: `CHANGELOG.md`
- Create: `THIRD_PARTY_NOTICES.md`

**Step 1:** Extend `.gitignore` for `references/repos/`, secrets, large model/data formats, experiment outputs, and temporary files.

**Step 2:** Create `STATUS.md` with the current baseline, current phase, known blockers, immediate milestones, and update rules.

**Step 3:** Create `CHANGELOG.md` and `THIRD_PARTY_NOTICES.md` with explicit provenance and license status.

**Step 4:** Verify ignored paths with `git check-ignore` and verify governance files are non-empty.

**Step 5:** Commit the governance files.

### Task 2: Build the Research Traceability Layer

**Files:**
- Create: `research/source-ledger.csv`
- Create: `docs/07-similar-projects.md`
- Create: `docs/08-module-roadmap.md`
- Create: `research/notes/README.md`
- Create: `research/notes/github-project-audit-2026-07-22.md`

**Step 1:** Add one source-ledger row per official source, same-competition fork, directly useful repository, architecture reference, and license-risk reference.

**Step 2:** Write the repository survey with applicability, concrete reusable ideas, integration cost, license status, and recommendation.

**Step 3:** Map projects and techniques to SOP parsing, orchestration, navigation, grasping, placement, recovery, data, evaluation, and innovation.

**Step 4:** Record the GitHub audit method, checked date, limitations, and the distinction between author claims and locally reproduced evidence.

**Step 5:** Validate the CSV header, unique source IDs, URLs, and required Markdown links; then commit.

### Task 3: Add Reproducible Reference Checkout Management

**Files:**
- Create: `references/README.md`
- Create: `references/repositories.json`
- Create: `scripts/fetch_references.sh`
- Create: `scripts/check_references.sh`

**Step 1:** Define each selected repository with category, URL, branch, expected commit, checkout mode, license, and competition-use status.

**Step 2:** Implement idempotent shallow clone/fetch behavior that skips Git LFS and refuses to overwrite dirty reference checkouts.

**Step 3:** Implement commit verification against `references/repositories.json`.

**Step 4:** Run `bash -n` and jq validation, then test the scripts on one small repository.

**Step 5:** Commit scripts and manifests without committing downloaded third-party sources.

### Task 4: Download and Inspect Selected Projects

**Files:**
- Local only: `references/repos/*`
- Modify: `research/notes/github-project-audit-2026-07-22.md`

**Step 1:** Download pinned, shallow checkouts of robomimic, PythonRobotics, py_trees, multimodal-bt-generation, KIOS, MimicLabs, CP-Gen, OK-Robot, ACT++, MobileManiBench, RoboMonkey, and the same-competition diagnostic fork.

**Step 2:** Do not download model weights, datasets, Git LFS assets, or private/authorized components.

**Step 3:** Inspect README, license, dependency files, and relevant code paths in each checkout.

**Step 4:** Update the audit note with verified local paths, commits, useful modules, and warnings.

**Step 5:** Run reference commit verification and confirm all reference source directories remain ignored by top-level Git.

### Task 5: Add Decision and Upstream Records

**Files:**
- Create: `decisions/README.md`
- Create: `decisions/0001-hybrid-competition-architecture.md`
- Create: `decisions/0002-reference-code-policy.md`
- Create: `config/upstream-lock.json`

**Step 1:** Record the selected hybrid architecture and the evidence needed to revisit it.

**Step 2:** Record the policy for competitor code, restricted licenses, and clean implementation boundaries.

**Step 3:** Lock the official repository URL, branch, commit, and audit date.

**Step 4:** Validate the JSON and referenced commit, then commit.

### Task 6: Strengthen Workspace Verification

**Files:**
- Modify: `scripts/check_workspace.sh`
- Modify: `README.md`

**Step 1:** Add checks for governance files, source ledger, reference manifest, upstream lock, unique source IDs, JSON validity, official commit, and ignored third-party directories.

**Step 2:** Add README navigation to status, repository survey, module roadmap, decisions, source ledger, and reference management.

**Step 3:** Run `bash scripts/check_workspace.sh` and resolve every failure.

**Step 4:** Inspect `git status`, `git ls-files`, and tracked file sizes to ensure no nested repositories, secrets, artifacts, or large files are staged.

**Step 5:** Commit verification and README updates.

### Task 7: Create and Verify the Private GitHub Repository

**Files:**
- Git metadata and remote configuration only

**Step 1:** Create `wp-a/jciiot-2026-workspace` with `--private` and set it as `origin`.

**Step 2:** Push branch `main`.

**Step 3:** Verify `gh repo view` reports `PRIVATE`, the remote URL is correct, and the local branch tracks `origin/main`.

**Step 4:** Run the complete workspace and reference verification after the push.

**Step 5:** Record the initialization in `CHANGELOG.md`, commit, and push the final state.
