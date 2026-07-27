# Official Execute Entry-Point Integration Design

Date: 2026-07-28

## Context

The fixed candidate at workspace commit `05a7b0d` reaches the official maximum
score on all five scenes when `scripts/run_official_experiment.py` calls
`robot_agent.workflows.competition_flow.run_official_task()` directly. That
proves the controller and generated trajectories, but it does not prove that a
reviewer can reproduce the same behavior by running the official `app.py` and
clicking Execute.

The official Execute path is fixed:

```text
app.py
  -> task_subprocess_runner.py
  -> RobotAgent(...)
  -> RobotAgent.run(prompt)
  -> skills registered by skills/library.py
  -> official recorder and scorer
```

`workflows/competition_flow.py` is not imported by that path. The integration
gap is therefore at the allowed skill factory, not in the geometric controller
or scorer.

## Requirements

- Preserve the existing deterministic geometric state machine and its physical
  grasp, lift, transport, placement, and verification evidence.
- Make the unmodified official Execute path invoke that state machine.
- Do not modify `app.py`, `task_subprocess_runner.py`, `core/`,
  `environments/`, `knowledge/task_config.json`, or the scorer.
- Do not require an LLM service or let an LLM generate low-level actions during
  scored execution.
- Resolve the selected task from the official `task_index`; read but never
  modify `knowledge/task_config.json`.
- Fail explicitly on missing metadata, invalid task configuration, or workflow
  failure. Do not teleport objects or report success without physical evidence.

## Considered Approaches

### 1. Register a deterministic competition skill (selected)

Add `skills/competition_task.py`, register it first in the allowed
`skills/library.py`, and use the official `GATE_PLANNER=false` feature gate for
scored execution. `RobotAgent.run()` then follows its existing single-skill
fallback and selects the competition skill for any non-empty task prompt.

This is the smallest change that reaches the official Execute path, keeps the
controller in allowed directories, and removes evaluator-side LLM variance.

### 2. Rewrite move, pick_up, and place_down

Keep the LLM four-step plan and spread competition state across the three
existing skills. This retains planner variance, duplicates orchestration state,
and makes L5's three-object lifecycle difficult to represent safely.

### 3. Keep only the independent experiment runner

Document `scripts/run_official_experiment.py` as the entry point. This preserves
the current code but does not satisfy the official instruction to run `app.py`
and click Execute, so it is rejected.

## Selected Architecture

`CompetitionTaskSkill` owns only entry-point adaptation. It receives the backend,
scene context, and occupancy grid from `wired_skills()`. On `run(context)` it:

1. Reads `task_index` from the metadata produced by the unmodified
   `RobotAgent._skill_metadata()`.
2. Loads the matching task entry from the official `task_config.json`.
3. Calls `run_official_task()` with the same backend, scene context, grid, task,
   and one-attempt policy used by the validated experiment runner.
4. Converts the workflow result into an official `SkillResult`, preserving the
   state history and failure information in the payload.

`skills/library.py` sets the documented planner feature gate before
`AgentConfig` is instantiated and registers `CompetitionTaskSkill` before the
general-purpose official skills. Auxiliary skills remain registered for
inspection and plan-only UI functionality, but scored Execute calls are routed
deterministically.

```text
prompt + task_index
  -> RobotAgent.run (planner gate off)
  -> registry.find
  -> CompetitionTaskSkill
  -> official task_config entry
  -> run_official_task
  -> CompetitionFlow
  -> SkillResult
  -> unmodified trajectory recorder
  -> unmodified official scorer
```

## Error Handling

- Missing or non-integer `task_index`: return a failed `SkillResult` with a
  stable error code.
- Index outside the official task list or malformed task entry: return failure;
  do not silently select L1.
- Workflow exception: convert it to a failed result with exception type and
  message so the subprocess can save a terminal FAIL trajectory and manifest.
- Workflow returns `success=false`: propagate failure and the complete state
  history.

No error path reports success, selects a different task, mutates object poses,
or falls back to teleportation.

## Verification

Testing proceeds in three layers:

1. Unit tests for task-index extraction, official config lookup, workflow-result
   propagation, and explicit failure paths.
2. Integration tests proving the materialized official `skills/library.py`
   registers the competition skill first and causes a default `RobotAgent` to
   have the planner gate disabled without changing core code.
3. Server acceptance on a freshly materialized official commit: invoke the
   official agent/subprocess path for L1-L5, save trajectories, and rescore them
   with the unmodified official scorer. Required result is maximum score, the
   required successful grasp count, zero collision frames, target distance below
   0.8 m, and a successful workflow result for every scene.

The independent multi-seed batch remains valid controller-stability evidence,
but it is not used as evidence for the entry-point integration until layer 3
passes.
