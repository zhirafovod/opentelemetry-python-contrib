# GenAI Util Refactoring Plan: Rename Task -> Step

This document is the authoritative, iterative plan for refactoring all occurrences of the GenAI "Task" concept to "Step" across development packages in this mono-repo. No backward compatibility will be maintained. All code, docs, examples, metrics, and attributes MUST exclusively use "Step" after completion.

Progress should be appended under the "Refactoring Progress Log" section with dated entries. Each completed action should reference commit hashes when available.

## Scope

Included packages (paths):

- `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/` (core types, handler, emitters, instruments, callbacks, attributes, config, tests)
- `util/opentelemetry-util-genai-dev/src` (any additional subpackages referencing Task)
- `util/opentelemetry-util-genai-emitters-splunk`
- `util/opentelemetry-util-genai-emitters-traceloop`
- `util/opentelemetry-util-genai-evals-deepeval`
- `util/opentelemetry-util-genai-evals-nltk`
- `util/opentelemetry-util-genai-traceloop-translator`
- `instrumentation-genai/opentelemetry-instrumentation-langchain-dev` (source + tests + examples + docs)

Explicitly out of scope (do NOT modify):

- Non-GenAI util packages unrelated to agentic workflow abstractions.
- Historical changelogs predating the refactor (no retroactive edits besides new entry).

## Global Renaming Map

Core type & lifecycle:

- Class `Task` -> `Step`
- Handler methods: `start_task` -> `start_step`; `stop_task` -> `stop_step`; `fail_task` -> `fail_step`
- Generic handler dispatch branches referencing `Task` updated to `Step`
- Registry logic (`finish_by_run_id`, `fail_by_run_id`, `start`, `finish`, `fail`) update `Task` cases

Dataclass fields (retain semantics, only names containing `task`):

- `task_type` -> `step_type`
- (Other fields remain: `objective`, `source`, `assigned_agent`, `status`, `description`, `input_data`, `output_data`)

Constants / attribute keys:

- `GEN_AI_TASK_NAME` -> `GEN_AI_STEP_NAME` value `gen_ai.step.name`
- `GEN_AI_TASK_TYPE` -> `GEN_AI_STEP_TYPE` value `gen_ai.step.type`
- `GEN_AI_TASK_OBJECTIVE` -> `GEN_AI_STEP_OBJECTIVE` value `gen_ai.step.objective`
- `GEN_AI_TASK_SOURCE` -> `GEN_AI_STEP_SOURCE` value `gen_ai.step.source`
- `GEN_AI_TASK_ASSIGNED_AGENT` -> `GEN_AI_STEP_ASSIGNED_AGENT` value `gen_ai.step.assigned_agent`
- `GEN_AI_TASK_STATUS` -> `GEN_AI_STEP_STATUS` value `gen_ai.step.status`
- `GEN_AI_TASK_INPUT_DATA` -> `GEN_AI_STEP_INPUT_DATA` value `gen_ai.step.input_data`
- `GEN_AI_TASK_OUTPUT_DATA` -> `GEN_AI_STEP_OUTPUT_DATA` value `gen_ai.step.output_data`

Span names & formatting:

- Existing pattern: `gen_ai.task {task.name}` -> `gen_ai.step {step.name}`
- Any explicit span kind mapping enumerations referencing `TASK` -> `STEP` (e.g., semconv enums).

Metrics:

- Histogram / timing metrics: `gen_ai.task.duration` -> `gen_ai.step.duration`
- Any other metric attribute keys: replace `gen_ai.task.*` -> `gen_ai.step.*`

Events & logs:

- Event names such as `gen_ai.client.task.operation.details` -> `gen_ai.client.step.operation.details`
- Log attribute keys `gen_ai.task.*` -> `gen_ai.step.*`

Environment variables: (None found directly with `TASK` prefix; if any appear later, rename similarly.)

Documentation and filenames:

- Any doc filename containing `task` -> replace with `step` (case-insensitive). For example: `callbacks_mapping.md` content updates, examples, README sections.
- Update explanatory prose: "task" -> "step" when referring to the concept (avoid changing unrelated natural language if meaning differs, but here goal is eradication—replace all conceptual mentions).

Examples & tests:

- Variable names in examples: `task` -> `step` where representing the concept.
- Test assertions referencing keys (`gen_ai.task.type`) updated.
- Test classes or helper methods named with `task` updated.

## Semantic Considerations

"Step" replaces "Task" as the discrete unit of work within workflows and agents.

Impacts:

- Trace readability: span names will shift; no compatibility layer; downstream analysis tools must adapt.
- Metrics dashboards must be adjusted to new metric names (`gen_ai.step.*`).
- Any external JSON/log consumers expecting `gen_ai.task.*` will break—acceptable per scope.
- Evaluation flows unaffected except if they inspect entity type names; ensure no evaluator logic hardcodes `Task`.

## Ordered Execution Plan

1. Freeze branch (ensure no concurrent merges touching these files).
2. Implement core type rename in `types.py` (Task -> Step) including `__all__` update.
3. Rename handler methods in `handler.py`; update all internal references.
4. Update attribute constants in `attributes.py`.
5. Adjust emitters (`span.py`, `metrics.py`, `utils.py`) replacing keys, span names, event names.
6. Update instruments (`instruments.py`) metric names.
7. Update semconv or enum declarations referencing TASK (e.g., `SpanKindValues.TASK` -> `STEP`).
8. Perform recursive search & replace across scoped directories for:
   - `Task(` -> `Step(`
   - `task_type` -> `step_type`
   - `gen_ai.task.` -> `gen_ai.step.`
   - `start_task` -> `start_step`
   - `stop_task` -> `stop_step`
   - `fail_task` -> `fail_step`
   - span name format `gen_ai.task` -> `gen_ai.step`
   - Enum / constants `TASK` -> `STEP` (avoid unrelated contexts)
9. Update instrumentation (`callback_handler.py` etc.) mapping logic producing Task entities.
10. Refactor tests: search for regex `task` (case-insensitive) and update expectations, fixtures, variable names.
11. Rename example variable names and textual prompts containing conceptual "task" to "step" where they refer to Task concept.
12. Update documentation files and READMEs; rename file names if contain "task".
13. Run `pytest -k genai` (or full suite) to reveal residual references; fix until green.
14. Run grep searches to assert zero remaining `gen_ai.task.` keys and `Task` symbol imports in scoped packages.
15. Update CHANGELOG or create temporary dev log entry summarizing the refactor (since dev package, may log in plan only if CHANGELOG policy differs).
16. Final validation: metrics registration, emitter pipeline, example scripts run sanity check.
17. Remove this plan's TODO items referencing completion milestones.

## Detailed Mechanical Steps & Commands (indicative)

(Not executed automatically; coder performs.)

Search verification:

- grep -R "Task" util/opentelemetry-util-genai-dev/src | wc -l
- grep -R "gen_ai.task" util/opentelemetry-util-genai-dev/src | wc -l

Refactor sequence suggestion:

A. Edit `types.py`
B. Edit `handler.py`
C. Edit `attributes.py`
D. Edit emitters: `emitters/span.py`, `emitters/metrics.py`, `emitters/utils.py`
E. Edit `instruments.py`
F. Edit instrumentation package enums / usage (`semconv_ai.py`, `callback_handler.py`)
G. Update tests and examples
H. Update docs & READMEs

## Edge Cases / Checks

- Ensure no accidental rename inside third-party import strings or unrelated textual examples where "task" is generic (intentionally replace anyway per eradication directive).
- Confirm `__all__` export updated.
- Confirm registry dispatch branches updated; otherwise finish/fail operations on Steps would not trigger.
- Validate metrics instrumentation still registers after name change.
- Validate no leftover environment variable assumptions.

## Validation Strategy

1. Unit tests: run existing suite; focus on failing imports & assertion mismatches.
2. Grep: zero matches for `gen_ai.task.` after completion.
3. Runtime smoke test: execute at least one LangChain example to ensure spans create with `gen_ai.step` prefix and no handler attribute errors.
4. Metrics introspection: instrument a Step and verify `gen_ai.step.duration` appears.

## Rollback Strategy (if critical failures)

Since no backward compatibility is desired, rollback implies reverting to pre-refactor commit using git (e.g., `git revert` or branch reset). Document in Progress Log.

## Progress Log Template

Append entries chronologically.

```markdown
### 2025-10-27
- [ ] Step 2: types.py renamed Task->Step (commit: <hash>)
- [ ] Step 3: handler.py lifecycle methods updated (commit: <hash>)
- [ ] Step 4: attributes constants updated (commit: <hash>)
...
```

## Outstanding Decisions

- Whether to rename `task_type` -> `step_type` (DECIDED: yes)
- Whether to normalize any natural language in prompts (DECIDED: yes, replace conceptual mentions)
- Any migration guide? (DECIDED: Not required in dev packages)

## Approval & Execution

Upon approval, proceed sequentially. Keep this file updated after each major change.

---
Maintainer: (add name)
Refactor Start Date: 2025-10-27
Target Completion: (set date)
