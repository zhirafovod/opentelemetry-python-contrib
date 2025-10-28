# Refactoring Plan: Extract Evaluator Logic to `opentelemetry-util-genai-evals`

This plan describes separating evaluator-related code from `opentelemetry-util-genai-dev` into a new pluggable package `opentelemetry-util-genai-evals`. Downstream packages (e.g. `opentelemetry-util-genai-evals-deepeval`, `opentelemetry-util-genai-evals-nltk`) will depend on the new base evaluator package. The core util package must remain lightweight, loading evaluation features only when the evals package is installed via a generic completion_callback

Assumption: Target plan file path is `util/evals.refactoring.md` (user said `utils/`; repo uses `util/`).

## Goals

- Decouple evaluator lifecycle (manager, async queue, sampling, normalization, registry) from core telemetry handler.
- Provide clear extension surface for external evaluator integrations.
- Maintain existing environment variable UX with minimal rename (add `EVALS_` scoping where logical).
- Preserve public contract for invoking evaluations via completion callbacks when evals package present.
- Zero mandatory dependency from core util to any evaluation runtime libraries (like deepeval or nltk).
- Evaluator package should use the standard project.entry-points.opentelemetry_genai_upload_hook entry point on install to be enabled. 

## In-Scope Components (to extract)

From `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/evaluators/`:

- `base.py` (Evaluator base class)
- `manager.py` (async queue, sampling, worker thread)
- `builtins.py` (simple builtin evaluators)
- `normalize.py` (skip flags & normalization rules)
- `registry.py` (registration + default metrics mapping)
- Any evaluation-specific helper functions imported only by the above.
- Evaluation-related constants & types used solely by evaluators.

Handler integration points to modify:

- In `handler.py`: remove direct access to evaluator, or evaluator manager storage;
- Replace internal `_evaluation_manager` creation with a more generic `_completion_callback_manager` or similar concept

Emitters integration:

- `emitters/evaluation.py` remains in core package (it emits results). If it uses evaluator registry logic, adjust to use callback-provided results only.

Environment variables (current subset):

- `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS`
- `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION`
- `OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL`
- `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE`

These will be parsed primarily in new evals package. Core util should retain minimal passthrough for sample rate if referenced.

## Out of Scope

- Changing evaluation metric attribute schema.
- Introducing new evaluator types beyond existing builtins during extraction.
- Adding backward compatibility shims for old module paths.
- Introducing new evaluator types beyond existing builtins during extraction.
- Adding backward compatibility shims for old module paths.

## New Package Layout Proposal

```text
opentelemetry-util-genai-evals/
  pyproject.toml
  README.md
  src/opentelemetry/util/genai/evals/
    __init__.py            # public exports (Evaluator, Manager, register, load_evaluators, bootstrap)
    version.py             # __version__
    base.py                # migrated
    manager.py             # migrated
    builtins.py            # migrated (optional to keep minimal examples)
    normalize.py           # migrated
    registry.py            # migrated
    env.py                 # env parsing centralization (sample rate, evaluator list, interval)
    bootstrap.py           # helper to create Manager given handler ref & settings
    defaults.py            # default evaluator factories listing
  tests/ (relocated from original evaluator tests)
```

Entry point groups (in `pyproject.toml`):

- `opentelemetry_util_genai_evaluators` -> factories provided by evals base (builtins).
- Additional evaluator integrations (deepeval, nltk) continue to publish their factories to same group.

## Completion Callback Plugin Architecture (NEW)

Introduce a distinct, pluggable entry point group for generic completion callbacks so third-party packages can hook into any GenAI entity completion without depending on evaluator internals. This is separate from evaluator factories, allowing non-evaluation post-processing (e.g., persistence, custom analytics, enrichment) to participate uniformly.

### Entry Point Group

`opentelemetry_util_genai_completion_callbacks` (new)

Each entry point should resolve to either:

1. A class implementing the `CompletionCallback` protocol (with `on_completion(self, invocation: GenAI) -> None`).
2. A zero-argument factory callable returning an instance implementing the protocol.

### Environment Variables (Proposed)

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS` | Comma-separated list of callback entry point names to enable (subset filter). If unset, all loadable callbacks are attempted. | unset (load all) |
| `OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS` | If truthy (`1,true,yes,on`), skip loading built-in callbacks (including evaluator manager auto-registration). | unset (load defaults) |

### Loading Logic (Handler Integration)

Pseudo-code to be added to `handler.py` (during `TelemetryHandler.__init__` after initializing `_completion_callbacks`):

```python
def _load_completion_callbacks_filtered() -> list[CompletionCallback]:
  try:
    from opentelemetry.util._importlib_metadata import entry_points
  except Exception:
    return []
  requested = os.getenv("OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS")
  requested_set: set[str] | None = None
  if requested:
    requested_set = {n.strip() for n in requested.split(",") if n.strip()}
  callbacks: list[CompletionCallback] = []
  for ep in entry_points(group="opentelemetry_util_genai_completion_callbacks"):
    name = getattr(ep, "name", None)
    if requested_set and name not in requested_set:
      continue
    try:
      obj = ep.load()
      # If obj is class with on_completion, instantiate; if callable returning instance, invoke
      if hasattr(obj, "on_completion"):
        instance = obj  # may be instance or class
        if isinstance(obj, type):  # class -> instantiate
          instance = obj()
      elif callable(obj):
        instance = obj()
      else:
        continue
      if not hasattr(instance, "on_completion"):
        continue
      callbacks.append(instance)  # type: ignore[arg-type]
    except Exception:
      _LOGGER.debug("Failed to load completion callback %s", name, exc_info=True)
      continue
  return callbacks

# In TelemetryHandler.__init__ after self._completion_callbacks initialized:
if not _is_truthy_env(os.getenv("OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS")):
  for cb in _load_completion_callbacks_filtered():
    self.register_completion_callback(cb)
```

### Evaluator Manager Interaction

Evaluator manager remains a special completion callback (registered only if evaluators are present). Its registration should respect `OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS` (i.e., skip if disabled). The environment variable `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS` does NOT need to list the manager; it is implicit when evaluators exist unless defaults are disabled.

### Mechanical Tasks (Augment Existing Migration List)

Add after existing step 6 (handler modifications):

6a. Define new entry point group in core util `pyproject.toml` (or optionally in evals package if base callbacks shipped there) under `[project.entry-points.opentelemetry_util_genai_completion_callbacks]` with at least one example (e.g., noop or simple logging callback).

6b. Implement `_load_completion_callbacks_filtered` in `handler.py` and invoke inside `TelemetryHandler.__init__` honoring env variables.

6c. Update evaluator manager bootstrap to check `OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS` before registering.

6d. Write tests:

- Loading all callbacks when no env filter set.
- Filtering by name (set env with single callback).
- Disabling defaults prevents evaluator manager registration.
- Faulty callback entry point does not break initialization (silently skipped).

6e. Documentation: extend architecture doc with a section "Completion Callback Plugins" including env vars, expected protocol surface, and example `pyproject.toml` snippet.

6f. CHANGELOG entry referencing new entry point group and environment variables.

6g. Grep validation: ensure no stale references to old internal evaluator callback registration remain after refactor.

### Risks / Mitigations (Additional)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Callback raises exception on load | Initialization failure / crash | Wrap load in try/except, log debug, continue |
| User sets filter but names mismatch | No callbacks loaded unintentionally | Emit debug log listing requested names & loaded subset; consider warning if result empty |
| Ordering of callbacks matters (e.g., evaluator should run before persistence) | Unexpected side effects order | For now rely on entry point iteration order; later allow `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACK_ORDER` if needed |
| Disable defaults hides evaluator manager silently | Confusion for users expecting evaluations | Document clearly; add debug log when defaults disabled |

### Future Enhancements (Not In Scope Now)

- Callback dependency ordering or priority field.
- Support async callbacks (awaitable `on_completion`).
- Rich health reporting for failed callbacks.
- Separate entry point group for pre-start hooks (different lifecycle phase).


Core util modifications:

- Remove evaluator module directory.
- Add optional import sequence:

```python
try:
    from opentelemetry.util.genai.evals.bootstrap import create_evaluation_manager
except ImportError:
    create_evaluation_manager = None
```

- In handler initialization: if `create_evaluation_manager` exists, create manager; else skip.
- Adjust completion callback registration: only register manager if present and has evaluators.

## Integration Points Mapping

| Concern | Before | After |
|---------|--------|-------|
| Manager creation | Direct import + instantiate `Manager(self)` | Conditional call `create_completion_manager(self)` |
| Env parsing | Shared with other emitters | Isolated in `evals.env` (core passes settings or re-parses) |
| Builtin evaluators | Inside `evaluators/builtins.py` | In new package; loaded via entry points |
| Registry lookup | `registry.py` in core | `evals.registry` in new package |
| Emission of results | `handler.evaluation_results()` -> `emitters/evaluation.py` | Unchanged (results objects shape preserved) |

## Mechanical Migration Steps (Ordered)

1. Create new package directory scaffold under `util/` named `opentelemetry-util-genai-evals`.
2. Copy evaluator related modules from core to new package (rename folder path accordingly).
3. Add `env.py` to encapsulate environment variable parsing previously scattered.
4. Implement `bootstrap.py` with `create_completion_manager(handler)` returning manager or `None`.
5. Update imports within moved files to new package path (e.g., relative imports for shared utilities still in core: they may need `from opentelemetry.util.genai.utils import ...`). Ensure no circular dependency.
6. In core util `handler.py`:
   - Remove internal evaluator imports and `_initialize_default_callbacks` logic referencing old modules.
   - Add dynamic import for registered callbacks and attempt manager creation.
   - Update error handling/logging to reflect optional nature.
7. Remove `evaluators/` directory from core (after confirming test coverage migrated).
8. Adjust `emitters/evaluation.py` if it imports registry types; change imports to reference new package or remove dependency (receive `EvaluationResult` only).
9. Update `types.py` export list if `EvaluationResult` remains (it does). Ensure no evaluator base classes linger.
10. Relocate tests targeting evaluator logic into new package `tests/` and update import paths.
11. Update documentation snapshot (`util/README.architecture.packages.md`) removing evaluator internals from core and introducing new package section.
12. Update downstream evaluator integration packages (`deepeval`, `nltk`) pyproject dependencies to add `opentelemetry-util-genai-evals`.
13. Run grep to ensure no residual `evaluators.` references in core package:
    - `grep -R "evaluators" util/opentelemetry-util-genai-dev/src`
14. Run pytest subsets: evaluator tests in new package, core util tests, integration tests for deepeval/nltk.
15. Update CHANGELOG (or plan log) with summary of extraction.
16. Final validation: import core handler without evals package installed (should not raise), then install evals package and confirm manager loads and evaluation metrics emitted.
