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
