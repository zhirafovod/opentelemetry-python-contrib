# Evaluators Refactoring — Findings and Plan

This note summarizes issues observed in the current evaluators implementation and proposes a focused refactor to improve separation of concerns, reuse, and maintainability.

## Key Findings

- Responsibility drift in evaluator plugins
  - `deepeval.py` contains extensive input/output normalization, options coercion, per-metric capability checks, environment key resolution, and custom metric construction. This makes the plugin large, harder to test, and duplicates logic that other evaluators could reuse.
  - Example concerns: custom flattening of inputs/outputs, tool-call detection, retrieval/context extraction, per‑metric skipping logic tightly coupled to test case construction.

- Data normalization duplicated across evaluators
  - Both the Deepeval and NLTK evaluators normalize text in slightly different ways (e.g., extracting text, agent inputs, etc.). This invites drift and subtle inconsistencies in behavior.

- Skip heuristics live in concrete plugins
  - LLM tool‑call only responses are skipped in the Deepeval plugin, but this is not a Deepeval‑specific concept and should be handled centrally to make behavior consistent across evaluators.
  - Agent non‑invoke operations are already skipped in Deepeval; that policy could be centralized too.

- Default plan assembly contains ad‑hoc policy
  - `manager.py` assembles default evaluator/metric plans from plugin registrations. We recently added an inline policy to filter out the builtin `length` metric there. These kinds of policies create hidden behavior coupling and should live in a dedicated policy/config layer.

- Early failure prevents per‑metric skip
  - Deepeval’s `_build_test_case` returns `None` when either input or output is missing. Some metrics (e.g., sentiment on output only) can still run. Early failure prevents per‑metric skip/selection based on actual requirements.

- Plugin is monolithic
  - Deepeval evaluator mixes: metric registry, option parsing/coercion, OpenAI key setup, LLM test case adaptation, Deepeval run orchestration, and result conversion. These should be split into small, testable helpers.

## Refactoring Goals

- Single place for canonical input/output/context extraction reusable by all evaluators.
- Centralized skip policy (e.g., tool‑only LLM outputs, agent non‑invoke operations) so all evaluators behave consistently.
- Per‑metric capability checks drive evaluation decisions (don’t fail globally when some required fields are missing).
- Slim evaluator plugins by extracting common helpers and plugin‑specific helpers into dedicated modules.

## Proposed Design

1) Introduce a shared normalization layer
   - New module: `opentelemetry/util/genai/evaluators/normalize.py`
   - Responsibilities:
     - Canonicalize any `GenAI` invocation into a `CanonicalEvalCase` with:
       - `type_name` (`LLMInvocation`, `AgentInvocation`, …)
       - `input_text`, `output_text` (may be empty), `context`, `retrieval_context`
       - `metadata` (flattened/filtered from attributes)
       - `flags`: `is_tool_only_llm`, `is_agent_non_invoke`, etc.
     - Utilities:
       - `extract_text_from_messages(messages)`
       - `extract_agent_io(agent)` (system + input context + output result + prompt_capture)
       - `extract_context(attributes)` + `extract_retrieval_context(attributes)`
       - `flatten_to_strings(value)` (single authoritative implementation)

2) Centralize evaluation gating/skip logic
   - Extend `Manager` to call the normalizer and skip items by common policy:
     - Skip `LLMInvocation` when `is_tool_only_llm`.
     - Skip `AgentInvocation` when not `invoke`.
   - Optionally expose a small hook interface (e.g., `should_evaluate(invocation)`) in a new `policy.py` for custom user policies.

3) Per‑metric capability declaration (optional, incremental)
   - Allow evaluators to declare capability requirements per metric (e.g., `requires_input`, `requires_output`).
   - Manager (or evaluator) can use these to selectively run metrics instead of failing the whole evaluator when some fields are missing.
   - For Deepeval specifically: always build an LLM test case (with empty strings when necessary) and use per‑metric checks to mark missing requirements as `skipped` rather than erroring out globally.

4) Slim the Deepeval evaluator
   - Split plugin into:
     - `deepeval_metrics.py`: metric registry mapping, custom GEval builders, option coercion helpers.
     - `deepeval_adapter.py`: adapters from `CanonicalEvalCase` to Deepeval `LLMTestCase`.
     - `deepeval_runner.py`: run Deepeval evaluation with output capture/suppression.
     - `deepeval.py`: thin Orchestrator that wires the above together.
   - Keep OpenAI key handling plugin‑local but move it into a tiny helper function for clarity.

5) Default plan/config clean‑up
   - Move ad‑hoc metric filtering (e.g., excluding `length`) from `manager.py` to a small policy/config helper so that behavior is discoverable and testable.
   - Consider supporting an environment variable for default metric excludes, e.g., `OTEL_INSTRUMENTATION_GENAI_EVALS_DEFAULT_EXCLUDES="length"`.

## Migration Plan (Incremental)

- Phase 1 (low‑risk):
  - Add `normalize.py` with canonical extractors (`flatten_to_strings`, context/retrieval extractors, agent IO extractors, tool‑only detection).
  - Update Deepeval to use normalizer for building input/output/context, but keep all other code intact.
  - Adjust Deepeval to build an LLM test case even when one of input/output is empty and rely on per‑metric missing‑param skip (no global early failure).

- Phase 2 (consistency):
  - Move tool‑only LLM and non‑invoke agent skip logic into `Manager` using normalizer flags.
  - Remove corresponding skip logic from Deepeval (and other evaluators) to avoid duplication.

- Phase 3 (plugin slimming):
  - Extract Deepeval helpers into `deepeval_metrics.py`, `deepeval_adapter.py`, `deepeval_runner.py`.
  - Keep `deepeval.py` as a thin orchestrator and public entrypoint.

- Phase 4 (policy/config):
  - Introduce default metric exclude policy in a small `policy.py` or `config.py` and remove the inline filtering from `manager.py`.
  - Document env overrides for defaults and excludes.

## Acceptance Criteria

- Normalization utilities used by Deepeval and NLTK evaluators (no duplicate flatteners).
- Manager applies common skip policies; Deepeval contains no tool‑only detection.
- Deepeval no longer globally errors on missing IO; per‑metric skip is reported when requirements aren’t met.
- Deepeval code is split into small helpers; `deepeval.py` substantially shorter and easier to read.
- Default plan policy (like length exclusion) is isolated and configurable.

## Risks / Notes

- Deepeval’s `LLMTestCase` may require non‑None fields; using empty strings is typically safe and allows per‑metric checks to work. Verify against the installed Deepeval version.
- Keep OpenAI key handling in the plugin to avoid affecting other evaluators or introducing secrets into shared layers.

## Suggested File Additions

-

## Change Log

- Added `evaluators/normalize.py` providing canonical extraction and flags.
- Manager now consults normalization flags to skip:
  - Tool‑call only `LLMInvocation`s.
  - Non‑invoke `AgentInvocation`s.
- Deepeval now builds `LLMTestCase` from the normalizer and no longer performs bespoke agent input/output aggregation.
- Deepeval no longer performs tool‑only skip (centralized in Manager).
- Previous ad‑hoc `length` default metric removed from builtin registration; Manager filters it from defaults (to be moved into a distinct policy/config module in a later phase).
- Added Deepeval helpers:
  - `deepeval_adapter.py` (LLMTestCase builder from normalized data)
  - `deepeval_metrics.py` (metric registry, option coercion, metric instantiation, custom GEval builders)
  - `deepeval_runner.py` (captured evaluation runner)
  - Slimmed `deepeval.py` to orchestrate the above helpers.
- Fixed `_build_test_case` regression (accidental `@staticmethod` decoration) that prevented Deepeval evaluations from running and reinstated calls for LLM invocations.

- `opentelemetry/util/genai/evaluators/normalize.py`
- `opentelemetry/util/genai/evaluators/policy.py` (optional)
- `opentelemetry/util/genai/evaluators/deepeval_metrics.py`
- `opentelemetry/util/genai/evaluators/deepeval_adapter.py`
- `opentelemetry/util/genai/evaluators/deepeval_runner.py`
