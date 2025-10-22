# Agent Type Refactoring Plan

## Current Findings
- **Legacy `AgentInvocation` ambiguity** – Creation and invocation used the same dataclass with a stringly `operation` field, forcing consumers to branch on the literal value. Splitting the model eliminates this conditional logic and better matches the semantic convention lifecycle.
- **Handler lifecycle conflated phases** – `TelemetryHandler.start_agent/stop_agent` treated creation and invocation identically and always pushed agent identity, making it hard to skip evaluations for create-only phases.
- **Integration constructors mirrored the old shape** – The LangChain callback handler (and other emitters) always instantiated `AgentInvocation` regardless of operation, so downstream components could not differentiate phases without inspecting strings.
- **Evaluation skip logic expected `"invoke"`** – Normalization and evaluator adapters gated on the legacy literal `invoke`, meaning spec-compliant `invoke_agent` operations were silently skipped.
- **Span naming already matched the spec** – We intentionally preserved the existing span/attribute formatting (`create_agent {name}` / `invoke_agent {name}`) throughout the refactor.

## Completed Changes
- Added a shared `_BaseAgent` plus explicit `AgentCreation` (fixed `operation="create_agent"`) and `AgentInvocation` (fixed `operation="invoke_agent"` with invoke-only fields); exporter list updated accordingly (`util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py`).
- Telemetry handler now branches on the concrete agent type: only invocation phases push identity, run evaluations, or pop the stack; generic `start/finish/fail` paths accept either class (`util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/handler.py`).
- Span/content emitters, evaluation emitters, and shared utils accept both agent classes while only capturing invoke-specific payloads (e.g., input/output content) for `AgentInvocation` (`.../emitters/span.py`, `.../emitters/content_events.py`, `.../emitters/utils.py`).
- Evaluation pipeline recognizes the split: manager skips `AgentCreation`, normalization flags only non-`invoke_agent` operations, Deepeval adapter requires the spec literal, and `_GENAI_TYPE_LOOKUP` includes the new class (`.../evaluators/manager.py`, `.../evaluators/normalize.py`, `util/opentelemetry-util-genai-evals-deepeval/src/.../deepeval.py`).
- LangChain instrumentation now instantiates `AgentCreation` vs `AgentInvocation`, adjusts context propagation, and updates docs/examples; OpenAI agent tests use `invoke_agent`/`create_agent` explicitly (`instrumentation-genai/opentelemetry-instrumentation-langchain-dev/.../callback_handler.py`, docs/examples, and OpenAI agent unit tests).
- Examples, tests, and utility emitters have been refreshed to use the new types directly (e.g., `langgraph_*` demos, evaluation/metrics tests, NLTK evaluator).
- Documented the refactor scope in this README (new file added to track ongoing work).

## Next Steps
- **Testing**  
  - Run and update automated tests across modules (`pytest util/opentelemetry-util-genai-dev/tests`, `util/opentelemetry-util-genai-evals-*`, instrumentation suites) to confirm the split didn’t introduce regressions.  
  - Add dedicated tests covering agent creation spans/events and ensuring evaluations are skipped.
- **Documentation & examples**  
  - Update public-facing guides and changelog to explain the new data model and migration steps.  
  - Review remaining examples/integration snippets for references to the old `operation` literals.
- **Dependent packages**  
  - Audit other instrumentation packages (e.g., tracer translators, Splunk emitter, Traceloop translator) for any lingering assumptions about `AgentInvocation.operation`.  
  - Coordinate version bumps for packages that depend on the old class shape.
- **Tooling & schema**  
  - Consider adding linting or mypy rules preventing direct string comparisons against `operation`.  
  - Explore exposing helper constructors or factory functions to simplify creation/invocation lifecycle management for SDK users.
