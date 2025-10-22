# Agent Type Refactoring Plan

## Current Findings
- **Legacy `AgentInvocation` ambiguity** – Creation and invocation used the same dataclass with a stringly `operation` field, forcing consumers to branch on the literal value. Splitting the model eliminates this conditional logic and better matches the semantic convention lifecycle.
- **Handler lifecycle conflated phases** – `TelemetryHandler.start_agent/stop_agent` treated creation and invocation identically and always pushed agent identity, making it hard to skip evaluations for create-only phases.
- **Integration constructors mirrored the old shape** – The LangChain callback handler (and other emitters) always instantiated `AgentInvocation` regardless of operation, so downstream components could not differentiate phases without inspecting strings.
- **Evaluation skip logic expected `"invoke"`** – Normalization and evaluator adapters gated on the legacy literal `invoke`, meaning spec-compliant `invoke_agent` operations were silently skipped.
- **Span naming already matched the spec** – We intentionally preserved the existing span/attribute formatting (`create_agent {name}` / `invoke_agent {name}`) throughout the refactor.

## Refactoring Proposal
1. **Split the data model**
   - Introduce `AgentCreation` (mirrors shared fields but fixes `operation` to `create_agent`).
   - Restrict `AgentInvocation` to the invocation phase (fix `operation` to `invoke_agent` and move invoke-only fields there).
   - Share common behaviour via a lightweight base mixin if needed.
2. **Lifecycle API adjustments**
   - Add `TelemetryHandler.start_agent_creation/stop_agent_creation/fail_agent_creation` (thin wrappers around shared helpers) while keeping backwards-compatible `start_agent` that dispatches on instance type.
   - Update emitter pipeline registrations so each emitter knows how to handle `AgentCreation` and `AgentInvocation`.
3. **Instrumentation updates**
   - Make LangChain callback handler instantiate the appropriate util type (`AgentCreation` for create operations, `AgentInvocation` for invoke).
   - Audit other emitters/instrumentations (e.g., OpenAI agents, Span translator) for the same change.
4. **Evaluation alignment**
   - Update normalization to treat `AgentInvocation` as evaluable by default and skip instances of `AgentCreation`.
   - Fix evaluator guards (Deepeval and any others) to expect `invoke_agent`.
5. **Semantic attribute enforcement**
   - Ensure both classes expose consistent `semantic_convention_attributes`, keeping span names and `gen_ai.operation.name` aligned with the spec.
6. **Testing & docs**
   - Extend unit tests to cover both types, especially evaluation skip logic and span emission.
   - Update public docs/examples to use the new types explicitly.

## Task Breakdown
1. **Data model**
   - Create `AgentCreation` data class.
   - Update `AgentInvocation` to hard-code `operation="invoke_agent"`.
2. **Handler & emitters**
   - Refactor handler lifecycle methods to dispatch on the new types.
   - Adjust span/content/metrics/evaluation emitters to accept `AgentCreation`.
3. **Instrumentation**
   - Update LangChain callback handler (and other integrations) to instantiate the correct class.
4. **Evaluation pipeline**
   - Fix `normalize_invocation` and manager `_GENAI_TYPE_LOOKUP`.
   - Align Deepeval adapter (and other plugins) with the `invoke_agent` literal.
5. **Verification**
   - Refresh examples/tests to assert both agent phases behave as expected.
   - Document the changes in package README / changelog once implemented.
