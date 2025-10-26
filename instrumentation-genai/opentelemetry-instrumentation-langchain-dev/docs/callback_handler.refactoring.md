# LangChain Callback Handler Refactoring Plan

Goal: Ensure all spans for a single agent workflow execution share one trace_id with correct parent/child relationships (AgentInvocation root → LLMInvocation → Task(tool_use) → follow‑up LLMInvocation, etc.).

## 1. Current Issues Summary

| Issue | Impact | Notes |
|-------|--------|-------|
| `parent_run_id` stored but never used to set span parent | Each entity starts a fresh trace | SpanEmitter always calls `start_as_current_span` without an explicit parent context; if no active span, new trace root. |
| No span registry (run_id → Span) | Cannot look up parent span by run_id | Callback handler tracks entities, not spans; TelemetryHandler doesn’t expose active span map. |
| Agent / workflow spans may end before children start | Children lose ancestry → new trace | Need lifecycle coordination (defer end until descendants finished). |
| Identity stack tracks run_ids only (no span objects) | Identity attributes propagate but not trace context | Must augment with span references or centralized registry. |
| SpanEmitter oblivious to hierarchy semantics (tool vs model vs task) | Flat sequence of unrelated traces | Parenting logic needs semantic rules. |
| Duplicate doc sections & unclear desired model (fixed) | Confusion for implementers | Mapping file updated with desired unified model. |

## 2. Target Architecture Changes

1. Span Registry: `TelemetryHandler._active_spans: dict[UUID, Span]` with thread-safe access (simple Lock).
2. Parenting Resolution Path:
   - Invocation has `parent_run_id` (explicit) OR implicit top of agent/workflow stack.
   - Before creating a new span, lookup parent span; if found create child via context=`trace.set_span_in_context(parent_span)`.
3. Deferred Termination for Agent/Workflow:
   - Maintain `child_counts: dict[UUID, int]`.
   - On start of any child with parent_run_id=X increment `child_counts[X]`.
   - On end/fail of child decrement; only end agent/workflow span when its count reaches 0 and a completion event (chain_end) or an explicit finalization callback occurs.
4. Tool Parenting Rule:
   - If tool call originates from a model tool request (detect via presence of active LLMInvocation run_id on stack or by passing model run_id explicitly), parent = that LLM span.
   - Else parent = AgentInvocation span (or Workflow if multi-agent orchestration outside agent context).
5. Follow-up LLM Calls After Tool:
   - Parent = tool task span for granular chaining (config flag `GENAI_LANGCHAIN_FLATTEN_TOOL_CHILDREN=true` to flatten to agent parent if desired).
6. Orphan Handling:
   - If parent_run_id provided but parent span not found, still create span (root) and set attribute `gen_ai.parent.missing=true` + `gen_ai.parent.run_id=<uuid>` for diagnostics.

## 3. Step-by-Step Refactoring Tasks

Status Update (Phase 0 Implemented):

- Added deferred stop logic with `pending_stop` flag and `_child_counts` in `LangchainCallbackHandler` so agent/workflow spans remain open until children finish. This should unify trace IDs for sequential agent→LLM→tool chains where previously parent spans ended prematurely.
- Remaining tasks below adjust to reflect what is DONE vs PENDING.

| Step | Description | Files | Category | Status |
|------|-------------|-------|----------|--------|
| 1 | Add span registry + helpers (`register_span`, `get_span`, `unregister_span`) | `util/.../handler.py` | Infra | Pending |
| 2 | Pass handler instance into SpanEmitter (constructor injection or global accessor) | `emitters/span.py`, `emitters/configuration.py` | Infra | Pending |
| 3 | Modify SpanEmitter `on_start` to resolve parent context | `emitters/span.py` | Behavior | Pending |
| 4 | Extend callback handler `_start_entity` to increment parent child count | `instrumentation-genai/.../callback_handler.py` | Behavior | Done (basic) |
| 5 | Extend `_stop_entity` / error paths to decrement and finalize parent when count==0 & pending | `callback_handler.py` | Behavior | Done (basic) |
| 6 | Introduce config flag `GENAI_LANGCHAIN_FLATTEN_TOOL_CHILDREN` | `environment_variables.py`, `callback_handler.py` | Config | Pending |
| 7 | Tool parent detection enhancement (model-issued vs agent orchestration) | `callback_handler.py` | Semantics | Pending |
| 8 | Add orphan diagnostic attributes | `emitters/span.py` | Observability | Pending |
| 9 | Unit tests for unified trace_id scenarios (agent→model→tool→model) | `tests/...` | Testing | Pending |
| 10 | Performance smoke test (ensure negligible overhead) | `tests/...` | Testing | Pending |
| 11 | Update docs to reflect final state & remove "Interim" wording | docs | Docs | Partial (interim added) |
| 12 | Add CHANGELOG entry | `CHANGELOG.md` | Release | Pending |

## 4. Detailed Parenting Algorithm (Pseudo)

Interim (Implemented): Only child counting & deferred stop at callback level.
Next (Pending): SpanEmitter parenting using explicit parent span context.

```python
# In SpanEmitter.on_start(invocation):
parent_ctx = None
explicit_parent_id = getattr(invocation, "parent_run_id", None)
if explicit_parent_id:
    parent_span = handler.get_span(explicit_parent_id)
else:
    parent_span = handler.peek_agent_or_workflow_span()  # new helper using identity stack
if parent_span:
    parent_ctx = trace.set_span_in_context(parent_span)
span_cm = tracer.start_as_current_span(span_name, context=parent_ctx, kind=SpanKind.CLIENT, end_on_exit=False)
span = span_cm.__enter__()
handler.register_span(invocation.run_id, span)
handler.increment_child_count(parent_span)  # if parent
```

```python
# On end/fail:
span.end()
handler.unregister_span(invocation.run_id)
if parent_span:
    handler.decrement_child_count(parent_span)
    if parent_is_agent_or_workflow and child_count == 0 and agent/workflow is logically complete:
        handler.finish_agent_or_workflow(parent_entity)
```

## 5. Data Structures

```python
class TelemetryHandler:
    _active_spans: dict[UUID, Span]
    _child_counts: dict[UUID, int]
    _entity_index: dict[UUID, GenAI]  # already partially exists

    def increment_child_count(self, parent_run_id: UUID): ...
    def decrement_child_count(self, parent_run_id: UUID): ...
    def peek_agent_or_workflow_span(self) -> Span | None: ...
```

## 6. Configuration Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `GENAI_LANGCHAIN_FLATTEN_TOOL_CHILDREN` | Make tool follow-up LLM calls parent to AgentInvocation instead of tool task span | `false` |
| `GENAI_LANGCHAIN_ORPHAN_DIAGNOSTICS` | Emit orphan diagnostics attributes/logs | `true` |

## 7. Testing Matrix

| Scenario | Expected Outcome |
|----------|------------------|
| Simple agent → model | 2 spans same trace; parent-child link set |
| Agent → model(tool request) → tool task | 3 spans same trace; tool parent = model span |
| Agent → model(tool request) → tool task → follow-up model | 4 spans same trace; follow-up model parent = tool task (or agent if flatten flag true) |
| Orphan child (inject bad parent_run_id) | Span created; attribute `gen_ai.parent.missing=true` |
| Parallel tool tasks | All tool spans share agent trace; each child parent = triggering model span |
| Error in tool | Tool span status=ERROR; trace continuity preserved |
| Nested agent inside workflow | Inner agent span child of workflow span (single trace) |

## 8. Migration / Backward Compatibility

- Existing attributes remain unchanged; only parenting changes.
- Tools relying on run_id relationships can continue; adding trace hierarchy is additive.
- If downstream systems assumed each entity is root, they still receive span-level data but now can aggregate.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Long-lived agent/workflow spans accumulate timing including child durations | Document expected semantic (agent duration = total execution); optionally emit separate metric for pure agent logic time if needed later. |
| Incorrect child counting leading to dangling agent span | Defensive checks & timeout fallback (end span if exceeds configurable max duration). |
| Performance overhead of registry lookups | O(1) dict access; negligible. |
| Recursive parent resolution complexity | Limit to explicit `parent_run_id` or top of stack; avoid deep traversal except for diagnostics. |

## 10. Incremental Delivery Plan

1. Implement registry + parenting for LLMInvocation only (minimal change).
2. Extend to Tool Tasks.
3. Add agent/workflow deferred end semantics.
4. Introduce config flags & diagnostics.
5. Finalize tests & docs; update CHANGELOG.

## 11. Acceptance Criteria

- All spans in manual demo have one trace_id (interim: should now hold if agent remains active; verify after tests).
- Parent span IDs set correctly (verify via span exporter inspection or test harness).
- No increase in uncaught exceptions or error logs under standard demo.
- Lint/test suites pass.

## 12. Post-Refactor Enhancements (Future Ideas)

- Add graph export (run_id tree → DOT / JSON).
- Optional span links for sibling tool calls (if flattened parenting).
- Emission of aggregated agent evaluation metrics across child LLMEvaluations.

---
**Status:** Pending implementation.
