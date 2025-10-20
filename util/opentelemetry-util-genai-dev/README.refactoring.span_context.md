# GenAI Span Context Refactoring Plan

Date: 2025-10-16
Branch: genai-utils-e2e-dev-span-context

## Original request
Do a thorough review of code in 
instrumentation-genai/opentelemetry-instrumentation-langchain-dev/src/opentelemetry/instrumentation/langchain/callback_handler.py
util/opentelemetry-util-genai-dev
util/opentelemetry-util-genai-evals-deepeval. Identify  
Identify how span_context is used currently, specifically for tagging properly all of the telemetry, including log records with the trace/span ids to correlate it to the span.

High-level thinking:
llm_start(LLMInvocation) - starts a span, and stores span in the invocation object
llm_end - stops the span, and update span in the LLMInvocation
span emitter - emits the span in the invocation
metric emitter - emits a metric measurement
log emitter - emits messages with the span context, and span event (or whatever sets all of the span context/attributes on the log)
evaluator receives llminvocation, with the original span and context
runs evaluation and creates evaluation result
submits evaluation result and invocation to handler on_evaluationResult
which calls the emitter to create an evaluationResult event with the invocation span context, which is still referring to the original LLMInvocation. 

Do thorough research of the codebase and provide all of the details for how it is implemented now. Provide ascii sequence or lifecycle diagram, idfeally similar to util/README.architecture.packages.md but focusing on span and context. Document the desired design, current design and tasks to fix it in the document util/opentelemetry-util-genai-dev/README.refactoring.span_context_codex.md

## Purpose
Provide a deep analysis of the current span context lifecycle across GenAI instrumentation (core util package, LangChain instrumentation, Deepeval evaluator, emitters) and define the desired unified design for reliable trace/span correlation across:
- Spans
- Metrics
- Events (Evaluation + Conversation)
- Log records (SDKLogRecord-based emitters)
- Evaluation results

Deliver concrete tasks to close the gap between current and desired behavior.

---
## Overview of Key Types & Components

| Component | Responsibility | Span Field Usage |
|-----------|----------------|------------------|
| `LLMInvocation` / `AgentInvocation` / `Workflow` / `Task` | Dataclasses representing GenAI operations | Field `span: Optional[Span]` populated by span emitters |
| `TelemetryHandler` | Lifecycle orchestration (start/stop/fail), evaluation trigger | Creates spans via emitter pipeline (`on_start`), logs span context via debug API |
| Span Emitters (`span.py`) | Create and finish spans; assign `invocation.span` | Use tracer.start_as_current_span; set semantic attributes |
| Metrics Emitters (`metrics.py`) | Record duration, tokens histograms | Use `trace.set_span_in_context(span)` for workflow/agent/task/llm metrics (duration/tokens) except evaluation metrics |
| Evaluation Metrics Emitter (`evaluation.py` → `EvaluationMetricsEmitter`) | Record canonical evaluation metric histograms | DOES NOT attach context (no `context=`) currently |
| Evaluation Events Emitter (`evaluation.py` → `EvaluationEventsEmitter`) | Emits one event per evaluation result | Derives span_context from `invocation.span.context` or fallback to `get_span_context()`; sets `span_id` & `trace_id` on event |
| Splunk Emitters (`emitters/splunk.py`) | Emit aggregated log records with evaluation batches and conversation details | Add `trace_id` / `span_id` as attributes; do not create log record with explicit span context API (SDKLogRecord has no trace fields directly passed here) |
| Evaluation Manager (`evaluators/manager.py`) | Async evaluation scheduling & sampling | Samples based on `invocation.span.get_span_context().trace_id` if available |
| Deepeval Evaluator (`util-genai-evals-deepeval`) | Produces `EvaluationResult` objects | Logs debug start; does not modify span context |
| Debug Module (`debug.py`) | Conditional structured logging with span hex IDs | Uses `span.get_span_context()` to render hex IDs |
| LangChain Callback Handler | Bridges LangChain runs to GenAI dataclasses and invokes handler | Calls handler start/stop; evaluation invoked after stop for LLMs |

## Current Lifecycle (LLMInvocation)

```text
User code / LangChain
   │
   ├─> CallbackHandler.on_chat_model_start / on_llm_start
   │      • Build UtilLLMInvocation dataclass
   │      • handler.start_llm(invocation)
   │            - TelemetryHandler.start_llm
   │                - debug log: handler.start_llm.begin
   │                - span emitter .on_start creates span, sets invocation.span
   │                - debug log: handler.start_llm.span_created (trace_id/span_id)
   │
   │
   ├─> Model executes
   │
   ├─> CallbackHandler.on_llm_end
   │      • Populate output_messages, usage tokens, response_id/model
   │      • handler.stop_llm(invocation)
   │            - TelemetryHandler.stop_llm
   │                - span emitter .on_end finalizes span
   │                - metrics emitters record duration & token metrics with context=set_span_in_context(span)
   │                - debug log: handler.stop_llm.complete (trace/span, duration)
   │                - evaluation trigger: evaluate_llm(invocation)
   │
   ├─> Evaluation Manager (async or immediate)
   │      • Sampler inspects invocation.span.get_span_context().trace_id
   │      • Evaluator (e.g., DeepevalEvaluator._evaluate_generic)
   │            - debug log: evaluator.deepeval.start
   │            - produce EvaluationResult(s)
   │
   ├─> TelemetryHandler receives evaluation results via callback (EvaluationManager completion)
   │      • handler._emitter.on_evaluation_results(results, invocation)
   │            - EvaluationMetricsEmitter: records histograms (NO context)
   │            - EvaluationEventsEmitter: emits events with span_id/trace_id copied from invocation.span
   │
   └─> Downstream backends ingest:
          • Finished span (trace correlation root)
          • Duration & token metrics (with context)
          • Evaluation metrics (UNSCOPED to trace/span currently)
          • Evaluation events (scoped by explicit trace_id/span_id fields)
```

### AgentInvocation / Workflow / Task Similarities
- `start_agent` / `stop_agent` creates & finalizes spans (agent span emitter). Evaluation for agent triggered on `stop_agent`.
- Duration metrics for agent/workflow/task include context. No evaluation metrics for workflow/task currently.
- Splunk evaluation results emitter attaches `trace_id`/`span_id` as attributes (hex strings) but not via log record context linking API.

## Current Span Context Handling Summary

| Telemetry Artifact | Has trace/span correlation? | Mechanism |
|--------------------|-----------------------------|-----------|
| Span | YES | Native SDK span object |
| Token & Duration Metrics | YES | `context=trace.set_span_in_context(span)` on record |
| Evaluation Metric Histograms | NO (missing) | Recorded without `context` -> not correlated |
| Evaluation Events | YES | `span_id`/`trace_id` fields on Event (from span context) |
| Splunk Aggregated Evaluation Log Record | Partial | Adds `trace_id`/`span_id` as attributes only |
| Debug Logs | YES (optional) | Hex IDs rendered from span context |
| Evaluation Sampling | YES (trace_id-based sampling) | Uses `span.get_span_context().trace_id` |

## Identified Gaps

1. Missing span context on evaluation metric histograms (cannot join metrics to spans via standard metrics->exemplars or context correlation). 
2. Inconsistent retrieval pattern (`get_span_context()` vs `span.context` attribute) — recently improved in `EvaluationEventsEmitter`, still inconsistent elsewhere.
3. Splunk evaluation log records do not embed native log record trace/span correlation (if SDKLogRecord supports implicit context via active span, consider using context parameter or structured fields rather than attributes only). 
4. No unified helper for extracting span context (repeated try/except blocks). 
5. Evaluation events rely on manually copying trace_id/span_id; other emitters (metrics) rely on context injection — design could unify on a helper `extract_span_context(invocation)`.
6. Potential race: evaluation may occur after span end (currently acceptable; context still valid, but document explicitly). 
7. No tests asserting evaluation metric histograms carry span context (they currently do not). 
8. Agent/Workflow/Task evaluation path limited (only LLM & Agent evaluated) — clarify whether to propagate context for future evaluation types.

## Desired Design

Principles:
- Single authoritative source of span context: `invocation.span.get_span_context()` (with graceful fallback to `invocation.span.context`).
- All telemetry artifacts emitted within a GenAI operation should carry trace/span correlation automatically.
- Metrics record with `context` argument when a span exists (including evaluation metrics). 
- Events set `span_id`/`trace_id` (existing) OR rely on active span context if Events API begins supporting implicit linkage. Keep explicit fields for now.
- Log records (Splunk emitters) should preferably be emitted while the span is active OR include explicit correlation fields; consider optional injection of `context` when creating an SDKLogRecord if the API permits (investigate). If not, continue adding attributes but standardize hex formatting.
- Provide utility: `span_ctx = safe_span_context(obj)` returning a validated context or None.
- Provide utility: `span_hex_ids(span_ctx)` returning `(trace_id_hex, span_id_hex)`.
- Ensure evaluation occurs after span end is documented; if correlation relies on context, safe because SpanContext remains immutable.

### Unified Lifecycle (Desired)
```text
start_llm(inv)
  -> create span (inv.span)
  -> record start debug
  -> metrics (start counters, optional) with context
end_llm(inv)
  -> finalize span
  -> duration/tokens metrics with context
  -> trigger evaluation
evaluate(inv)
  -> evaluator produces results
  -> evaluation metrics emitter records histograms WITH context (trace.set_span_in_context(inv.span))
  -> evaluation events emitter emits events with span_id/trace_id (helper used)
  -> aggregated log records use helper for hex IDs
```

## Helper Functions (New)
`safe_span_context(span_obj) -> SpanContext | None`
`span_hex_ids(span_context) -> (trace_hex | None, span_hex | None)`
`with_span_context(span, fn)` — convenience for metrics emission.

---
## Refactoring Task List

| ID | Task | Description | Effort | Status |
|----|------|-------------|--------|--------|
| SC-1 | Add span context helper module | Implement `span_context_utils.py` with extraction + hex formatting | Low | Pending |
| SC-2 | Update all emitters to use helper | Replace duplicated get_span_context blocks (handler, evaluation, metrics, splunk) | Med | Pending |
| SC-3 | Attach context to evaluation metrics | Pass `context=trace.set_span_in_context(invocation.span)` in `EvaluationMetricsEmitter` histogram.record | Low | Pending |
| SC-4 | Hex formatting consistency | Ensure trace/span hex strings are always 32/16 length | Low | Pending |
| SC-5 | Splunk emitters context research | Investigate whether `SDKLogRecord` supports direct context injection; update emission if possible | Med | Pending |
| SC-6 | Add tests for evaluation metrics correlation | New test: assert exemplar or metric data includes trace_id (may require OTel SDK support; if not, ensure context recorded by manual inspection or rely on future metrics->logs correlation additions) | Med | Pending |
| SC-7 | Expand debug logs | Add logs for evaluation metrics emission start/end including trace/span IDs | Low | Pending |
| SC-8 | Document evaluation-after-span-end semantics | README note clarifying SpanContext remains valid post-end | Low | Pending |
| SC-9 | Agent/Workflow future evaluation hooks | Stub evaluation path with context extraction ready | Low | Pending |
| SC-10 | Add fallback for custom span objects | Use attribute priority: `span.context` then `get_span_context()` | Low | Done (events emitter) |
| SC-11 | Ensure thread-safety | Review helper for pure functional nature (no shared mutation) | Low | Pending |
| SC-12 | Update architecture README | Link to this refactoring plan and summarize changes | Low | Pending |

---
## Proposed Implementation Steps (Sequenced)
1. SC-1: Introduce `span_context_utils.py` containing:
   ```python
   def extract_span_context(span) -> Any | None:
       if span is None: return None
       ctx = getattr(span, "context", None)
       if ctx and getattr(ctx, "trace_id", None): return ctx
       try: return span.get_span_context()
       except Exception: return None
   def span_hex_ids(ctx):
       if ctx and getattr(ctx, "is_valid", False):
           return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
       return None, None
   ```
2. SC-2: Replace ad-hoc code in `handler.py`, `evaluation.py`, `splunk.py`, `metrics.py`, `debug.py` with helper usage.
3. SC-3: Modify `EvaluationMetricsEmitter.on_evaluation_results` to build `context = trace.set_span_in_context(invocation.span)` and pass to `histogram.record`.
4. SC-5: Investigate `SDKLogRecord` for context support (if not, document limitation). If possible, create record earlier with active span.
5. SC-6: Add tests:
   - LLM invocation + evaluation metrics: confirm metrics present; optionally ensure `span.get_span_context().trace_id` matches exported metric points (depends on SDK feature set).
   - Event test already exists for span context -> extend to metrics.
6. SC-7: Add debug logs `emitter.evaluation.metrics.record` per metric canonical name.
7. SC-8 / SC-12: Update READMEs.

---
## Testing Strategy
- Unit tests for helper functions (valid span, ended span, invalid span).
- Evaluation metrics correlation test (if OTel Python supports retrieving exemplar or context). If not feasible, mark test as expected limitation with TODO.
- Regression tests for existing evaluation events remain green.
- Integration test through LangChain callback handler verifying full lifecycle after refactor.

---
## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Lack of API for metric-span correlation introspection | Document limitation; rely on span context for events/logs; consider future exemplars integration |
| Performance overhead adding context per evaluation metric | Minimal; single context construction call; histograms fixed set |
| Breaking change for users relying on absence of evaluation metrics context | Unlikely; addition is additive |
| Custom span implementations missing `get_span_context` | Helper prioritizes `span.context` attribute first |

---
## Open Questions
1. Should evaluation events use implicit active span instead of explicit IDs? (Deferred — explicit better for now.)
2. Should we add exemplar recording linking evaluation score to its span? (Requires stable API support.)
3. Should Splunk log records attempt to mimic OTel Logs semantic fields for trace/span IDs rather than custom attributes? (Investigate in SC-5.)

---
## Completion Criteria
- All emitters use helper for span context extraction.
- Evaluation metrics correlated with span context.
- Documentation updated (this file + architecture README link).
- Tests added for new helper and evaluation metrics correlation path.
- Debug logs enriched but remain opt-in.

---
## Next Actions
Begin SC-1 and SC-3 in a dedicated branch (e.g., `genai-span-context-refactor`) and iterate with tests before broader emitter updates.

---
Maintained by automated AI coder; update task statuses upon implementation.
