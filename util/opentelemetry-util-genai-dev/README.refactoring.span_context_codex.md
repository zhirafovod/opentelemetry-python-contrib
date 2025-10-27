# Span Context Correlation Audit (Codex)

Scope: `instrumentation-genai/opentelemetry-instrumentation-langchain-dev/src/opentelemetry/instrumentation/langchain/callback_handler.py`, `util/opentelemetry-util-genai-dev`, `util/opentelemetry-util-genai-evals-deepeval`.

---
## Current Span & Context Lifecycle

```text
LangChain Callback Handler                                   GenAI Telemetry Handler + Emitters
--------------------------------------                      --------------------------------------
on_chain_start/_start_entity                                start_*() (handler.py:221-340)
  ↓ register Util* in _entities                                   ↓ CompositeEmitter.on_start() (composite.py:34-78)
  ↓ TelemetryHandler.start_*()                                    ↓ SpanEmitter.on_start(): start_as_current_span(), set invocation.span/context_token (span.py:260-290)
                                                                  ↓ other emitters observe start (metrics/content no-op)
                                                                  
on_chain_end/_stop_entity → TelemetryHandler.stop_*()      stop_llm()/stop_agent()/... (handler.py:257-338)
  ↓ CompositeEmitter.on_end() (order: evaluation → metrics → content_events → span) (composite.py:21-53)
       · MetricsEmitter records histograms with context=trace.set_span_in_context(invocation.span) (metrics.py:214-285)
       · ContentEventsEmitter builds SDKLogRecord (ContentEventsEmitter.on_end, content_events.py:51-76)
       · SpanEmitter.on_end applies attrs, exits context manager, ends span (span.py:292-312)
  ↓ handler._notify_completion() triggers CompletionCallbacks (handler.py:315-338)
       · Evaluation Manager samples using invocation.span.get_span_context().trace_id (manager.py:103-126)
       · Manager.enqueue(invocation) → async evaluator thread
           → evaluator (deepeval) produces EvaluationResult
           → handler.evaluation_results() → CompositeEmitter.on_evaluation_results()
                · EvaluationEventsEmitter emits OTEL events with trace_id/span_id extracted from invocation.span (evaluation.py:326-420)
```

Context propagation helpers:
- `TraceloopCallbackHandler._start_entity` stores spans/agents in `_entities`, attaches context token for workflow/agent stack (callback_handler.py:295-360).
- `SpanEmitter` keeps span active until all other emitters run, because the composite closes spans last.
- `MetricsEmitter` and helpers call `trace.set_span_in_context(span)` so measurements inherit the active trace (metrics.py:214-312, emitters/utils.py:543-580).
- Evaluation events copy `span_context.trace_id`/`span_id` on each event (evaluation.py:326-420).

---
## Span Context Touch Points

- **Instrumentation (LangChain callback handler)**  
  - Creates `Util*` entities, calls `TelemetryHandler.start_*`/`stop_*`, and keeps the OTel context stack in sync via `context_api.attach/detach` (callback_handler.py:295-395).  
  - LLM invocations inherit agent/workflow identity before `start_llm` (callback_handler.py:327-345).

- **Telemetry Handler**  
  - Delegates lifecycle to composite emitters, but does not persist `SpanContext` beyond `invocation.span` (handler.py:221-338).  
  - Debug logging reads `invocation.span.get_span_context()` for trace/span IDs.

- **CompositeEmitter**  
  - Defines deterministic ordering: spans are opened first and closed last; metrics and content events execute while the span is still current (composite.py:18-78).

- **SpanEmitter**  
  - Starts spans with `start_as_current_span(..., end_on_exit=False)` and saves both the span object and context manager token on the invocation (span.py:268-290).  
  - On completion, re-applies finish attrs, exits the context manager, ends the span (span.py:292-312). The span object remains attached to the invocation even after `.end()`.

- **MetricsEmitter & helpers**  
  - Call `trace.set_span_in_context(span)` before recording histograms so exporters can associate measurements with the trace (metrics.py:214-312, emitters/utils.py:543-580).

- **ContentEventsEmitter & helpers**  
  - Build `SDKLogRecord` instances but do **not** set `trace_id`/`span_id`/`context` on the record (content_events.py:68-76, emitters/utils.py:319-540).  
  - As a result, OTLP log exports rely on SDK defaults and currently emit logs without trace/span correlation.

- **Evaluation Pipeline**  
  - Completion callback sampling requires `invocation.span.get_span_context()` (manager.py:103-126).  
  - `EvaluationEventsEmitter` copies `trace_id`/`span_id` onto events (evaluation.py:326-420).  
  - Deepeval evaluator itself does not manipulate span context; it consumes the invocation and returns `EvaluationResult` objects (deepeval.py throughout).

---
## Observed Gaps

1. **Log events lack explicit trace/span identifiers.** `SDKLogRecord` constructors avoid the `trace_id` / `span_id` parameters and do not attach a context, so downstream processors cannot correlate content events back to the originating span (content_events.py:68-76; emitters/utils.py:319-540). Today this relies on SDK-side implicit context, which is not guaranteed once the span is closed or when emitters run outside the active context.
2. **No canonical `SpanContext` snapshot stored on GenAI objects.** Consumers fetch IDs from `invocation.span` even after the span is ended. Replacing the span emitter, disabling spans, or serializing invocations would break downstream sampling (handler.py:257-338; manager.py:103-126).
3. **Metrics correlation depends on a live span object.** When span emission is disabled (`enable_span=false`), `invocation.span` stays `None`, causing metrics to record without context and evaluation sampling to fall back to the "no trace id" branch (metrics.py:214-312; manager.py:103-126).
4. **Evaluation Manager lacks guard rails for missing spans.** `invocation.span.get_span_context()` is called without a `None` check; if spans are disabled or an alternate emitter replaces the span, this raises (manager.py:103-110).
5. **Auxiliary log helpers (workflow/agent/step)** share the same missing-context issue (emitters/utils.py:498-740). When re-enabled, these would emit logs that also lack trace correlation.

---
## Desired Design

1. **Persist span identity independently of the span object.**  
   - Extend `GenAI` (types.py) with immutable fields such as `trace_id_hex`, `span_id_hex`, and/or a lightweight `SpanContextSnapshot`.  
   - `SpanEmitter.on_start` captures the span context immediately and stores it on the invocation before any downstream emitters run.  
   - Handlers and evaluators read the snapshot, not the mutable span object, so correlation survives span shutdown, serialization, or span-less configurations.

2. **Guarantee trace/span IDs on all telemetry surfaces.**  
   - Update `_llm_invocation_to_log_record` and related helpers to accept the stored context and pass `trace_id`, `span_id`, and `trace_flags` into `SDKLogRecord` (emitters/utils.py).  
   - Ensure `ContentEventsEmitter` injects the snapshot when emitting logs (content_events.py).  
   - For metrics, prefer passing `context=trace.set_span_in_context(span)` when a span exists; otherwise inject the stored IDs via metric attributes to keep a correlation hook.

3. **Make evaluation resilient to span availability.**  
   - Teach `Manager.on_completion` to fall back to the stored context snapshot or to skip sampling gracefully when spans are disabled.  
   - Propagate the snapshot into `EvaluationEventsEmitter` so events still include IDs even if the span object is missing.

4. **Centralise context access helpers.**  
   - Provide a utility (e.g., `get_trace_span_ids(invocation)`) that returns `(trace_id_hex, span_id_hex, trace_flags)` using the snapshot and optional live span.  
   - Migrate debug logging, metrics, logs, and evaluation emitters to this helper to avoid duplication and inconsistent fallbacks.

---
## Step Breakdown

1. **Model update**  
   - Add persistent context fields to `GenAI` / `LLMInvocation` and capture them in `SpanEmitter.on_start`.  
   - Populate the fields when spans are created externally (e.g., agent/workflow spans) and provide backwards-compatible defaults.

2. **Logging fixes**  
   - Modify `_llm_invocation_to_log_record`, `_workflow_to_log_record`, `_agent_to_log_record`, `_step_to_log_record`, `_embedding_to_log_record` (emitters/utils.py:319-740) to set `trace_id`, `span_id`, `trace_flags`, or `context`.  
   - Pass the snapshot from `ContentEventsEmitter.on_end` and any future workflow/agent emitters.
   - Add regression tests asserting that emitted OTLP logs include matching IDs.

3. **Metrics correlation**  
   - Update `_record_duration` / `_record_token_metrics` to accept the snapshot so they can fall back to attribute-based correlation when span objects are absent (emitters/utils.py:543-580).  
   - Verify histograms still record with context when spans are active.

4. **Evaluation hardening**  
   - Guard `Manager.on_completion` against missing spans by using the stored snapshot; log once when sampling falls back.  
   - Ensure `EvaluationEventsEmitter` always copies IDs from the snapshot, not directly from the span.

5. **Tooling & docs**  
   - Document the snapshot contract in handler docs / README.  
   - Extend existing tests (e.g., `tests/test_mixed_sequence.py`) to assert trace/span propagation across spans, metrics, logs, and evaluation events.

Delivering the above will make span correlation deterministic across spans, metrics, logs, and evaluation events, while tolerating optional span emission and custom emitter stacks.

