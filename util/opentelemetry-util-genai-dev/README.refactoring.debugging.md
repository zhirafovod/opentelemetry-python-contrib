# GenAI Debug Logging Refactoring Plan

This document tracks the introduction of a structured, opt-in debug logging facility
for GenAI telemetry types (e.g. `LLMInvocation`, `AgentInvocation`, `ToolCall`, etc.)
across the development & evaluation/emitters packages:

Target packages:

1. `util/opentelemetry-util-genai-dev`
2. `util/opentelemetry-util-genai-evals-deepeval`
3. `util/opentelemetry-util-genai-emitters-splunk`

## Goals

- Provide traceable, low-noise DEBUG logs showing which span context (trace_id, span_id)
  is associated with each GenAI invocation object as it flows through handler, evaluator,
  and emitter components.
- Make logging opt-in via environment variable (no output unless explicitly enabled).
- Keep the logging footprint lightweight (no expensive serialization unless debug is on).
- Offer a consistent representation string for all GenAI types to simplify troubleshooting.
- Explicitly log all points where a new span (and thus new trace_id/span_id) is created.

## Environment Variable Contract

Primary flag: `OTEL_GENAI_DEBUG`

Accepted truthy values (case-insensitive): `1`, `true`, `yes`, `on`, `debug`.

Fallback alias also recognized for future flexibility: `OTEL_INSTRUMENTATION_GENAI_DEBUG`.

If unset or set to any other value: debug logging remains disabled.

## Public (Developer-Facing) Debug API

Implemented in `opentelemetry.util.genai.debug` (new module) within `genai-dev` package.

Functions:

- `is_enabled() -> bool` — returns if debug logging is active.
- `genai_debug_log(event: str, obj: GenAI | None = None, **info)` — conditional debug logger.
- `summarize_genai(obj: GenAI | None) -> str` — returns concise string representation.

Representation includes (when available):

`<ClassName run_id=... parent_run_id=... model=... provider=... trace_id=xxxxxxxx span_id=yyyyyyyy>`

Trace/span IDs are rendered as hex if valid; omitted if no span or invalid context.

## Insertion Points (Initial Scope)

1. `TelemetryHandler.start_llm()` — log BEFORE span creation and AFTER with new IDs.
2. `TelemetryHandler.stop_llm()` — log with final span context and duration.
3. `TelemetryHandler.fail_llm()` — log error and span context.
4. Deepeval evaluator `DeepevalEvaluator._evaluate_generic()` — log evaluation start (type, metrics, span IDs).
5. Splunk emitter `SplunkConversationEventsEmitter.on_end()` — log event emission attempt and span IDs.
6. Splunk evaluation results emitter `_emit_event()` — log aggregation emission (count, span IDs).
7. Metric emitters (optional later) — candidate for logging derived metric values.

## Logging Format

Single-line, structured key=value format for easy grep:

`GENAIDEBUG event=handler.start_llm.begin class=LLMInvocation trace_id=... span_id=... run_id=... model=... provider=... parent_run_id=... input_tokens=... output_tokens=...`

Only include keys that are non-empty. Additional contextual keys can be passed via `**info`.

## Edge Cases & Considerations

- Missing span: trace/span keys omitted.
- Invalid span context (`is_valid` false): a marker `trace_valid=false` can be added.
- Multiple packages declare `opentelemetry.util.genai` namespace. To avoid import errors,
  each usage site wraps import in a try/except providing a no-op fallback.
- Performance: summarization avoids iterating over large message bodies; content is not printed.
- Thread safety: global logger is standard Python logging; formatting is stateless.

## Future Improvements (Not in Initial Scope)

- Structured logging adapter emitting JSON lines.
- Sampling (e.g., `OTEL_GENAI_DEBUG_SAMPLE_RATE`).
- Integration with OpenTelemetry Logs pipeline (currently plain Python logger only).
- Rich diff of attribute changes between start/stop phases.

## Tasks & Status

| ID | Task | Status |
|----|------|--------|
| 1 | Create debug module with helpers | DONE |
| 2 | Add imports & calls in handler methods | DONE |
| 3 | Add evaluator logging | DONE |
| 4 | Add Splunk emitters logging | DONE |
| 5 | Document environment variable in `environment_variables.py` (optional) | PENDING |
| 6 | Basic tests (enable flag + sample invocation) | PENDING |
| 7 | Update this changelog section | DONE |

## Minimal Test Plan

1. Set `OTEL_GENAI_DEBUG=1` and run example LangChain manual script — inspect console output.
2. Ensure no debug output when variable unset.
3. Verify hex formatting matches `trace.get_current_span().get_span_context()` output.
4. Confirm evaluator & emitters logs appear after handler start.

## Changelog (append entries as implemented)

```
2025-10-15 Added debug module (`debug.py`) and instrumentation in handler, Deepeval evaluator, Splunk emitters.
```

## Usage Example

```bash
export OTEL_GENAI_DEBUG=1
python examples/manual/main.py
```

Sample output excerpt (illustrative):

```
GENAIDEBUG event=handler.start_llm.begin class=LLMInvocation run_id=0f4... model=gpt-4 provider=openai
GENAIDEBUG event=handler.start_llm.span_created class=LLMInvocation trace_id=9d0c... span_id=7a3b... run_id=0f4...
GENAIDEBUG event=evaluator.deepeval.start class=LLMInvocation metrics=bias,toxicity trace_id=9d0c... span_id=7a3b...
GENAIDEBUG event=emitter.splunk.conversation.on_end class=LLMInvocation output_messages=2 trace_id=9d0c... span_id=7a3b...
```

---
Maintained by automated AI coder; update tasks & changelog entries upon each patch.
