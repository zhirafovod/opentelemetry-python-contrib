# Traceloop -> GenAI Semantic Convention Translator Emitter

This optional emitter promotes legacy `traceloop.*` attributes attached to an `LLMInvocation` into
Semantic Convention (or forward-looking custom `gen_ai.*`) attributes **before** the standard
Semantic Convention span emitter runs. It does **not** create its own span.

## Why Use It?
If you have upstream code (or the Traceloop compat emitter) producing `traceloop.*` keys but you
want downstream dashboards/tools to rely on GenAI semantic conventions, enabling this translator
lets you transition without rewriting upstream code immediately.

## What It Does
At `on_start` of an `LLMInvocation` it scans `invocation.attributes` for keys beginning with
`traceloop.` and (non-destructively) adds corresponding keys:

| Traceloop Key (prefixed or raw) | Added Key                 | Notes |
|---------------------------------|---------------------------|-------|
| `traceloop.workflow.name` / `workflow.name` | `gen_ai.workflow.name`  | Custom (not yet in spec) |
| `traceloop.entity.name` / `entity.name`     | `gen_ai.agent.name`     | Approximates entity as agent name |
| `traceloop.entity.path` / `entity.path`     | `gen_ai.workflow.path`  | Custom placeholder |
| `traceloop.callback.name` / `callback.name` | `gen_ai.callback.name`  | Also sets `gen_ai.operation.source` if absent |
| `traceloop.callback.id` / `callback.id`     | `gen_ai.callback.id`    | Custom |
| `traceloop.entity.input` / `entity.input`   | `gen_ai.input.messages` | Serialized form already present |
| `traceloop.entity.output` / `entity.output` | `gen_ai.output.messages`| Serialized form already present |

Existing `gen_ai.*` keys are never overwritten.

## Enabling
Fast path (no entry point needed):

```bash
export OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR=1
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_compat

Optional (remove original traceloop.* after promotion):
export OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY=1
```

The flag auto-prepends the translator before the semantic span emitter. You can still add
`traceloop_translator` explicitly once an entry point is created.

You can also load this emitter the same way as other extra emitters. There are two common patterns:

### 1. Via `OTEL_INSTRUMENTATION_GENAI_EMITTERS` with an extra token
If your emitter loading logic supports extra entry-point based names directly (depending on branch state), add the translator token (e.g. `traceloop_translator`). Example:

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_translator,traceloop_compat
```

Ordering is important: we request placement `before=semconv_span` in the spec, but if your environment override reorders span emitters you can enforce explicitly (see next section).

### 2. Using Category Override Environment Variable
If your build supports category overrides (as implemented in `configuration.py`), you can prepend:

```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_compat
export OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN=prepend:TraceloopTranslator
```

The override ensures the translator emitter runs before the semantic span emitter regardless of default resolution order.

## Example
Minimal Python snippet (assuming emitters are loaded via entry points and the translator is installed):

```python
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text

inv = LLMInvocation(
    request_model="gpt-4",
    input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
    attributes={
        "traceloop.entity.name": "ChatLLM",
        "traceloop.workflow.name": "user_flow",
        "traceloop.callback.name": "root_chain",
        "traceloop.entity.input": "[{'role':'user','content':'Hello'}]",
    },
)
handler = get_telemetry_handler()
handler.start_llm(inv)
inv.output_messages = [OutputMessage(role="assistant", parts=[Text("Hi")], finish_reason="stop")]
handler.stop_llm(inv)
# Result: final semantic span contains gen_ai.agent.name, gen_ai.workflow.name, gen_ai.input.messages, etc.
```

## Non-Goals
- It does not remove or rename original `traceloop.*` attributes (no destructive behavior yet).
- It does not attempt deep semantic inference; mappings are intentionally conservative.
- It does not serialize messages itself—relies on upstream emitters to have placed serialized content already.
