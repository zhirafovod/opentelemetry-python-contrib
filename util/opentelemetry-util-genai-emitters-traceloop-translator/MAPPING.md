# Traceloop → GenAI Semantic Conventions Mapping

This document explains how the `TraceloopTranslatorEmitter` converts Traceloop span attributes into `gen_ai.*` semantic convention or forward-looking extension attributes.

## Goals

- Bridge existing Traceloop telemetry to emerging OpenTelemetry GenAI semantic conventions.
- Preserve provenance (workflow, prompt registry metadata) without overwriting user-provided `gen_ai.*` attributes.
- Provide safe handling of sensitive content (prompt templates, messages) via explicit opt-in.
- Offer heuristics to populate `gen_ai.operation.name` when feasible (tool/workflow cases).

## Environment Flags

| Variable | Default | Effect |
|----------|---------|--------|
| `OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY` | `true` | Remove original `traceloop.*` keys after mapping (except `traceloop.span.kind`). Set to `false` to keep originals. |
| `OTEL_GENAI_CONTENT_CAPTURE` | `0` | When set to non-zero/non-false, enables mapping of content attributes (`entity.input/output`, prompt template & variables). |
| `OTEL_GENAI_MAP_CORRELATION_TO_CONVERSATION` | `true` | Controls mapping `traceloop.correlation.id` → `gen_ai.conversation.id`. Disable to skip this mapping. |
| (internally enforced) prompt template max | 4096 | Internal truncation threshold; no env var required. |

## Mapping Categories

### Workflow / Entity (Hierarchy Extensions)
| Traceloop | GenAI (Extension) | Notes |
|-----------|-------------------|-------|
| `traceloop.workflow.name` | `gen_ai.workflow.name` | No official spec field yet; |
| `traceloop.entity.name` | `gen_ai.agent.name` | Treat entity as agent component. |
| `traceloop.entity.path` | `gen_ai.workflow.path` | No official spec field yet;  |
| `traceloop.entity.version` | `gen_ai.workflow.version` | No official spec field yet; |
| `traceloop.association.properties` | `gen_ai.association.properties` | Free-form association metadata. |

### Content (Spec-Aligned)
| Traceloop | GenAI Spec |
|-----------|------------|
| `traceloop.entity.input` | `gen_ai.input.messages` |
| `traceloop.entity.output` | `gen_ai.output.messages` |

Only mapped when `OTEL_GENAI_CONTENT_CAPTURE` is enabled.

### Prompt Registry / Template Extensions
| Traceloop | GenAI Extension |
|-----------|-----------------|
| `traceloop.prompt.managed` | `gen_ai.prompt.managed` |
| `traceloop.prompt.key` | `gen_ai.prompt.key` |
| `traceloop.prompt.version` | `gen_ai.prompt.version` |
| `traceloop.prompt.version_name` | `gen_ai.prompt.version_name` |
| `traceloop.prompt.version_hash` | `gen_ai.prompt.version_hash` |
| `traceloop.prompt.template` | `gen_ai.prompt.template` (opt-in + truncation) |
| `traceloop.prompt.template_variables` | `gen_ai.prompt.template_variables` (opt-in) |



### Correlation → Conversation (Conditional)
`traceloop.correlation.id` → `gen_ai.conversation.id` (only if enabled and value matches regex `^[A-Za-z0-9._\-]{1,128}$`).

### Operation / Tool Heuristics
If `gen_ai.operation.name` is not already set:
- When `traceloop.span.kind == "tool"` and callback name available → set `gen_ai.operation.name = "execute_tool"` and `gen_ai.tool.name`.
- When `traceloop.span.kind` in `{workflow, agent, chain}` → set `gen_ai.operation.name = "invoke_agent"`.

### Unmapped Legacy
`traceloop.span.kind` is preserved even when stripping for diagnostic classification until official agent/workflow span conventions mature.

## Non-Overwriting Behavior
`attrs.setdefault(target, value)` ensures existing `gen_ai.*` attributes provided by upstream instrumentation or user code are never overwritten.

## Sensitive Content Handling
Prompt template and messages are potentially sensitive:
- Templates longer than 4096 characters are truncated with suffix `…(truncated)` (internal constant, not configurable via env).
- Future: could add `gen_ai.prompt.template.length` or external storage references; current design keeps minimal footprint.

## Audit / Version Tag
Translator adds `gen_ai.mapping.version = "traceloop_translator/1.0"` for trace auditing, without overwriting existing value.

## Example Minimal Usage
```python
from opentelemetry.util.genai.emitters.traceloop_translator import traceloop_translator_emitters

# Register emitters before invoking instrumentation workflow
emitters = traceloop_translator_emitters()
# Add to your TelemetryHandler / pipeline according to existing emitter factory integration.
```

Set environment (bash/zsh):
```bash
export OTEL_GENAI_CONTENT_CAPTURE=1
export OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY=true
```

## Future Adjustments
- Replace custom `gen_ai.workflow.*` / `gen_ai.agent.name` with official agent/workflow span attributes once stabilized.
- Extend heuristics to derive `gen_ai.tool.type` if metadata available.
- Support external content storage hooks and record reference attributes (e.g., `gen_ai.input.messages.ref`).


---
For questions or proposing spec-aligned attribute replacements, refer to: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
