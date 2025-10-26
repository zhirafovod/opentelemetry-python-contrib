# OpenTelemetry LangChain Instrumentation – Callback → GenAI Type Mapping (Concise)

Scope:
`instrumentation-genai/opentelemetry-instrumentation-langchain-dev` callback handler → GenAI utility types in `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py`.

Purpose:
Document how LangChain callbacks translate runtime events into GenAI telemetry entities (Workflow, AgentInvocation, Task, LLMInvocation) and what telemetry (spans, metrics, content events, evaluation) each produces.

---

## Desired Unified Trace Model (Refinement)

Problem Observed:

Current runs (e.g. `examples/manual/t.py`) produce multiple distinct `trace_id` values – one per GenAI entity span (Agent, LLM, Tool, etc.). This breaks end‑to‑end correlation for a single agent workflow execution.

Desired Behavior:

All spans representing one logical agent execution (agent orchestration + internal model/tool calls) MUST share the same `trace_id` while preserving proper parent/child relationships:

```text
AgentInvocation (root span)  <-- trace root (no parent span id)
    ├─ Task(model request) parent = AgentInvocation
    │    └─ LLMInvocation parent = Task
    ├─ Task(tool_use orchestration) parent = AgentInvocation
    │    ├─ Tool(tool_use) parent = Task
    │    └─ Task(tools_to_model follow-up) parent = Task
    └─ Task(model request) parent = AgentInvocation
             └─ LLMInvocation parent = Task
```

Note: Earlier interim attempt introduced a synthetic Workflow root span. This has been removed; the canonical root for a LangChain agent run is the first `AgentInvocation` span. Synthetic workflow creation risked early termination and missing root representation in exported traces.

Observed Current (Multi-Trace) Example:

```text
Trace ID: 2cb4ae9529785433d3c8c06569d7c7fc
└── Span ID: 006e80bf6418cd80 (Parent: none) - invoke_agent weather-agent

Trace ID: 47b70102abf2ab885bb9e02daea90b50
└── Span ID: e77c3ab7c30b1089 (Parent: none) - invoke_agent model_to_tools

Trace ID: 8fa43ffc52b5778c3a7690c006e97cc8
└── Span ID: 531f7e98193c6cf2 (Parent: none) - chat gpt-5-mini

Trace ID: b8a23fba43bec2cfa69c0a87d745f292
└── Span ID: e47574f829107acc (Parent: none) - invoke_agent model
    ├── Span ID: 950170aa32eb2783 (Parent: e47574f829107acc) - gen_ai.task get_weather
    ├── Span ID: 813e766d6d7a13d1 (Parent: e47574f829107acc) - invoke_agent tools_to_model
    └── Span ID: 609fdc23fe29a097 (Parent: e47574f829107acc) - invoke_agent tools
        ├── Span ID: 6656fa97870a7518 (Parent: 609fdc23fe29a097) - chat gpt-5-mini
        ├── Span ID: 037e0c9c75582042 (Parent: 609fdc23fe29a097) - invoke_agent model_to_tools
        └── Span ID: af700c43cde11470 (Parent: 609fdc23fe29a097) - invoke_agent model
```

Problem: A single logical run is fragmented across 4 different trace IDs. Only one sub-tree (rooted at `invoke_agent model`) retains its descendants; earlier agent/model invocations ended before child spans started, severing trace continuity.

Key Points (Desired):

1. Single trace per agent run (root = first AgentInvocation span).
2. `parent_run_id` MUST be converted into actual span parenting (not only an attribute on the dataclass).
3. Tool and task spans retain same trace; parent chosen based on semantics:
   - If a tool call arises from a model tool request, parent should be that LLMInvocation span (keeps causal ordering).
   - If a tool call is orchestrated directly by agent code (outside model response), parent can be AgentInvocation.
4. Model responses that trigger subsequent model calls (e.g. tool_to_model follow‑up) form nested LLMInvocation spans parented to the task or original model span, depending on invocation context.
5. All other tasks (chain steps) parent = nearest active AgentInvocation or Workflow.

### Gaps Causing Multiple Trace IDs

| Gap | Effect | Implemented / Remaining |
|-----|--------|------------------------|
| SpanEmitter ignored `parent_run_id` | Orphan spans with new trace IDs | Parent span passed via `parent_span` attribute; emitter uses explicit context (DONE) |
| Parent agent span ended before children | Fragmented traces | Deferred stop via `pending_stop` + child counts (DONE; further tuning may be needed) |
| No run_id → span registry | Parenting impossible after context switch | Registry added in `TelemetryHandler` (DONE) |
| `parent_run_id` unused | Hierarchy lost | Callback handler now resolves and sets `parent_span` (DONE) |
| Synthetic workflow root | Missing or inconsistent root span | Removed; using `AgentInvocation` as root (DONE) |
| Orphan diagnostics absent | Hard to detect missed parenting | Attribute `orphan_parent_run_id` set when parent span missing (DONE) |

### Current Parenting Rules

1. First `AgentInvocation` without `parent_run_id` becomes trace root.
2. `Task` spans parent to nearest active `AgentInvocation` (or upstream workflow if present externally).
3. `LLMInvocation` parents to the task that initiated the model request.
4. Tool execution spans parent to their initiating task.
5. Follow‑up model/tool steps create new tasks under the same agent, preserving a single trace.

### Data Flow Adjustments

```python
# callback_handler._start_entity(entity):
parent_span = telemetry_handler.lookup_span(entity.parent_run_id or implicit)
parent_ctx = trace.set_span_in_context(parent_span) if parent_span else None
with tracer.start_as_current_span(name, context=parent_ctx, kind=SpanKind.CLIENT, end_on_exit=False) as span:
    entity.span = span
    telemetry_handler.register_span(entity.run_id, span)
```

TelemetryHandler additions:

```python
self._active_spans: dict[UUID, Span] = {}
def register_span(run_id: UUID, span: Span): self._active_spans[run_id] = span
def get_span(run_id: UUID) -> Span | None: return self._active_spans.get(run_id)
def unregister_span(run_id: UUID): self._active_spans.pop(run_id, None)
```

SpanEmitter `on_start`: resolve parent before creating new span:

```python
parent_ctx = None
if getattr(invocation, "parent_run_id", None):
    parent_span = handler.get_span(invocation.parent_run_id)
    if parent_span:
        parent_ctx = trace.set_span_in_context(parent_span)
cm = tracer.start_as_current_span(span_name, context=parent_ctx, kind=SpanKind.CLIENT, end_on_exit=False)
```

### Required Instrumentation Contract Updates

Add explicit guarantee that for any entity with a populated `parent_run_id`, the instrumentation will attempt to establish a parent span relationship. If parent span is ended prematurely, fallback is still a new trace root – but this should be treated as a correctness bug during development.

### Edge Cases & Policies

- Orphan child (parent span missing): mark attribute `gen_ai.parent.missing=true` for diagnostics.
- Concurrent tool/model calls: maintain thread‑safe span map (Lock already exists in callback handler; replicate in handler or reuse).
- Reentrancy (nested agent invokes another agent): treat inner agent as child span (set `parent_run_id` accordingly) → allowed for multi‑agent graphs.

### Metrics & Evaluation Impact

- Shared trace_id enables holistic latency breakdown and token attribution across the agent execution.
- Evaluation spans (LLMInvocation) inherit agent context for unified trace analytics.

### Action Summary (see `callback_handler.refactoring.md` for detailed tasks)

1. Create span registry and parent resolution in TelemetryHandler.
2. Modify SpanEmitter to honor explicit parent context.
3. Delay agent/workflow span termination until all descendants complete (reference count).
4. Update callback handler to increment/decrement descendant counters.
5. Add diagnostic attributes for orphaning and flattening choices.

### Implemented Interim Fix

As of branch `genai-utils-e2e-dev-langchain-demos`, agent/workflow spans now defer stopping using a `pending_stop` flag plus `_child_counts` map inside `LangchainCallbackHandler`. This keeps the root agent span open until all child LLM/tool/task spans complete, enabling a unified trace ID for sequential operations in a single run. Further improvements (explicit parent context, span registry in TelemetryHandler, flattening options) remain planned.

---
 
## Minimal Diff Guidance (for future code PR)

1. Add `TelemetryHandler.get_span(run_id)` / `register_span` / `unregister_span`.
2. Pass handler reference into SpanEmitter or provide global accessor.
3. Extend entity lifecycle start functions to call `register_span` after span creation.
4. On end/fail, call `unregister_span` after `span.end()` but only if no active children (agent/workflow). Maintain `child_count` attribute on entities.
5. Unit tests: assert single trace_id across sequence of spans in sample workflow, nested tool calls, errors.

---
 
## Updated Callback Intent Table (Hierarchy‑Aware)

| Callback | Span Parent (new rule) |
|----------|------------------------|
| on_chain_start (AgentInvocation root) | None (root) |
| on_chain_start (Workflow root)       | None (root) unless inside outer workflow |
| on_chat_model_start / on_llm_start   | AgentInvocation or Workflow span |
| on_tool_start (model requested)      | Preceding LLMInvocation span |
| on_tool_start (agent orchestrated)   | AgentInvocation span |
| on_llm_start after tool              | Tool Task span |
| on_chain_start (nested task)         | Nearest active AgentInvocation/Workflow span |

---
 
## Testing Strategy (Outline)

1. Execute `examples/manual/t.py` → assert all collected spans share identical trace_id.
2. Simulate tool function call chain with follow‑up model call → assert parent chain depth increments.
3. Error path: raise tool exception → ensure child has same trace_id; parent flagged error; subsequent siblings still share trace.
4. Concurrency: parallel tool tasks under agent → each child trace_id equals agent trace_id.

---
 
## Observability Diagnostics

Add optional debug logs:

```python
genai_debug_log("parent_resolution", child_run=child.run_id, parent_run=parent_run_id, found=bool(parent_span))
```

Add attribute on child span if parent missing:
`gen_ai.parent.missing = true`

---
 
## Non-Goals

- Cross-trace linking outside single agent execution.
- Retroactive re-parenting after span end (would require span processor mutation).
- Full graph visualization (can be built later using run_id relationships).

---
 
## Summary

Implement parent span resolution + lifecycle coordination so all GenAI spans for one agent execution form a single trace tree. This enables holistic analysis (latency, token usage, evaluations) and aligns with desired semantic model where `AgentInvocation` is the orchestrator root.
 
```text
on_*_error(error):
  entity.attributes.error_message = str(error)
  fail_* method called (type-specific)
  For LLMInvocation: output_messages cleared
  Span status marked error (in handler emitters)
  Prompt capture of error (if enabled)
```

---

 
## Edge Cases & Notes

- Missing `parent_run_id` in `on_chain_start`: first non-agent run becomes Workflow.
- Implicit parent stack ensures nested LLM calls inherit agent/workflow identity even if explicit parent omitted.
- `on_llm_start` routes to `on_chat_model_start` for unified handling, then overrides `operation` to `generate_text`.
- Truncation threshold: 8192 bytes UTF-8; original length recorded for forensic analysis.
- `request_functions` omission: empty list if no structured tool definitions.
- Token usage absent: histograms not updated for that invocation.

---

## Minimal Pseudocode Summary

```python
if callback == chain_start:
    entity = classify_chain(serialized, metadata, tags, parent_run_id)
elif callback == tool_start:
    entity = build_task(tool_use)
elif callback in {chat_model_start, llm_start}:
    entity = build_llm_invocation(messages/prompts)
start_entity(entity)
...
if callback == llm_end:
    enrich_llm(invocation, generations, llm_output)
    stop_entity(invocation)
    evaluate(invocation)
else if callback ends *_start entity:
    attach_outputs(entity)
    stop_entity(entity)
```

---

