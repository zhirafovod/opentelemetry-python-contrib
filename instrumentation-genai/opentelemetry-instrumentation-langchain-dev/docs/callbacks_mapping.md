# OpenTelemetry LangChain Instrumentation – Callback → GenAI Type Mapping (Concise)

Scope:
`instrumentation-genai/opentelemetry-instrumentation-langchain-dev` callback handler → GenAI utility types in `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py`.

Purpose:
Document how LangChain callbacks translate runtime events into GenAI telemetry entities (Workflow, AgentInvocation, Task, LLMInvocation) and what telemetry (spans, metrics, content events, evaluation) each produces.

---

## Core GenAI Types Referenced

```text
Workflow          # top-level orchestration (multi-step / multi-agent)
AgentInvocation   # agent creation or invocation lifecycle
Task              # chain step or tool use (task_type = chain | tool_use)
LLMInvocation     # model call (chat/generate) with request/response semantics
```
Auxiliary fields emitted only if present: tool/functions (request_functions), message content (input_messages/output_messages), token usage (input_tokens/output_tokens), stop sequences, temperature/top_p/top_k penalties, choice count, service tier, finish reasons.

---

## Callback Classification Logic (Start Phase)

```text
on_chain_start(serialized, inputs, run_id, parent_run_id, metadata, tags):
    name = heuristic(serialized, kwargs)
    if is_agent_run(serialized, metadata, tags): → AgentInvocation
    else if parent_entity == None:            → Workflow
    else:                                     → Task(task_type = "chain", parent = parent_entity)

on_tool_start(serialized, input_str|inputs, run_id, parent_run_id,...):
    → Task(task_type = "tool_use") parent = entity(parent_run_id)

on_chat_model_start(serialized, messages, run_id, parent_run_id,...):
    → LLMInvocation(operation = "chat") with structured request_* attrs

on_llm_start(serialized, prompts, run_id,...):
    internally builds messages from prompts → LLMInvocation then sets operation = "generate_text"
```

### Agent Detection Heuristics (is_agent_run)
Checks (case-insensitive):
- metadata keys: `ls_span_kind|ls_run_kind|ls_entity_kind|run_type|ls_type` containing "agent"
- flags: `ls_is_agent|is_agent` boolean true or string in {"true","1","agent"}
- tags list items containing substring "agent"
- serialized name or id (string/list) containing "agent"
Result: first positive → classify as AgentInvocation.

### Parent Resolution & Implicit Stack
- Explicit `parent_run_id` looked up in `_entities`.
- If absent (None) and starting Agent/Workflow, entity run_id is pushed on context stack `genai_active_entity_stack`.
- LLMInvocation without explicit parent attempts implicit stack parent propagation (agent/workflow ids for cross-linking).

---

## Callback → GenAI Type & Telemetry Table

| Callback | GenAI Type Created | Operation / task_type (set or mutated) | Start Phase Telemetry Actions | End Phase Telemetry / Evaluation |
|----------|--------------------|----------------------------------------|-------------------------------|----------------------------------|
| `on_chain_start` | `Workflow` (no parent & not agent) / `AgentCreation` or `AgentInvocation` (agent heuristics) / `Task` (otherwise) | Workflow: — / Agent: `operation=create_agent` or `invoke_agent` / Task: `task_type=chain` | Instantiate entity; propagate parent ids; start span; preallocate metrics; serialize `inputs` (workflow.initial_input / task.input_data / agent.input_context); push to context stack (workflow/agent) | Workflow: `final_output` stored then span stop; Agent: `output_result`; Task: `output_data`; duration metrics recorded |
| `on_chain_end` | (same as started) | — | (No new start actions) | Store outputs in corresponding output field; stop span; record duration; no evaluation |
| `on_tool_start` | `Task` | `task_type=tool_use` | Start span; preallocate metrics; serialize tool input(s) into `input_data`; parent resolution | On `on_tool_end` stores `output_data`, stops span |
| `on_tool_end` | `Task` | — | — | Serialize `output` -> `output_data`; stop span; record duration |
| `on_chat_model_start` | `LLMInvocation` | `operation=chat` | Build input messages; collect request_* attributes (temperature, top_p, etc.); start span; metrics prealloc; events or prompt capture | Completed on `on_llm_end` |
| `on_llm_start` | `LLMInvocation` (same instance) | Mutates `operation=generate_text` | Reuses chat model start logic; no additional telemetry beyond operation override | Completed on `on_llm_end` |
| `on_llm_end` | `LLMInvocation` | — | — | Build `output_messages`; set `response_model_name`, `response_id`, token usage; finish reasons; emit events or prompt capture; stop span; record duration & token metrics; trigger evaluation (`evaluate_llm`) |
| `on_*_error` (llm/chain/tool/agent/retriever) | Active entity | — | — | Populate `error_message` & `error_type`; invoke fail_*; clear LLM `output_messages`; mark span error; unregister entity |

Notes:

- "metrics prealloc" indicates duration/token instruments prepared; actual values observed at stop.
- Evaluation only occurs for `LLMInvocation` after successful or failed completion (`on_llm_end` / error handlers).
- Parent propagation: workflow/agent ids cascaded to children (LLMInvocation/task) in `_start_entity`.

Metrics:

- Duration histogram: recorded on end (latency per entity).
- Token histogram: recorded when `input_tokens` / `output_tokens` available (LLMInvocation).

Content Events vs Prompt Capture:

- If `should_emit_events()` true: emits `MessageEvent` / `ChoiceEvent` (LLM start/end) via `content_events` emitter.
- Else if `should_send_prompts()` true: captures serialized payload into `entity.attributes['prompt_capture']` (`inputs` / `outputs` / `error`).

Evaluation:

- Triggered only for `LLMInvocation` after `on_llm_end` (`evaluate_llm(invocation)`). Evaluation results then flow through GenAI handler to evaluation emitters.

---

## Attribute Harvesting Sources (Functions & Outputs)

| Source / Data | Callback Handler Function(s) | Extraction Logic Summary | Resulting Fields / Placement |
|---------------|------------------------------|--------------------------|------------------------------|
| Raw callback `metadata` | `_sanitize_metadata_dict`, `_collect_attributes` | Removes nulls; coerces complex types to strings; splits out `ls_` prefixed legacy keys into `langchain_legacy`; filters reserved hyperparameter keys after normalization | `entity.attributes` (with nested `langchain_legacy`) |
| Invocation params (`invocation_params`) for LLM | `on_chat_model_start` local helpers `_pop_float/_pop_int/_pop_stop_sequences`, `_extract_request_functions` | Pop known hyperparameters; convert numeric types; stop sequences normalized to list[str]; remove duplicates to avoid double counting | LLMInvocation fields: `request_temperature`, `request_top_p`, `request_top_k`, `request_frequency_penalty`, `request_presence_penalty`, `request_seed`, `request_max_tokens`, `request_choice_count`, `request_stop_sequences`, `request_functions` |
| Model & provider names | `on_chat_model_start` (raw `serialized` + metadata) | Fallback order: invocation_params.model_name → metadata.ls_model_name/model_name → serialized.name → "unknown-model"; provider resolved from ls_provider/provider keys | `LLMInvocation.request_model`, `LLMInvocation.provider` |
| Agent classification | `_is_agent_run` (used by `on_chain_start`) | Heuristic over metadata keys, flags, tags, serialized name/id content | Determines creation of `AgentInvocation` vs `Workflow`/`Task`; sets `operation` create vs invoke |
| Input messages (chat / prompts) | `_build_input_messages` (chat), `on_llm_start` (prompt wrapping) | Each incoming prompt or `BaseMessage` converted to `InputMessage(role, parts=[Text])` | `LLMInvocation.input_messages` |
| Output messages | `on_llm_end` | First generation's message content + finish_reason captured; tool/function calls not yet expanded into `ToolCall` dataclass here | `LLMInvocation.output_messages`, `response_finish_reasons` (implicit list) |
| Finish reason | `on_llm_end` | Extracted from `generation.generation_info.finish_reason` fallback "stop" | `OutputMessage.finish_reason`; aggregated list to `LLMInvocation.response_finish_reasons` |
| Token usage | `on_llm_end` | Reads `llm_output.usage.prompt_tokens` / `completion_tokens` or `token_usage` fallback | `LLMInvocation.input_tokens`, `LLMInvocation.output_tokens`; feeds token metrics |
| Tool definitions for function calling | `_extract_request_functions` (invocation params) | Filters each tool.function dict for allowed keys (name, description, parameters) | `LLMInvocation.request_functions` (semantic emission) |
| Truncation & original lengths | `_maybe_truncate`, `_store_serialized_payload`, `_record_payload_length` | Serializes JSON; if > 8KB replaces with `<truncated:N bytes>` and records original length mapping | Stored fields: `workflow.initial_input`, `workflow.final_output`, `agent.input_context`, `agent.output_result`, `task.input_data`, `task.output_data`, plus `entity.attributes.orig_length[field]` |
| Prompt capture (non-event mode) | `_capture_prompt_data` invoked in start/end callbacks conditional on `should_send_prompts()` | Stores serialized inputs/outputs/error under `entity.attributes['prompt_capture']` keyed by phase | `entity.attributes.prompt_capture.inputs|outputs|error` |
| Parent identity propagation | `_start_entity`, `_resolve_parent`, `_find_agent` | Reads explicit `parent_run_id` else context stack; copies agent/workflow ids into LLMInvocation | `LLMInvocation.agent_name`, `LLMInvocation.agent_id`, `LLMInvocation.workflow_id` |
| Error details | `_handle_error` → `_fail_entity` | Captures exception type/message; clears LLM outputs; sets type-specific output fields to error string | `entity.attributes.error_message`, `entity.attributes.error_type` + output field mutation |

Semantic Convention Emission:

Dataclass fields decorated with `metadata['semconv']` (e.g. `request_model`, `response_model_name`, `agent_name`, `agent_id`, `operation`, hyperparameters, token usage) are collected via emitter logic (span emitter) using `entity.semantic_convention_attributes()`; non-semconv harvested data remains in `entity.attributes` for vendor or legacy enrichment.

---

## Telemetry Produced Per GenAI Type

```text
Workflow:
  Span: workflow identity (name, workflow_type, description, framework)
  Metrics: duration
  Events/Content: prompt_capture initial_input/final_output (if enabled)
AgentInvocation:
  Span: agent name/id, operation(create|invoke), description, model, tools[]
  Metrics: duration
  Events/Content: input_context/output_result capture
Task (chain/tool_use):
  Span: task name, task_type, source(workflow|agent), assigned_agent, objective
  Metrics: duration
  Events/Content: input_data/output_data capture
LLMInvocation:
  Span: request_model, provider, operation(chat|generate_text), hyperparameters, response_model/id, token usage, stop sequences, finish reasons, functions
  Metrics: duration, token histograms
  Events: MessageEvent(s) & ChoiceEvent(s) OR prompt_capture of inputs/outputs
  Evaluation: post-stop evaluation results emitted (metrics/spans/logs depending on emitters)
```

---

## Error Handling Flow

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
End of concise callback → GenAI type mapping.
