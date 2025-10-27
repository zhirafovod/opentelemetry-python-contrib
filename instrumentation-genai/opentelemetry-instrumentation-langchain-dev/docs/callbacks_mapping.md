# OpenTelemetry LangChain Instrumentation – Callback → GenAI Type Mapping

This document summarizes how the LangChain callback handler in `instrumentation-genai/opentelemetry-instrumentation-langchain-dev` maps runtime events to the GenAI telemetry types defined in `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py`. The content reflects the Phase 1 handler already present on this branch.

## Callback → Entity Matrix

| Callback | GenAI entity | Notes |
|----------|--------------|-------|
| `on_chain_start` (no parent) | `AgentInvocation` when `tags`/`metadata` resolve an `agent_name` (e.g. tag `agent:foo`, `metadata.agent_name`); otherwise `Workflow`. | Sets `framework="langchain"`, stores serialized inputs (`input_context` or `initial_input`), adopts `metadata.agent_type`, `metadata.model_name`, `metadata.system` when present. |
| `on_chain_start` (has parent) | `AgentInvocation` when `tags`/`metadata` introduce a *new* `agent_name` (different to the nearest ancestor agent); else `ToolCall` when tool hints are present; otherwise `Task` with `task_type="chain"`. | Agent promotion keeps orchestration spans aligned with agent boundaries. Tool paths reuse an existing `ToolCall` (if `on_tool_start` fired first), refresh `arguments`, and stash JSON in `attributes["tool.arguments"]`. Non-tool paths create a `Task` and serialize inputs into `task.input_data`. |
| `on_chat_model_start` / `on_llm_start` | `LLMInvocation`. | Prompts/messages become `InputMessage` parts, model name resolved from serialized payload → metadata → kwargs; agent context propagated from parent entity; `on_llm_start` simply calls the chat handler and overwrites `operation="generate_text"`. |
| `on_llm_end` | — | Fills `output_messages`, extracts token counts from `response.llm_output.usage`, then stops the invocation. |
| `on_tool_start` | `ToolCall`. | Creates (or updates) a `ToolCall` with normalized name/id, serializes `inputs`/`input_str`, records agent context when parent is an `AgentInvocation`, sets `attributes["tool.arguments"]`. |
| `on_tool_end` | — | Serializes tool output into `attributes["tool.response"]` and stops the call. |
| `on_chain_end` | — | Resolves the registered entity for `run_id` (`Workflow`, `AgentInvocation`, `Task`, or `ToolCall`), copies serialized outputs to the matching field, then stops it. |
| Any `*_error` | — | Invokes `fail_by_run_id(run_id, GenAIError(message=str(error), type=type(error)))`; the util layer handles span status and cleanup. |

Common behaviours:

- Tags are preserved on every entity via `attributes["tags"]`.
- `_serialize` prefers JSON (UTF-8 preserved) with `str()` fallback; un-serializable payloads become `None`.
- Agent-aware parenting walks up to the nearest `AgentInvocation`, ensuring `Task`, `ToolCall`, and `LLMInvocation` entities inherit `agent_name` / `agent_id`. If callbacks supply a different `agent_name`, the handler promotes that chain to a new `AgentInvocation` span before continuing.


## Sample Trace (`examples/manual/t.py`)

```text
Trace ID: 879de17ad8b2ce0565d081fcceea5238
└── Span ID: eed435b2128ed02a  - invoke_agent weather-agent [op:invoke_agent]
    ├── Span ID: ba3fca28e7b484da  - gen_ai.task model
    │   ├── Span ID: 9567d0b878dce12b  - chat ChatOpenAI [op:chat]
    │   └── Span ID: 9ef4fd28e1fac76f  - gen_ai.task model_to_tools
    ├── Span ID: 1c4c2c3561a6bdaa  - gen_ai.task tools
    │   ├── Span ID: d9715ba2dc8026fe  - tool get_weather [op:execute_tool]
    │   └── Span ID: da16fac4673f3da1  - gen_ai.task tools_to_model
    └── Span ID: f5330649bb2b3ba0  - gen_ai.task model
        ├── Span ID: a85348539e216d94  - chat ChatOpenAI [op:chat]
        └── Span ID: c0ecb30e07fe5291  - gen_ai.task model_to_tools
```

Highlights:

- A single agent span roots the trace; chain tasks and tool calls remain within the same trace.
- Tool execution is captured twice: once as the orchestration `Task` (`gen_ai.task tools`) and once as the `ToolCall` span (`tool get_weather`).
- Follow-up model/tool orchestration stays under the agent root through `parent_run_id` propagation.


## Lifecycle Cheat Sheet

```text
on_chain_start (root)
    ├─ agent metadata → start_agent(AgentInvocation)
    └─ otherwise     → start_workflow(Workflow)
        │
        └─ on_chain_start (child)
             ├─ tool metadata → start_tool_call(ToolCall)
             │       └─ on_tool_end → stop_tool_call
             └─ default       → start_task(Task)
                     ├─ on_chat_model_start / on_llm_start → start_llm
                     ├─ on_llm_end                          → stop_llm
                     └─ on_chain_end                        → stop_task
        └─ on_chain_end → stop_agent / stop_workflow

Errors (any callback) → fail_by_run_id → TelemetryHandler.fail_* → span status = ERROR
```

Keep this document aligned with the handler whenever new callbacks or entity types gain support.

