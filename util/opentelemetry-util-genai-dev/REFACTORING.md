# GenAI Telemetry Refactoring Snapshot (Phase 3.5 → 4)

Date: 2025-09-27 (Post README refresh update)  
Status: Active development branch (pre-public stability).  
IMPORTANT: API is still experimental; breaking changes permitted without deprecation cycle.

---
## 1. Purpose
Snapshot of current architecture and the **remaining** focused refactor items after completing the documentation alignment (README + coverage matrix) and structural simplifications.

---
## 2. Current Architectural Snapshot (Updated)
| Area | State |
|------|-------|
| Domain Objects | `LLMInvocation`, `EmbeddingInvocation`, `ToolCall`, `EvaluationResult`, `Error`, message dataclasses & parts |
| Emission Model | Composition: `CompositeGenerator` + emitters (`SpanEmitter`, `MetricsEmitter`, `ContentEventsEmitter`) |
| Span Logic | Single `SpanEmitter` (file: `generators/span_emitter.py`) using context manager (`start_as_current_span`) |
| Metrics | LLM: duration + token histograms; ToolCall: duration; Embedding: none (by design) |
| Content Events | LLM only; explicit exclusion for ToolCall & Embedding documented in emitter + README |
| Handler | `TelemetryHandler` with type‑specific helpers and generic `start/finish/fail` multiplexer |
| Protocol | Informal emitter contract: `start/finish/error` (+ optional `handles`) |
| Evaluations | LLM only (histogram + consolidated event + optional spans) |
| Environment Parsing | All generator + evaluation flags parsed centrally in `config.parse_env()` -> `Settings` |
| Attribute Constants | PARTIAL centralization (core + evaluation subset); some literals still inline (mainly evaluation aggregation & auxiliary keys) |
| Legacy Alias | `SpanGenerator = SpanEmitter` exported for transitional naming; removal planned pre-stability |
| Tests | Coverage includes mixed sequence, thread-safety, tool call span attributes, evaluation paths |

---
## 3. Recent Work Completed
- README completely rewritten to emitter / flavor model.
- Coverage matrix + tool call & embedding examples added.
- Mixed sequence + thread-safety tests present and green.
- ContentEventsEmitter docstring explicitly states exclusions.
- Evaluation env flags fully centralized in `Settings` (no direct ad hoc `os.environ` reads in handler evaluation logic).
- Root span logic already context-based (no `_current_span` state to remove).

---
## 4. Remaining Gaps (Post-Refresh)
| Gap | Status | Impact |
|-----|--------|--------|
| Full attribute constant centralization | PARTIAL | Harder to adapt to semconv churn; grep still yields inline `gen_ai.evaluation.*` & aggregation keys |
| Evaluation attribute aggregation constants (count/min/max/avg/names) | NOT DONE | Minor duplication & inconsistency risk |
| Alias deprecation plan (`SpanGenerator`) | NOT FINALIZED | Potential confusion for new contributors |
| Evaluation generalization (Embeddings / ToolCall) | NOT STARTED | Limits reuse of evaluator infra |
| Documentation of evaluation span parenting choice (link vs parent) | PARTIAL | Ambiguity for downstream span topology expectations |
| Attribute version / feature flag strategy (guard experimental) | NOT STARTED | Harder to communicate semconv evolution |
| Optional helper utilities (e.g. `get_genai_semconv_version()`) | NOT STARTED | Observability tooling convenience gap |
| Redaction / truncation policy guidance | NOT STARTED | Risk of large payload spans/events |

---
## 5. Design Principles (Stable)
1. Composition over inheritance.
2. Single handler façade; emitters are pluggable.
3. Centralize config & attribute naming.
4. Minimize surface area until divergence is proven.
5. Fast iteration over early stability guarantees.

---
## 6. Definition of Done (Refined)
Refactor phase considered DONE when:
- All `gen_ai.*` attribute keys (excluding tests) sourced from `attributes.py` (including evaluation aggregation keys).
- Alias `SpanGenerator` either removed or explicitly documented as temporary with removal milestone.
- Evaluation span parenting behavior documented (decision recorded in ADR or README snippet).
- README + emitter docs remain consistent with code (spot check passes).
- Optional: small helper for semconv version exported.

(General feature expansion—evaluation generalization, redaction utilities—tracked separately; not blocking refactor completion.)

---
## 7. Updated Implementation Queue (Ordered)
1. Attribute constant pass: add remaining evaluation aggregation & supporting constants; replace literals in handler.
2. Introduce optional constants for operation value fallbacks (e.g. `tool_call`, `embedding`) for uniformity.
3. Decide & document evaluation span parenting (currently link-based) – record rationale (link avoids accidental parent latency skew). Update README or mini ADR.
4. Alias strategy: either remove `SpanGenerator` export or add deprecation note in code comment + REFACTORING with removal condition (e.g. before first beta tag).
5. (Optional) Provide `get_genai_semconv_version()` returning pinned schema URL / semconv version for debug logs.
6. Draft attribute versioning / churn note (simple mapping / diff guidance in README or dedicated ATTRIBUTES.rst).
7. (Stretch) Add redaction guidance section (point to future hook or env toggle) to mitigate large message bodies.
8. (Stretch) Evaluate feasibility of evaluator generalization for embeddings, then tool calls (may need new result semantic categories).

---
## 8. Risk & Mitigation (Focused)
| Risk | Mitigation |
|------|-----------|
| Attribute drift vs semconv | Complete constant centralization; single mapping file. |
| Contributor confusion over alias | Clarify & schedule removal. |
| Unexpected large event payloads | Add redaction guidance & future hook placeholder. |
| Misinterpreted evaluation span hierarchy | Document link-based approach & reasoning. |

---
## 9. Progress Tracker (Rolling)
```
Centralize remaining literals:    PENDING
Evaluation agg constants:         PENDING
Alias decision documented:        PENDING
Evaluation span parenting doc:    PENDING
Semconv version helper:           PENDING (optional)
Attribute versioning note:        PENDING
Redaction guidance:               PENDING (stretch)
Evaluator generalization:         PENDING (stretch)
```

---
## 10. Notes
Core structural refactor tasks (composite architecture, docs alignment, test coverage for new domain types, context-based spans) are complete. Remaining work is *consolidation* (constants + documentation clarity) and *future-facing hardening* (evaluator scope, attribution/version helpers).

---
End of updated refactoring snapshot.
