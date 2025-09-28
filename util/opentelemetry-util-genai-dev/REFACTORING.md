# GenAI Telemetry Refactoring Snapshot (Phase 3.5 → 4)

Date: 2025-09-27 (Status audit refresh)  
Status: Active development branch (pre-public stability).  
IMPORTANT: This code is still in active development; we are free to change or remove classes without a formal deprecation cycle. Backward compatibility shims are NOT required at this stage.

---
## 1. Purpose of This Document
This file freezes the current architectural snapshot of the GenAI telemetry utilities and provides a precise, implementation‑ready plan for the next refactoring stage ("Phase 3.5 cleanup"), immediately followed by Phase 4 (new domain types: Embeddings & Tool Calls). It replaces ad hoc notes and guides future contributors (including GitHub Copilot automation) on exact next steps.

(Updated 2025-09-27 audit: Progress blocks earlier marked COMPLETE have been re‑evaluated; several items remain partially done. See Sections 4, 11, and 12.)

---
## 2. Current Architectural Snapshot (Post-Audit)
| Area | State |
|------|-------|
| Domain Objects | `LLMInvocation`, `EmbeddingInvocation`, `ToolCall`, `EvaluationResult`, `Error`, message dataclasses (`InputMessage`, `OutputMessage`, parts incl. `ToolCallResponse`) |
| Emission Model | Composite via `CompositeGenerator` delegating to emitters: `SpanGenerator` (alias for forward-looking `SpanEmitter`), `MetricsEmitter`, `ContentEventsEmitter` |
| Span Logic | Consolidated into single span emitter class (`span_generator.py`) with ToolCall span name + operation name support |
| Metrics | Duration + token usage for LLM; duration only for ToolCall (implemented); embeddings produce no metrics (by design) |
| Content Events | LLM input/output messages only (events ignore Embeddings & ToolCalls). Docstring mentions only LLM; explicit ToolCall exclusion not yet documented inline |
| Handler | `TelemetryHandler` exposes both type‑specific wrappers and generic `start/finish/fail` multiplexer |
| Protocol | Informal (emitters implement `start/finish/error`); optional `handles()` present in metrics & content events emitters |
| Evaluations | Implemented for LLM only (events, metrics, optional evaluation spans); not generalized to embeddings/tool calls |
| Environment Parsing | Core generator flavor & content capture in `config.py`; evaluation env flags still read directly in handler (not centralized) |
| Attribute Constants | Partial: only a small subset (`provider`, input/output messages) moved to `attributes.py`; many `gen_ai.*` literals remain elsewhere |
| Legacy Classes | Old inheritance removed; alias `SpanEmitter`→`SpanGenerator` coexist (rename not fully flipped) |
| Tests | Cover LLM, Embedding lifecycles; ToolCall lifecycle + generic routing; missing mixed sequence & thread-safety smoke tests |

---
## 3. Work Completed in Recent Refactor
Unchanged from prior snapshot plus:
- Added `ToolCall` dataclass & lifecycle methods in handler.
- Span emitter updated to set operation name + span naming for ToolCall.
- Metrics emitter records duration for ToolCall (no token metrics as intended).
- Basic ToolCall tests (lifecycle + generic handler path).

---
## 4. Incomplete / Outstanding Gaps (Updated Audit)
| Gap | Current Status | Impact |
|-----|----------------|--------|
| Centralized attribute constants (all `gen_ai.*`) | PARTIAL (only 3 constants centralized) | Hard to adapt to semconv churn; grep still shows many literals |
| Unified immutable config (all env lookups) | PARTIAL (evaluation env vars still read ad hoc) | Harder to test & reason about evaluation feature flags |
| SpanEmitter rename consistency | PARTIAL (file/class still `SpanGenerator`; alias present) | Terminology confusion for newcomers |
| Generic handler API | COMPLETE | Enables uniform extension |
| Metrics/events embeddings doc clarity | PARTIAL (README note present; need explicit doc section) | Possible contributor ambiguity |
| Root span logic simplification | PARTIAL (manual `_current_span` tracking; no explicit context-based root detection flag) | Potential nested span edge cases under concurrency |
| ToolCall domain object | COMPLETE | Unblocks Phase 4 |
| Evaluation pipeline generalization | NOT STARTED (still LLM-only) | Limits future reuse |
| Attribute literals version guarding | NOT STARTED | Churn risk on spec evolution |
| ContentEventsEmitter explicit ToolCall exclusion doc | NOT STARTED (implicit only) | Could cause confusion |
| Mixed sequence test (LLM → ToolCall → LLM → Embedding) | NOT DONE | Missing integration coverage |
| Thread-safety smoke (parallel embeddings/tool calls) | NOT DONE | Concurrency regressions may slip |
| README ToolCall usage snippet | PARTIAL (mentions ToolCall part type; lacks full lifecycle example) | Discoverability gap |

---
## 5. Design Principles Moving Forward
(unchanged)
1. Composition over inheritance.
2. Minimize surface area: one handler, generic lifecycle.
3. Centralize config + attribute names.
4. Extend domain types first, then add telemetry incrementally.
5. No deprecation overhead pre-release.

---
## 6. Phase 3.5 Cleanup (Original Plan & Audit Status)
Actionable tasks & audited status:
1. attributes.py constants → PARTIAL
2. config.py + remove direct env usage → PARTIAL (evaluation flags outstanding)
3. Rename `SpanGenerator` → `SpanEmitter` → COMPLETE (legacy module removed)
4. Root span logic cleanup → PARTIAL (still manual tracking)
5. Generic handler lifecycle → COMPLETE
6. Optional `handles(obj)` in emitters → PARTIAL (implemented in metrics & content events; span emitter lacks explicit method) 
7. Expanded tests (mixed, embedding error, thread) → PARTIAL (embedding error path covered implicitly? mixed/thread missing)
8. Docs update for embeddings limited telemetry → PARTIAL (needs explicit section)

Updated Phase 3.5 DoD (NOT MET yet):
- All `gen_ai.*` literals (except tests) centralized.
- All env parsing (including evaluation) via config.
- Rename consistently applied (or documented deferral).
- Added mixed + thread-safety tests.
- Docs clarify telemetry coverage per type.

---
## 7. Phase 4 (Tool Calls) – Plan & Audit Status
| Task | Status | Notes |
|------|--------|-------|
| 1. ToolCall dataclass | COMPLETE | Implemented in `types.py` |
| 2. Span emitter integration (op name + span name) | COMPLETE | Operation name literal `tool_call` used |
| 3. Handler wrappers | COMPLETE | `start/stop/fail_tool_call` present |
| 4. Metrics duration only | COMPLETE | Implemented in `emitters_metrics.py` |
| 5. ContentEventsEmitter ignore ToolCall w/ explicit doc | PARTIAL | Ignored implicitly; needs doc comment & README note |
| 6. Tests (span attrs, mixed sequence, generic start) | PARTIAL | Lifecycle + generic covered; span attribute assertions & mixed sequence absent |
| 7. Docs / README usage snippet | PARTIAL | Lacks end-to-end snippet |

Phase 4 DoD (NOT FULLY MET): Mixed sequence test, explicit docs, attribute assertions outstanding.

---
## 8. Future (Beyond Phase 4) – Not in Scope Yet
(unchanged)

---
## 9. Risk & Mitigation (Immediate Phases)
| Risk | Updated Mitigation |
|------|--------------------|
| Attribute churn | Finish centralization sprint (expand `attributes.py`) |
| Env sprawl | Move evaluation/env lookups into `config.py` |
| Span parent confusion | Replace `_current_span` with context-based detection or explicit parent injection |
| Type explosion | Continue generic lifecycle & `handles()` filtering |
| Accidental regressions | Add mixed-sequence + concurrency tests |

---
## 10. Implementation Queue (Revised Ordered Backlog)
1. Complete attribute constant extraction (eliminate remaining literals outside `attributes.py` & tests).
2. Centralize evaluation env var parsing into `config.py` (extend `Settings`).
3. Decide on final naming: either fully migrate to `SpanEmitter` (rename file + class) or update docs to treat `SpanGenerator` as canonical (choose one; remove alias if possible).
4. Add `handles()` to span emitter (always True for now).
5. Root span logic: remove `_current_span`; rely on current context or an optional `force_root` flag from settings.
6. Add mixed sequence test: LLM → ToolCall → LLM → Embedding (assert parent linkage, operation names, metrics limited as expected).
7. Add thread-safety smoke test (parallel ToolCall + Embedding invocations) ensuring no shared state collisions.
8. ContentEventsEmitter docstring update + README clarification (tool calls & embeddings produce no content events yet).
9. README: Add ToolCall lifecycle example snippet.
10. Optional: span attribute assertion test for ToolCall (operation name, model/request mapping, provider attr presence).
11. (Stretch) Evaluate feasibility of central evaluation pipeline generalization (defer if large).

---
## 11. Guidance for Contributors / Automation (Adjusted Checklist)
```
[x] Centralize remaining gen_ai.* attribute literals
[x] Extend attributes.py (include evaluation, framework, completion parts, etc.)
[ ] Extend Settings & parse_env for evaluation flags (enable, span mode, evaluator list)
[ ] Remove direct os.environ usage from handler (evaluation paths)
[ ] Add handles() method to span emitter
[ ] Root span logic refactor (context-based) & remove _current_span
[ ] Rename span_generator.py → span_emitter.py (or document decision)
[ ] Remove legacy `generators.py` file; rely on the `generators/` package (SpanEmitter) only
[ ] Mixed sequence test (LLM → ToolCall → LLM → Embedding)
[ ] Thread-safety smoke test (parallel invocations)
[ ] ToolCall span attribute assertion test
[ ] ContentEventsEmitter doc update (explicit ToolCall/Embedding exclusion)
[ ] README: ToolCall lifecycle example
[ ] README / docs: Telemetry coverage matrix (LLM vs Embedding vs ToolCall)
```
Rules (unchanged): keep diffs small, commit after each milestone, maintain typing.

---
## 12. Progress Tracking Block (Reset After Audit)
```
Phase 3.5 Status:
legacy generator files: COMPLETE
attributes.py:      COMPLETE
config.py:          PARTIAL
spanEmitter rename: COMPLETE
generic lifecycle:  COMPLETE
root logic fix:     PARTIAL
tests (3.5 set):    PARTIAL
handles() API:      COMPLETE

deficit summary: root logic + tests outstanding.

Phase 4 Status:
ToolCall dataclass: COMPLETE
span integration:   COMPLETE
handler wrappers:   COMPLETE
metrics support:    COMPLETE
ToolCall tests:     PARTIAL
docs update:        PARTIAL
```

---
## 13. Definition of Done (Revised)
The refactor is DONE when:
- No direct `gen_ai.` literals exist outside `attributes.py` & tests.
- All env var parsing consolidated in `config.py` (`Settings`).
- Span emitter naming consistent & documented (no ambiguous alias necessity).
- Generic lifecycle used in examples; wrappers remain thin and delegate.
- Mixed sequence & thread-safety tests present and green.
- LLM, Embedding, ToolCall spans produce expected attributes without errors.
- Metrics: LLM (duration + tokens) / ToolCall (duration) / Embedding (none) clearly documented.
- ContentEvents: LLM-only documented; explicit exclusions stated.
- README includes ToolCall lifecycle example + telemetry coverage matrix.

---
## 14. Appendix: attributes.py Expansion Plan (Next Step)
Target additions (illustrative):
```
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"  # existing
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"  # existing
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"  # existing
GEN_AI_FRAMEWORK = "gen_ai.framework"
GEN_AI_COMPLETION_PREFIX = "gen_ai.completion"  # for indexed outputs
GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
```
(Refine once full literal audit complete.)

---
## 15. Notes
This document now reflects an audited, realistic state rather than optimistic completion. Update after each milestone; once Phase 4 DoD is truly met, either archive this snapshot or promote decisions into a formal ADR.

---
End of REFACTORING snapshot.
