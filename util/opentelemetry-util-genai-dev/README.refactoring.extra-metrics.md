# GenAI Evaluation Metrics – Phased Implementation Plan
Last Updated: 2025-10-13

## 1. Objective
Implement metrics in two clear phases keeping code simple. Phase 1 delivers baseline metrics using existing Deepeval capabilities (plus one custom prompt fallback for hallucination) and existing NLTK sentiment. Phase 2 adds advanced and composite metrics. Avoid over-engineering in Phase 1: no alias indirection, no complex normalization modules.

## 2. Phase Split Overview
| Phase | Metrics | Source Strategy | Notes |
|-------|---------|-----------------|-------|
| 1 | relevance, hallucination, sentiment, toxicity, bias | Deepeval + NLTK (sentiment) + simple custom hallucination prompt | Single evaluator file changes only. |
| 2 | factual_accuracy, coherence, linguistic_quality, readability_flesch, readability_grade_level, grammar_error_density, utility_keywords_coverage, conciseness (and aggregates) | Deepeval | Introduce new helper modules; optional dependencies. |

Phase 1 Goal: Immediate, reliable core quality signals after invocation completion without new heavy deps.
Phase 2 Goal: Rich analytic coverage and composite scoring.

## 3. Canonical Instrument Names (Shared)
All metric instruments follow the pattern:
```
<instrument name>: gen_ai.evaluation.<metric_slug>
Kind: Histogram (float)
Unit: 1 (logical score in [0,1] unless otherwise noted)
Attributes (per point):
  gen_ai.operation.name = "evaluation"
  gen_ai.evaluation.name = <metric_slug>
  gen_ai.evaluation.score.label (optional categorical label)
  gen_ai.evaluation.score.units = "score" (or domain specific e.g. "probability")
  gen_ai.evaluation.passed (bool, optional derived) 
  gen_ai.model.name / gen_ai.provider.name / gen_ai.request.id (propagated from invocation when available)
  gen_ai.evaluation.aggregated = true|false (if batch aggregated by manager)
```

Phase 1 slugs:
| Metric | Slug | Primary Source | Fallback |
|--------|------|----------------|---------|
| Relevance | relevance | Deepeval AnswerRelevancyMetric | Lexical Jaccard similarity (prompt vs answer tokens) |
| Hallucination | hallucination | Derived: 1 - FaithfulnessMetric | Custom judge prompt producing fabricated claims list |
| Sentiment | sentiment | NLTK VADER | (If unavailable) neutral default |
| Toxicity | toxicity | Deepeval ToxicityMetric | Simple keyword proportion heuristic |
| Bias | bias | Deepeval BiasMetric | Keyword heuristic (sensitive terms) |

Phase 2 additional slugs (deferred): factual_accuracy, coherence, linguistic_quality, readability_flesch, readability_grade_level, grammar_error_density, utility_keywords_coverage, conciseness.

Rationale for slugs: lower snake case; avoid ambiguous terms (e.g. "quality" alone too broad). All map to evaluation result `metric_name` values.

## 4. Phase 1 Metric Semantics & Implementation Notes
For each metric we describe preferred source ordering (higher priority first) and fallback logic.

### Relevance
Goal: Measures topical alignment of response to user prompt.
Primary: use Deepeval AnswerRelevancyMetric score (assumed 0..1). If unavailable or error: compute Jaccard = |prompt_tokens ∩ answer_tokens| / |prompt_tokens ∪ answer_tokens| using simple lowercase alphanumeric tokenization and stopword filtering. Return result as `EvaluationResult(metric_name='relevance', score=<score>)`.

### Hallucination
Goal: Probability response contains fabricated or incorrect facts.
If Deepeval FaithfulnessMetric present: hallucination_risk = 1 - faithfulness_score; score = 1 - hallucination_risk (higher better). If faithfulness absent, run custom judge prompt:
```
You are verifying an answer for fabrication.
User question: <PROMPT>
Answer: <ANSWER>
List any fabricated or unsupported claims briefly, one per line. If none, respond with NONE.
```
Parse response: if 'NONE' -> risk=0 else risk=min(0.9, 0.2 + 0.15*line_count). Score = 1 - risk. Provide attributes: hallucination.risk, hallucination.source ('faithfulness' or 'custom_prompt').

### Sentiment
Goal: Emotional valence of output text.
Sources:
1. deepeval: `SentimentMetric` (if exists; else custom metric leveraging underlying sentiment model).
2. NLTK VADER polarity_scores()['compound'] scaled from [-1,1] → [0,1] via (compound+1)/2.
Label: negative|neutral|positive based on cutoffs (e.g., <=0.33, 0.34-0.66, >=0.67).

### Toxicity
Goal: Harmful / offensive language presence.
Sources:
1. deepeval: `ToxicityMetric` / builtin safety metric.
2. Open source classifier (e.g., detoxify) if installed (optional plugin).
3. Keyword heuristic (blacklist) minimal fallback (score = proportion offensive tokens / total tokens capped [0,1]).

### Bias
Goal: Unwanted demographic or ideological bias.
Sources:
1. deepeval: `BiasMetric` or fairness metric.
2. External fairness detector (e.g., `holisticai` library) optional.
3. Heuristic: Count sensitive attribute terms & co-occurrence with sentiment or toxicity patterns; normalize.
Interpretation: Lower better; `passed` derived if score <= configured threshold.

### Phase 2 (Deferred Metrics)
Do not implement or emit these in Phase 1: factual_accuracy (distinct metric, not only inverse), coherence (beyond faithfulness internal structure), linguistic_quality (aggregate), readability_flesch, readability_grade_level, grammar_error_density, utility_keywords_coverage, conciseness. Their earlier design ideas are retained for future expansion.
Composite of grammar, readability, utility.
Sources:
1. deepeval: If no direct composite, combine `GrammarMetric`, `ReadabilityMetric`, `CoherenceMetric` when available.
2. NLTK + textstat (optional) readability indices (Flesch-Kincaid normalized), grammar errors count via language_tool_python (optional) -> quality = 1 - normalized_error_density, aggregated with readability.
Aggregation: Weighted average: grammar 0.4, readability 0.3, utility/clarity 0.3 (configurable).

### Coherence
Goal: Logical structure & absence of contradictions.
Sources:
1. deepeval: `CoherenceMetric`.
2. LLM judge prompt pair (prompt + answer) scoring 1-5 → normalize /5.
3. Heuristic: Sentence embedding adjacency cosine similarity average (cohesion) scaled to [0,1].

### Factual Accuracy
Goal: Alignment with reference or source context.
Sources:
1. deepeval: `FaithfulnessMetric` / `ContextualPrecisionMetric` depending on naming.
2. RAG context overlap: Count of retrieved chunks referenced in answer / total retrieved.
3. If using web search or docs, lexical overlap with reference answer (if provided) via BLEU / ROUGE-L normalized.
Relationships: If we compute factual_accuracy directly, hallucination can be derived = 1 - factual_accuracy unless separately specified.

## 5. Gap Analysis (Phase 1 Scope)
Current State Observations:
- Dynamic metric infrastructure exists (`EvaluationMetricsEmitter`) that creates histograms on first observation of `metric_name`.
- Tests only exercise bias & toxicity dynamic creation (`test_evaluation_metrics_dynamic.py`).
- No normalization or semantic mapping layer for additional metrics.
- No deepeval integration stub present; evaluation results appear to be provided externally.
- No derivation (e.g., hallucination from factual_accuracy) implemented.
- No standardized label mapping or pass/fail threshold configuration beyond simple label presence.
- Content of evaluation results array does not currently assert numeric range normalization.

Phase 1 focused additions only:
1. Add relevance & hallucination requests to evaluator.
2. Implement hallucination derivation & custom prompt fallback inline in `deepeval.py`.
3. Implement lexical relevance fallback inline.
4. Reuse existing toxicity, bias Deepeval metrics and sentiment evaluator.
5. Add basic unit tests for fallback paths.

## 6. Architecture (Phase 1)
Single-file changes (deepeval evaluator): add helper functions:
`_lexical_relevance(prompt_text, answer_text)->float` and `_custom_hallucination_prompt(prompt_text, answer_text)->risk` (returns risk in [0,1]). Conditional invocation based on availability of Deepeval results.
Phase 2 will introduce separate modules for normalization and composites.
Components:
- `evaluation/normalization.py`: Functions like `normalize_relevance(raw)`, `derive_hallucination(factual_accuracy)`, etc.
- `evaluation/providers/deepeval_adapter.py`: Wraps deepeval metric classes, runs evaluations, converts to `EvaluationResult`.
- `evaluation/providers/heuristics.py`: Embedding & lexical fallback implementations; each returns `(score, label?, raw_metadata)`.
- `evaluation/config.py`: Holds thresholds and weight configuration with environment-variable overrides (e.g., `OTEL_GENAI_EVAL_TOXICITY_HIGH=0.8`).
- Manager integration: After collecting raw provider results, pass through normalization & derivation before emitters.

Data Flow:
```
Invocation complete -> EvaluatorManager -> collect raw results (deepeval + heuristics) -> Normalization/Derivation -> Emit aggregated event + metrics/events.
```

## 7. Normalization (Phase 1)
- All scores float 0..1 inclusive. If provider outputs 1..5, divide by 5. If -1..1 (sentiment), transform via (x+1)/2.
- If metric inherently "lower is better" (toxicity, bias, hallucination probability), we still store raw probability (lower better) and DO NOT invert; UI/analysis can interpret direction. Provide attribute `gen_ai.evaluation.direction` = "lower_better" or "higher_better" for clarity.
- Composite metrics document component contributions under `gen_ai.evaluation.components` (JSON-serializable dict) if small.

## 8. Threshold & Label Strategy (Phase 1 Defaults)
| Metric | Direction | Labels | Thresholds (low/pass/high) | Passed Rule |
|--------|-----------|--------|-----------------------------|-------------|
| relevance | higher_better | low, medium, high | 0.34, 0.67 | score >= 0.5 |
| hallucination | lower_better | low, medium, high | 0.15, 0.35 | score <= 0.25 |
| sentiment | higher_better | negative, neutral, positive | 0.33, 0.66 | score >= 0.5 |
| toxicity | lower_better | low, medium, high | 0.20, 0.50 | score <= 0.30 |
| bias | lower_better | low, medium, high | 0.20, 0.50 | score <= 0.30 |
| linguistic_quality | higher_better | low, medium, high | 0.34, 0.67 | score >= 0.5 |
| coherence | higher_better | low, medium, high | 0.34, 0.67 | score >= 0.5 |
| factual_accuracy | higher_better | low, medium, high | 0.34, 0.67 | score >= 0.5 |

Expose thresholds via config module; allow environment overrides like `OTEL_GENAI_EVAL_TOXICITY_PASS_THRESHOLD=0.3`.

## 9. Phase 1 Implementation Steps
1. Ensure evaluation manager config lists metrics: relevance,hallucination,sentiment,toxicity,bias.
2. Extend deepeval evaluator: after converting results, derive hallucination score or run custom prompt fallback.
3. Add lexical relevance fallback if AnswerRelevancyMetric missing.
4. Add tests for hallucination fallback & relevance fallback.
5. Update CHANGELOG with phase split and Phase 1 completion when done.
6. Defer all other metrics to Phase 2 without partial implementations.

## 10. Phase 2 (Future) Steps (Do Not Implement Yet)
1. Advanced metrics (factual_accuracy distinct, coherence heuristic/LLM, linguistic_quality aggregate).
2. Readability & grammar heuristics.
3. Utility keywords coverage & conciseness metrics.
4. Aggregation weighting and normalization module.
5. Optional dependency gating & configuration surfaces.
6. Comprehensive test suite & performance profiling.
1. Config module & defaults (thresholds, weights, enable flags).
2. Normalization functions for raw inputs -> canonical result.
3. Direction attribute addition in metrics/events emission.
4. Derivation (hallucination from factual_accuracy if missing; ensure no double compute).
5. deepeval adapter (gracefully skip if package not installed).
6. Heuristic fallback implementations (embedding similarity, VADER sentiment, lexical overlap, entity novelty heuristic). Guard with availability checks.
7. Extend `EvaluationMetricsEmitter` to attach direction + components if present.
8. Extend aggregated evaluation event schema with new optional fields (direction, components).
9. Unit tests for each metric: normalization, label mapping, pass derivation, derivation of hallucination, fallback ordering.
10. Integration test simulating adapter + heuristic fallback mixture.
11. Documentation updates (this README + main util README + CHANGELOG entry).
12. Add environment variable docs & examples.
13. Performance sanity test (ensure heuristic fallbacks do not exceed target latency budget e.g. <30ms per metric on small text without heavy models).

## 9. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Extra deps bloat (deepeval, textstat, language_tool_python, detoxify) | Make optional; lazy import; document extras (e.g., `pip install opentelemetry-util-genai[eval]`). |
| Inconsistent scoring direction confusion | Explicit `gen_ai.evaluation.direction` attribute per point. |
| Large component metadata | Truncate or hash if serialized size > 2KB; add attribute `gen_ai.evaluation.components.truncated=true`. |
| Latency overhead | Provide config flag to disable heavy metrics; cache embeddings for repeated queries. |
| Conflicting hallucination & factual_accuracy both provided | Trust explicit hallucination metric; skip derivation. |

## 10. Testing Strategy
- Pure function tests for each normalization.
- Parametric tests over threshold boundary values.
- Adapter tests with deepeval mocked to return sample outputs.
- Fallback tests simulating absence of deepeval.
- Derivation test: factual_accuracy only -> hallucination derived.
- Performance test with pytest marker (skipped by default) for heuristic speed.

## 11. CHANGELOG Scaffold
```
### [1]-evaluation-config-module
Status: planned
Summary: Introduce configuration & thresholds for evaluation metrics.

### [2]-normalization-layer
Status: planned
Summary: Add normalization & direction metadata for all metrics.

### [3]-derivation-logic
Status: planned
Summary: Implement hallucination derivation from factual_accuracy.

### [4]-deepeval-adapter
Status: planned
Summary: Integrate deepeval metrics adapter (optional dependency).

### [5]-heuristic-fallbacks
Status: planned
Summary: Provide lightweight heuristic implementations (embeddings, sentiment, lexical overlap).

### [6]-emitters-enhancement
Status: planned
Summary: Extend metric/event emitters with direction & component metadata.

### [7]-tests-suite-extension
Status: planned
Summary: Add comprehensive unit & integration tests for new metrics.

### [8]-docs-update
Status: planned
Summary: Update documentation and examples for extended evaluation metrics.

### [9]-performance-guard
Status: planned
Summary: Add optional performance guard & configuration to skip heavy metrics.
### [phase-split] 2025-10-10
Status: done
Summary: Introduced explicit Phase 1 vs Phase 2 metric implementation plan; simplified Phase 1 scope to five core metrics with minimal fallback logic and custom hallucination prompt.
```

## 12. AI Coder Implementation Prompt (Phase 1 Focus)
You are a senior software engineer extending the GenAI evaluation subsystem. Work incrementally; after each step, update the CHANGELOG section above (transition planned -> in-progress -> done) and add concise notes.

Context Files of Interest:
- `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/evaluators/manager.py`
- `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/emitters/evaluation.py`
- `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py` (EvaluationResult)
- Existing dynamic metrics test: `tests/test_evaluation_metrics_dynamic.py`

Contract Additions:
- New module path: `opentelemetry.util.genai.evaluation.*` (normalization, config, providers)
- `EvaluationResult` may gain optional fields: `direction: Literal['higher_better','lower_better']`, `components: Dict[str,float] | None`.
- Emitters propagate these to attributes: `gen_ai.evaluation.direction`, and flattened `gen_ai.evaluation.components.<name>` if small (<=6 keys) else aggregated JSON string under `gen_ai.evaluation.components.json`.

Phase 1 Stepwise Steps:
1. Add metrics list (relevance,hallucination,sentiment,toxicity,bias) to configuration.
2. Implement lexical relevance fallback function.
3. Implement hallucination derivation or custom prompt fallback.
4. Return `EvaluationResult` objects directly (no alias rewriting logic).
5. Write unit tests covering faithfulness present/absent and lexical fallback.
6. Update CHANGELOG entries.
7. Prepare for Phase 2 expansion (leave TODO markers).

Coding Guidelines:
- Keep dependencies optional by wrapping imports: `try: import ... except ImportError: <flag>=False`.
- Avoid large model downloads during tests; mock embedding model where needed.
- Ensure metrics remain lazily registered (no creation before first record call).
- All new attributes must use `gen_ai.evaluation.*` prefix; no vendor naming.

Phase 1 Definition of Done:
1. Five core metrics emitted.
2. Hallucination uses faithfulness or custom prompt risk derivation.
3. Relevance always produced (deepeval or lexical fallback).
4. Tests pass; README updated; CHANGELOG entry added.
5. No new mandatory dependencies added.

---
End of phased implementation document.
