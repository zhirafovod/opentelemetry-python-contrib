# GenAI Extra Evaluation Metrics Plan

Status: DRAFT
Owner: (add GitHub handle)
Last Updated: 2025-10-10

## 1. Objective
Extend the evaluation telemetry pipeline to consistently emit canonical histogram instruments for a broader set of GenAI quality metrics while keeping the surface area vendor-neutral and minimally opinionated. Core existing canonical metrics (bias, toxicity, relevance, sentiment, hallucination) will be validated and enriched; additional derived or composite metrics (linguistic_quality, coherence, factual_accuracy) will be integrated via a pluggable evaluation stack leveraging deepeval first, with graceful fallback to simple heuristics / lightweight NLP (NLTK) where deepeval lacks coverage or the user opts out.

## 2. Canonical Instrument Names
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

Proposed / Existing slug map:
| Human Concept | Slug | Exists Today? | Notes |
|---------------|------|---------------|-------|
| Relevance | relevance | PARTIAL (naming reserved, need implementation) | Score similarity between answer & query (semantic + lexical). |
| Hallucination | hallucination | PARTIAL | Inverse factuality; may derive from factual_accuracy or external LLM judge. Lower better; we normalize so higher=better OR store as risk? Decision: Represent "hallucination" as probability of hallucination (higher=worse). Provide derived passed if below threshold. |
| Sentiment | sentiment | PARTIAL | Polarity mapped to [0,1] (0=neg,1=pos) with midpoint neutral. |
| Toxicity | toxicity | IMPLEMENTED (dynamic test) | Already created when results provide metric_name="toxicity". Need standard scoring guidance. |
| Bias | bias | IMPLEMENTED (dynamic test) | Value 0 (no bias) → 1 (high bias) or inverse? Keep raw detector output normalized to [0,1]; lower=better. |
| Linguistic Quality (Grammar/Utility/Readability) | linguistic_quality | NEW | Composite of grammar correctness, readability, clarity. Use deepeval composite or custom aggregator. |
| Coherence | coherence | NEW | Logical flow & internal consistency. |
| Factual Accuracy | factual_accuracy | NEW | Overlap with hallucination; factual_accuracy = 1 - hallucination probability when both present; else direct detection. |

Rationale for slugs: lower snake case; avoid ambiguous terms (e.g. "quality" alone too broad). All map to evaluation result `metric_name` values.

## 3. Metric Semantics & Sourcing
For each metric we describe preferred source ordering (higher priority first) and fallback logic.

### Relevance
Goal: Measures topical alignment of response to user prompt.
Sources:
1. deepeval: `AnswerRelevancyMetric` (semantic similarity + LLM judge) if available.
2. Embedding cosine similarity using provider / fallback open model (sentence-transformers) normalized to [0,1].
3. NLTK lexical overlap / Jaccard similarity (stemmed & stopword removed) if embeddings disabled.
Derivation: If LLM judge returns categorical labels (e.g., High/Medium/Low), map to numeric via config table.

### Hallucination
Goal: Probability response contains fabricated or incorrect facts.
Sources:
1. deepeval: `HallucinationMetric` if present.
2. If factual_accuracy available, hallucination = 1 - factual_accuracy.
3. Lightweight heuristic: Named entity extraction (spaCy optional) cross-checked against prompt/reference set — if unseen entities proportion high, hallucination risk increases.
Label: If numeric < threshold_low -> "low", > threshold_high -> "high".

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

### Linguistic Quality
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

## 4. Gap Analysis (Current vs Target)
Current State Observations:
- Dynamic metric infrastructure exists (`EvaluationMetricsEmitter`) that creates histograms on first observation of `metric_name`.
- Tests only exercise bias & toxicity dynamic creation (`test_evaluation_metrics_dynamic.py`).
- No normalization or semantic mapping layer for additional metrics.
- No deepeval integration stub present; evaluation results appear to be provided externally.
- No derivation (e.g., hallucination from factual_accuracy) implemented.
- No standardized label mapping or pass/fail threshold configuration beyond simple label presence.
- Content of evaluation results array does not currently assert numeric range normalization.

Required Additions:
1. Metric normalization & mapping module to transform raw evaluator outputs into canonical numeric [0,1] score + optional label.
2. Derivation engine to produce secondary metrics (hallucination from factual_accuracy, ensure no duplication if both provided explicitly).
3. Configurable thresholds (per metric) for pass/fail and label binning.
4. deepeval adapter layer producing `EvaluationResult` objects.
5. Optional heuristics provider fallback (embedding similarity, VADER, lexical overlap) gated by lightweight dependencies (import guarded).
6. Documentation & CHANGELOG updates for new metrics & semantics.
7. Tests covering each metric path including fallback & derivation logic.

## 5. Architecture Additions
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

## 6. Metric Normalization Guidelines
- All scores float 0..1 inclusive. If provider outputs 1..5, divide by 5. If -1..1 (sentiment), transform via (x+1)/2.
- If metric inherently "lower is better" (toxicity, bias, hallucination probability), we still store raw probability (lower better) and DO NOT invert; UI/analysis can interpret direction. Provide attribute `gen_ai.evaluation.direction` = "lower_better" or "higher_better" for clarity.
- Composite metrics document component contributions under `gen_ai.evaluation.components` (JSON-serializable dict) if small.

## 7. Threshold & Label Strategy (Defaults)
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

## 8. Implementation Plan (Tasks)
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
```

## 12. AI Coder Implementation Prompt
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

Stepwise Tasks:
1. Create config module with dataclass & load-from-env helpers.
2. Implement normalization & label assignment functions.
3. Update `EvaluationResult` (if needed) & adjust tests.
4. Add derivation logic in manager pre-emission.
5. Implement deepeval adapter with graceful ImportError handling.
6. Implement heuristic fallbacks (embedding similarity util, VADER sentiment wrapper, lexical overlap, entity novelty heuristic) each behind feature flags.
7. Modify emitters to include direction/components attributes.
8. Add unit tests for each new function and path.
9. Update docs & CHANGELOG in this README.

Coding Guidelines:
- Keep dependencies optional by wrapping imports: `try: import ... except ImportError: <flag>=False`.
- Avoid large model downloads during tests; mock embedding model where needed.
- Ensure metrics remain lazily registered (no creation before first record call).
- All new attributes must use `gen_ai.evaluation.*` prefix; no vendor naming.

Definition of Done:
- All planned CHANGELOG entries moved to done with brief note.
- All tests (existing + new) pass.
- No mandatory dependency additions for base install.

---
End of document.
