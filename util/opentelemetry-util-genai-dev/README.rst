OpenTelemetry GenAI Utilities (opentelemetry-util-genai)
========================================================

.. contents:: Table of Contents
   :depth: 4
   :local:
   :backlinks: entry

Overview
--------
This package provides foundational data types, helper logic, and lifecycle utilities for emitting OpenTelemetry signals around Generative AI (GenAI) model invocations.

Key audiences:

* Instrumentation authors (framework / model / provider wrappers)
* Advanced users building custom GenAI telemetry capture pipelines
* Early adopters validating incubating GenAI semantic conventions (semconv)

Architecture (High Level)
-------------------------
Telemetry emission is composition‑based: a ``CompositeGenerator`` coordinates a list of small "emitters" each responsible for one signal category.

::

   Instrumented Code / Wrapper
       → Build domain object (LLMInvocation / EmbeddingInvocation / ToolCall)
       → handler.start_* / handler.start(...)  (composite fan‑out: span, metrics, events)
       → Provider/API call
       → Populate outputs (messages, token counts, response metadata)
       → handler.stop_* / handler.finish(...)
       → Exporter pipelines ship spans / metrics / logs
       → (Optionally) handler.evaluate_llm(invocation)

Emitters (current set):

* ``SpanEmitter`` – tracing spans & core attributes.
* ``MetricsEmitter`` – latency + (LLM) token histograms.
* ``ContentEventsEmitter`` – structured log events containing message content.

A telemetry "flavor" (selected via env var) is just a predefined combination of these emitters plus rules for where message content may appear.

Evaluation Emission (New)
-------------------------
Evaluation execution has been refactored into an extensible pipeline:

* ``EvaluationManager`` orchestrates evaluator resolution, execution, and telemetry emission.
* Pluggable evaluation emitters (metrics, events, spans) mirror the main generator design, enabling future customization (e.g. persistence, tracing-only, enrichment layers).

The public API ``TelemetryHandler.evaluate_llm`` is unchanged; internally it delegates to ``EvaluationManager``.

Core Concepts
-------------

GenAI Data Types
~~~~~~~~~~~~~~~~
The GenAI domain model is intentionally thin (pure data containers). All are defined in ``opentelemetry.util.genai.types``.

+----------------------+---------------------------------------------------------------------------------------------+
| Type                 | Purpose / Key Fields                                                                        |
+======================+=============================================================================================+
| LLMInvocation        | Represents a chat / completion style model call. Holds request / response models,           |
|                      | input/output messages, token counts, provider, custom attributes, span context fields.      |
+----------------------+---------------------------------------------------------------------------------------------+
| EmbeddingInvocation  | Represents an embedding generation call (input_texts, optional vector dimensions).          |
|                      | Excluded from metrics (tokens) & content events by design.                                  |
+----------------------+---------------------------------------------------------------------------------------------+
| ToolCall             | External tool/function invocation (arguments, name, optional provider). Duration only.      |
+----------------------+---------------------------------------------------------------------------------------------+
| InputMessage         | Chat input message: ``role`` + list of structured parts (e.g. Text, ToolCall references).   |
+----------------------+---------------------------------------------------------------------------------------------+
| OutputMessage        | Chat output message: ``role`` + parts + finish_reason.                                      |
+----------------------+---------------------------------------------------------------------------------------------+
| Text (part)          | Simple textual content part used inside messages.                                           |
+----------------------+---------------------------------------------------------------------------------------------+
| ToolCallResponse     | Structured part capturing response to a prior tool call (placeholder for future logic).     |
+----------------------+---------------------------------------------------------------------------------------------+
| EvaluationResult     | Result of a single evaluator metric: metric_name, score, label, explanation, attributes.    |
+----------------------+---------------------------------------------------------------------------------------------+
| Error                | Normalized error container (message + exception type).                                      |
+----------------------+---------------------------------------------------------------------------------------------+
| ContentCapturingMode | Enum controlling if/where message bodies are captured: NO_CONTENT / SPAN_ONLY /             |
|                      | EVENT_ONLY / SPAN_AND_EVENT.                                                                |
+----------------------+---------------------------------------------------------------------------------------------+

Design Notes:
* Data classes never perform emission logic; emitters inspect them via ``isinstance``.
* ``LLMInvocation.messages`` & ``chat_generations`` are convenience mirrors maintained for backward compatibility.
* Timestamps (``start_time`` / ``end_time``) are filled by the handler; instrumentation authors set input data only.

Util GenAI Components
~~~~~~~~~~~~~~~~~~~~~
The telemetry layer *interprets* data types into OpenTelemetry signals.

Telemetry Handler
^^^^^^^^^^^^^^^^^
``TelemetryHandler`` (``handler.py``) is the façade used by instrumentation.

Responsibilities:
- Parse and cache env configuration (flavor, content capture, evaluation flags).
- Construct the appropriate emitter set once (flavor governs composition).
- Provide strongly named lifecycle helpers (``start_llm``, ``stop_tool_call``) plus generic ``start/finish/fail`` dispatch.
- Post‑completion evaluation triggering (``evaluate_llm``) including metric & event emission for evaluation results.

Emitters
^^^^^^^^
Independent units (``emitters/*.py``) implementing ``start(obj)``, ``finish(obj)``, ``error(error,obj)`` and optional ``handles(obj)``.

- ``SpanEmitter``: Creates / finalizes spans. Applies semantic attributes, optional message serialization (depending on flavor + capture mode). Robust to missing output.
- ``MetricsEmitter``: Records latency for all supported objects and token usage for ``LLMInvocation`` only. Ignores embeddings for token metrics; records duration for ToolCall.
- ``ContentEventsEmitter``: Emits structured log events for input & output chat messages (LLM only) when event capture enabled.

Composite Generator
^^^^^^^^^^^^^^^^^^^
``CompositeGenerator`` (``emitters/composite.py``) is an ordered fan‑out orchestrator. It guarantees span start happens before metrics/events, and span end after they finish (metrics & events finish first, span last), ensuring emitters can still read live span context.

Evaluators
^^^^^^^^^^
Evaluator implementations (``evaluators/*``) provide domain-specific quality / scoring logic. A registry pattern allows lazy dynamic loading. An evaluator returns one or more ``EvaluationResult`` items.

- Built-ins (length, sentiment) loaded on demand.
- External packages (e.g., ``deepeval``) can integrate by registering a factory.

Upload Hooks
^^^^^^^^^^^^
Upload hooks (``upload_hook.py`` + optional entry-points) provide pluggable persistence of prompt / response artifacts (e.g., fsspec cloud/object storage) via a simple ``upload(...)`` interface (see ``FsspecUploadHook`` implementation).

Content Capture Enforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flavor + ``ContentCapturingMode`` together dictate whether messages appear on spans, events, both, or not at all (see matrices below). Emitters do *not* read env directly; the handler refreshes capture mode and updates emitters before starting new LLM spans.

Lifecycle Overview
^^^^^^^^^^^^^^^^^^^
1. Instrumentation builds an invocation data object.
2. Handler ``start_*`` delegates to ``CompositeGenerator`` → span emitter starts span.
3. Provider executes; instrumentation populates outputs (messages, tokens, response id/model, custom attributes).
4. Handler ``stop_*`` delegates finish → metrics/event emitters record while span still active → span emitter closes span.
5. Optional: ``evaluate_llm`` executes evaluators → metrics (scores), single evaluations event, and optionally evaluation spans.

Extension Points Summary
^^^^^^^^^^^^^^^^^^^^^^^^
- Add a new emitter: implement the three lifecycle methods and (optionally) ``handles()``; inject into a custom handler instance before use.
- Add a new evaluator: subclass / follow Evaluator protocol, register via ``register_evaluator(name, factory)``.
- Add an upload hook: publish an entry point ``opentelemetry_genai_upload_hook`` returning an object with ``upload(...)``.

Emitter Flavors (Environment Selection)
---------------------------------------
Set ``OTEL_INSTRUMENTATION_GENAI_EMITTERS`` (case‑insensitive): ``span`` (default) | ``span_metric`` | ``span_metric_event``.

+--------------------+-------------------------------+-------------------+---------------------------+-----------------------------------------------+
| Flavor             | Included Emitters             | Spans             | Metrics                   | Content Events & Message Content Placement    |
+====================+===============================+===================+===========================+===============================================+
| span               | SpanEmitter                   | Yes               | No                        | Message content → span attrs (if mode allows) |
+--------------------+-------------------------------+-------------------+---------------------------+-----------------------------------------------+
| span_metric        | SpanEmitter, MetricsEmitter   | Yes               | Duration + tokens (LLM)   | Message content → span attrs (if mode allows) |
+--------------------+-------------------------------+-------------------+---------------------------+-----------------------------------------------+
| span_metric_event  | SpanEmitter, MetricsEmitter,  | Yes (no messages  | Duration + tokens (LLM)   | Message content → events only (if mode allows)|
|                    | ContentEventsEmitter          | on span)          |                           |                                               |
+--------------------+-------------------------------+-------------------+---------------------------+-----------------------------------------------+

Message Content Capture Modes
-----------------------------
Requires enabling experimental semconv (see Environment Variables). Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of:

* ``NO_CONTENT`` (default)
* ``SPAN_ONLY``
* ``EVENT_ONLY``
* ``SPAN_AND_EVENT``

Interplay Rules:

* Flavor ``span`` / ``span_metric``: Only SPAN_ONLY / SPAN_AND_EVENT cause messages to be serialized onto span attributes. EVENT_ONLY acts like NO_CONTENT for these flavors.
* Flavor ``span_metric_event``: Messages are never added to spans. EVENT_ONLY / SPAN_AND_EVENT allow events; SPAN_ONLY is treated like NO_CONTENT to avoid duplication.

Telemetry Coverage Matrix (Invocation Types)
--------------------------------------------
+----------------------+---------------------------+----------------------------+--------------------------------------+----------------------------------------------+
| Invocation Type      | Span                      | Metrics                    | Content Events (messages)            | Message Content Placement                    |
+======================+===========================+============================+======================================+==============================================+
| LLMInvocation        | Yes (chat {model})        | Duration (+ tokens LLM)    | Only flavor=span_metric_event &      | Span (span/span_metric) or events            |
|                      |                           |                            | capture mode allows events           | (span_metric_event) per rules above          |
+----------------------+---------------------------+----------------------------+--------------------------------------+----------------------------------------------+
| ToolCall             | Yes (tool {name})         | Duration only              | No (explicitly excluded)             | Never (arguments already attributes)         |
+----------------------+---------------------------+----------------------------+--------------------------------------+----------------------------------------------+
| EmbeddingInvocation  | Yes (embedding {model})   | None                       | No (explicitly excluded)             | Never (vectors not recorded)                 |
+----------------------+---------------------------+----------------------------+--------------------------------------+----------------------------------------------+
| Evaluation (LLM only)| Optional spans (aggregated| Histogram (score)          | Single event ``gen_ai.evaluations``  | N/A (evaluation items separate structure)    |
|                      | or per-metric)            |                            |                                      |                                              |
+----------------------+---------------------------+----------------------------+--------------------------------------+----------------------------------------------+

Content Events Exclusions
-------------------------
* ToolCall invocations: excluded to avoid duplicating argument payloads and unbounded log growth.
* Embedding invocations: excluded to prevent large vector/text payload emission of limited diagnostic value.

Evaluation
----------
Evaluation runs occur post ``stop_llm`` (or after error) when enabled. Each evaluator produces one or more ``EvaluationResult`` items; results are:

1. Recorded into a histogram metric ``gen_ai.evaluation.score`` (for numeric scores in [0,1] when meaningful).
2. Emitted as a single structured event ``gen_ai.evaluations`` containing a list of evaluation objects.
3. Optionally represented as spans (``aggregated`` or ``per_metric``) depending on span mode.

Environment variables controlling evaluation are listed below. Currently evaluations apply only to ``LLMInvocation``.

Quick Usage Examples
--------------------
LLM Invocation
~~~~~~~~~~~~~~
.. code-block:: python

   from opentelemetry.util.genai.handler import get_telemetry_handler
   from opentelemetry.util.genai.types import (
       LLMInvocation, InputMessage, OutputMessage, Text
   )

   handler = get_telemetry_handler()
   inv = LLMInvocation(
       request_model="gpt-4o-mini",
       provider="openai",
       input_messages=[InputMessage(role="user", parts=[Text(content="Hello!")])],
       attributes={"framework": "fastapi"},
   )
   handler.start_llm(inv)
   # ... call provider ...
   inv.output_messages = [OutputMessage(role="assistant", parts=[Text(content="Hi there!")], finish_reason="stop")]
   inv.input_tokens = 12
   inv.output_tokens = 20
   handler.stop_llm(inv)

ToolCall Invocation
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from opentelemetry.util.genai.types import ToolCall

   tool = ToolCall(name="translate", id="t1", arguments={"text": "Hola"}, provider="demo")
   handler.start_tool_call(tool)
   # ... execute tool ...
   tool.attributes["result"] = "Hello"
   handler.stop_tool_call(tool)

Embedding Invocation
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from opentelemetry.util.genai.types import EmbeddingInvocation

   emb = EmbeddingInvocation(request_model="text-emb-v1", provider="demo", input_texts=["banana", "apple"])
   handler.start_embedding(emb)
   # ... embedding generation ...
   handler.stop_embedding(emb)

Evaluation Example
~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from opentelemetry.util.genai.handler import get_telemetry_handler
   handler = get_telemetry_handler()
   # after LLM invocation completed
   results = handler.evaluate_llm(inv)
   for r in results:
       print(r.metric_name, r.score, r.label)

Environment Variables
---------------------
Required for experimental GenAI semantic conventions (and content capture):

* ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``

GenAI utilities configuration:

* ``OTEL_INSTRUMENTATION_GENAI_EMITTERS`` – telemetry flavor & extras (``span`` | ``span_metric`` | ``span_metric_event`` plus optional ``traceloop_compat``). (Replaces older docs referencing ``OTEL_INSTRUMENTATION_GENAI_GENERATOR``.)
* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` – content capture mode (``NO_CONTENT`` | ``SPAN_ONLY`` | ``EVENT_ONLY`` | ``SPAN_AND_EVENT``).
* ``OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE`` – enable evaluations (true/false).
* ``OTEL_INSTRUMENTATION_GENAI_EVALUATORS`` – comma list of evaluator names (e.g. ``deepeval,length``).
* ``OTEL_INSTRUMENTATION_GENAI_EVALUATION_SPAN_MODE`` – ``off`` | ``aggregated`` | ``per_metric``.
* ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_HOOK`` – optional fully qualified function path for custom upload hook.
* ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` – base fsspec path for prompt/response storage.

Extensibility
-------------
Adding a New Emitter
~~~~~~~~~~~~~~~~~~~~
Emitters implement the trio ``start(obj)``, ``finish(obj)``, ``error(err, obj)`` (and optionally ``handles(obj)`` to filter objects). Example skeleton:

.. code-block:: python

   from opentelemetry.util.genai.types import LLMInvocation, Error

   class CustomEmitter:
       role = "custom"
       def start(self, obj):
           if isinstance(obj, LLMInvocation):
               ...
       def finish(self, obj):
           ...
       def error(self, err: Error, obj):
           ...
       def handles(self, obj):  # optional
           return isinstance(obj, LLMInvocation)

Integrate by creating a custom handler instance assembling emitters into a new ``CompositeGenerator``.

Adding an Evaluator
~~~~~~~~~~~~~~~~~~~
Implement the ``Evaluator`` interface (see ``evaluators/base.py``), register via ``register_evaluator(name, factory)``.

Dynamic Loading Notes:
* ``deepeval``: If an external integration package exposes ``opentelemetry.util.genai.evals.deepeval.DeepEvalEvaluator`` it will override the lightweight builtin placeholder automatically.
* Builtin evaluators (``length``, ``sentiment``) are registered lazily; they will be imported only if referenced.

Custom Evaluation Emission
~~~~~~~~~~~~~~~~~~~~~~~~~~
You can replace or supplement evaluation emission by constructing your own ``EvaluationManager`` (mirroring how emitters are composed) and attaching it to a custom handler instance before use.

Troubleshooting
---------------
* Missing message content: confirm experimental opt‑in + capture mode, and flavor rules (see matrix).
* No spans exported: ensure a global TracerProvider is configured prior to handler creation.
* Evaluations return empty: either disabled (env) or no evaluator names resolved.

Stability Disclaimer
--------------------
GenAI semantic conventions are incubating; attribute names and enabling conditions can change. Track CHANGELOG for updates.

Roadmap (Indicative)
--------------------
* Additional evaluation domain coverage (embeddings, tool calls)
* More granular token metrics (streaming / incremental)
* Potential redaction utilities for sensitive content
* Attribute stabilization & alignment with future semconv releases

Minimal End-to-End Test Snippet
--------------------------------
.. code-block:: python

   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import SimpleSpanProcessor, InMemorySpanExporter
   from opentelemetry import trace

   exporter = InMemorySpanExporter()
   provider = TracerProvider()
   provider.add_span_processor(SimpleSpanProcessor(exporter))
   trace.set_tracer_provider(provider)

   from opentelemetry.util.genai.handler import get_telemetry_handler
   from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text

   handler = get_telemetry_handler()
   inv = LLMInvocation(
       request_model="demo-model",
       provider="demo-provider",
       input_messages=[InputMessage(role="user", parts=[Text(content="ping")])],
   )
   handler.start_llm(inv)
   inv.output_messages = [OutputMessage(role="assistant", parts=[Text(content="pong")], finish_reason="stop")]
   handler.stop_llm(inv)
   spans = exporter.get_finished_spans()
   assert spans and spans[0].name == "chat demo-model"

License
-------
See repository LICENSE (Apache 2.0 unless otherwise stated).
