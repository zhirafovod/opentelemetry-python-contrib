OpenTelemetry GenAI Traceloop Translator Emitter
===============================================

This optional emitter promotes legacy ``traceloop.*`` attributes attached to an ``LLMInvocation`` into
GenAI semantic convention (or forward-looking custom ``gen_ai.*``) attributes *before* the standard
semantic span emitter runs. It does **not** create or end spans; it only rewrites / adds attributes on
the invocation (and, if present, sets them on the existing span).

Why
---
If upstream code (or the Traceloop compatibility emitter) still produces ``traceloop.*`` keys but you want
downstream dashboards to rely on the GenAI semantic conventions, enabling this translator lets you migrate
incrementally without rewriting the upstream instrumentation immediately.

Behavior
--------
The translator runs on ``on_start`` and ``on_end`` of an ``LLMInvocation``. It scans ``invocation.attributes``
and adds mapped keys (never overwriting existing ``gen_ai.*``). Content normalization (input/output messages)
is gated by the ``OTEL_GENAI_CONTENT_CAPTURE`` env var.

Mapping Table
-------------

============================== ================================ 
Traceloop Key                  Added Key                        
============================== ================================
``traceloop.workflow.name``    ``gen_ai.workflow.name``
``traceloop.entity.name``      ``gen_ai.agent.name``
``traceloop.entity.path``      ``gen_ai.workflow.path``
``traceloop.prompt.managed``   ``gen_ai.prompt.managed``
``traceloop.prompt.key``       ``gen_ai.prompt.key``
``traceloop.prompt.version``   ``gen_ai.prompt.version``
``traceloop.prompt.version_name`` ``gen_ai.prompt.version_name``
``traceloop.prompt.version_hash`` ``gen_ai.prompt.version_hash``
``traceloop.prompt.template``  ``gen_ai.prompt.template``
``traceloop.prompt.template_variables`` ``gen_ai.prompt.template_variables`` 
``traceloop.correlation.id``   ``gen_ai.conversation.id`
``traceloop.entity.input``     ``gen_ai.input.messages``
``traceloop.entity.output``    ``gen_ai.output.messages``
============================== ================================ ============================================

Note: Legacy callback fields like ``traceloop.callback.name`` / ``traceloop.callback.id`` are **not** currently
mapped. Add them to the mapping table if/when needed.

Legacy Stripping
----------------
By default, successfully mapped legacy keys are removed after translation to reduce attribute duplication.
Control via environment:

``OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY``
  * ``true`` / unset: strip mapped ``traceloop.*`` keys
  * ``false`` / ``0``: retain both legacy and new keys

Only keys that were actually mapped/normalized are stripped to avoid accidental data loss.

Environment Flags
-----------------
``OTEL_INSTRUMENTATION_GENAI_EMITTERS``: Include ``traceloop_translator`` alongside ``span`` (e.g. ``span,traceloop_translator``)
``OTEL_GENAI_CONTENT_CAPTURE``: Enable input/output content mapping ("1" to enable; "0" disables)
``OTEL_GENAI_MAP_CORRELATION_TO_CONVERSATION``: Toggle correlation → conversation id mapping
``OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY``: Toggle legacy key removal (see above)

Example
-------
.. code-block:: bash

   export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span,traceloop_translator"
   export OTEL_GENAI_CONTENT_CAPTURE="1"

.. code-block:: python

   from opentelemetry.util.genai.handler import get_telemetry_handler
   from opentelemetry.util.genai.types import LLMInvocation, InputMessage, Text

   handler = get_telemetry_handler()
   inv = LLMInvocation(
       request_model="gpt-4",
       input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
       attributes={
           "traceloop.workflow.name": "flowA",
           "traceloop.entity.name": "AgentA",
           "traceloop.entity.input": ["Hello"],  # various shapes normalized
       },
   )
   handler.start_llm(inv)
   inv.output_messages = []  # add output content if desired
   handler.stop_llm(inv)
   # inv.attributes now include gen_ai.workflow.name, gen_ai.agent.name, gen_ai.input.messages (legacy removed if strip enabled)

Installation
------------
.. code-block:: bash

   pip install opentelemetry-util-genai-traceloop-translator

Tests
-----
.. code-block:: bash

   pytest util/opentelemetry-util-genai-traceloop-translator/tests

