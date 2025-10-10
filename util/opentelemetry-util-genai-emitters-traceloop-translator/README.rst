OpenTelemetry GenAI Traceloop Translator Emitter
===============================================

This package provides an emitter that promotes legacy ``traceloop.*`` attributes
on GenAI LLM invocations into semantic convention (``gen_ai.*``) attributes.

Installation
------------

.. code-block:: bash

   pip install opentelemetry-util-genai-emitters-traceloop-translator

Usage
-----

Enable the translator (and any other emitters) via the environment:

.. code-block:: bash

   export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span,traceloop_translator"

Example
-------

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
           "traceloop.callback.name": "root_chain",
       },
   )
   handler.start_llm(inv)
   handler.stop_llm(inv)
   # inv.attributes now include gen_ai.workflow.name, gen_ai.agent.name, gen_ai.callback.name, etc.

Tests
-----

.. code-block:: bash

   pytest util/opentelemetry-util-genai-emitters-traceloop-translator/tests

