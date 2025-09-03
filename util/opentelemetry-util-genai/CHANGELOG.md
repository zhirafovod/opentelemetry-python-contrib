# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Repurpose the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable when GEN AI stability mode is set to `gen_ai_latest_experimental`,
to take on an enum (`NO_CONTENT/SPAN_ONLY/EVENT_ONLY/SPAN_AND_EVENT`) instead of a boolean. Add a utility function to help parse this environment variable.

## Unreleased

- Add upload hook to genai utils to implement semconv v1.37.

  The hook uses [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) to support
  various pluggable backends.
  ([#3752](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3752))
  ([#3759](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3752))
  ([#3763](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3763))
- Add a utility to parse the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable.
  Add `gen_ai_latest_experimental` as a new value to the Sem Conv stability flag ([#3716](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3716)).
- Generate Spans for LLM invocations
- Generate Metrics for LLM invocations
- Generate Logs for LLM invocations
- Helper functions for starting and finishing LLM invocations
