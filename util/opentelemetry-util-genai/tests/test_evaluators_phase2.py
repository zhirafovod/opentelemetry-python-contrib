# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from unittest.mock import patch

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATORS,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


class TestEvaluationPhase2(unittest.TestCase):
    def setUp(self):
        # Reset handler singleton for isolation
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")

        # Fresh invocation
        self.invocation = LLMInvocation(
            request_model="model-y", provider="prov"
        )
        self.invocation.input_messages.append(
            InputMessage(
                role="user", parts=[Text(content="Tell me something short")]
            )
        )
        self.invocation.output_messages.append(
            OutputMessage(
                role="assistant",
                parts=[Text(content="Hello world!")],
                finish_reason="stop",
            )
        )

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE: "true",
            OTEL_INSTRUMENTATION_GENAI_EVALUATORS: "length",
        },
        clear=True,
    )
    def test_length_evaluator_emits_event_and_metric(self):
        handler = get_telemetry_handler()

        # Patch histogram and event emit to capture calls
        recorded = {"metrics": [], "events": []}

        original_hist = handler._evaluation_histogram  # pylint: disable=protected-access

        def fake_record(value, attributes=None):  # noqa: D401
            recorded["metrics"].append((value, dict(attributes or {})))

        original_emit = handler._event_logger.emit  # pylint: disable=protected-access

        def fake_emit(event):  # noqa: D401
            recorded["events"].append(event)

        handler._evaluation_histogram.record = fake_record  # type: ignore
        handler._event_logger.emit = fake_emit  # type: ignore

        results = handler.evaluate_llm(self.invocation)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.metric_name, "length")
        self.assertIsNotNone(res.score)
        # Metrics captured
        self.assertEqual(len(recorded["metrics"]), 1)
        metric_val, metric_attrs = recorded["metrics"][0]
        self.assertAlmostEqual(metric_val, res.score)
        self.assertEqual(
            metric_attrs.get("gen_ai.evaluation.name"), "length"
        )  # attribute enriched
        # Event captured
        self.assertEqual(len(recorded["events"]), 1)
        evt = recorded["events"][0]
        self.assertEqual(evt.name, "gen_ai.evaluations")
        self.assertIn("evaluations", evt.body)
        body_list = evt.body["evaluations"]
        self.assertEqual(len(body_list), 1)
        body_item = body_list[0]
        self.assertEqual(body_item["gen_ai.evaluation.name"], "length")
        if hasattr(
            handler._evaluation_histogram, "record"
        ):  # restore originals
            handler._evaluation_histogram = original_hist  # type: ignore
        handler._event_logger.emit = original_emit  # type: ignore

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE: "true",
            OTEL_INSTRUMENTATION_GENAI_EVALUATORS: "deepeval",
        },
        clear=True,
    )
    def test_deepeval_missing_dependency_error_event(self):
        handler = get_telemetry_handler()
        recorded = {"events": []}
        original_emit = handler._event_logger.emit  # pylint: disable=protected-access

        def fake_emit(event):
            recorded["events"].append(event)

        handler._event_logger.emit = fake_emit  # type: ignore
        results = handler.evaluate_llm(self.invocation)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.metric_name, "deepeval")
        self.assertIsNotNone(res.error)
        # Event item should contain error.type
        self.assertEqual(len(recorded["events"]), 1)
        evt = recorded["events"][0]
        body_list = evt.body["evaluations"]
        self.assertEqual(body_list[0]["gen_ai.evaluation.name"], "deepeval")
        self.assertIn("error.type", body_list[0])
        handler._event_logger.emit = original_emit  # restore


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
