"""
Custom generator that extends SemConvGenerator and adds extra attributes after stop.
"""

from opentelemetry.util.genai.generators.semconv_generator import (
    SemConvGenerator,
)


class CustomSemConvGenerator(SemConvGenerator):
    def _on_after_stop(self, data):
        # Add arbitrary attributes to the span after stop
        if hasattr(data, "span") and data.span is not None:
            data.span.set_attribute("custom.attribute", "custom_value")
            data.span.set_attribute("custom.flag", True)
        # Call super in case future logic is added
        super()._on_after_stop(data)
