from opentelemetry.util.genai.generators import SpanEmitter, SpanGenerator


def test_span_generator_alias_is_span_emitter():
    # Alias should point to the same class object
    assert SpanGenerator is SpanEmitter

