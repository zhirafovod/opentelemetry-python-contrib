# Translator

## Automatic Span Processing (Recommended)

Add `TraceloopSpanProcessor` to your TracerProvider to automatically transform all matching spans:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.genai.processors import TraceloopSpanProcessor

# Set up tracer provider
provider = TracerProvider()

# Add processor - transforms all matching spans automatically
processor = TraceloopSpanProcessor(
    attribute_transformations={
        "remove": ["debug_info"],
        "rename": {"model_ver": "llm.model.version"}, 
        "add": {"service.name": "my-llm"}
    },
    name_transformations={"chat *": "llm.openai.chat"},
    traceloop_attributes={
        "traceloop.entity.name": "MyLLMEntity"
    }
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

```

## Transformation Rules

### Attributes
- **Remove**: `"remove": ["field1", "field2"]`
- **Rename**: `"rename": {"old_name": "new_name"}`  
- **Add**: `"add": {"key": "value"}`

### Span Names
- **Direct**: `"old name": "new name"`
- **Pattern**: `"chat *": "llm.chat"` (wildcard matching)