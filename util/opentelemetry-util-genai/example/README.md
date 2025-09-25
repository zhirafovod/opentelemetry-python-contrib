# Minimal OpenTelemetry GenAI Util Example

This proof-of-concept demonstrates how to use the core utilities from `opentelemetry-util-genai` to instrument a simple GenAI (LLM) workflow, without any framework complexity.

## Files
- `simple_llm.py`: A mock LLM class that simulates a model call.
- `instrumented_example.py`: Shows how to use GenAI types, telemetry handler, and generator for tracing LLM invocations.
- `requirements.txt`: Minimal dependencies.

## Usage
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the example (success):
   ```sh
   python instrumented_example.py "Hello, world!"
   ```
3. Run the example (error case):
   ```sh
   python instrumented_example.py "fail this call"
   ```

## What it demonstrates
- Creating GenAI data types (`InputMessage`, `OutputMessage`, `LLMInvocation`)
- Using `TelemetryHandler` for lifecycle management
- Emitting spans/metrics/events with `SemConvGenerator`
- Error handling and evaluation hooks

This POC is under 100 lines and is easy to adapt for your own GenAI applications.
