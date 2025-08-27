import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.langchain import LangChainInstrumentor

from opentelemetry import _events, _logs, trace, metrics
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# configure logging and events
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
_events.set_event_logger_provider(EventLoggerProvider())

def main():
    import random
    
    # List of capital questions to randomly select from
    capital_questions = [
        "What is the capital of France?",
        "What is the capital of Germany?", 
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of United Kingdom?",
        "What is the capital of Japan?",
        "What is the capital of Canada?",
        "What is the capital of Australia?",
        "What is the capital of Brazil?",
        "What is the capital of India?",
        "What is the capital of United States?"
    ]
    
    # Set up instrumentation once
    LangChainInstrumentor().instrument()

    api_key = os.getenv("OPENAI_API_KEY")

    # ChatOpenAI setup
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_sequences=["\n", "Human:", "AI:"],
        seed=100,
        api_key='dummy-key',
        base_url='https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini',
        default_headers={
            "api-key": api_key
        },
        model_kwargs={
            "user": '{"appkey": "<key>"}'
        }
    )
    
    # Statistics tracking
    correct_instructions = 0
    wrong_instructions = 0
    
    # Run 100 iterations
    for i in range(1, 11):
        print(f"\n--- Run {i}/100 ---")
        
        selected_question = random.choice(capital_questions)
        print(f"Selected question: {selected_question}")
        
        give_wrong_answer = random.random() < 0.3
        
        if give_wrong_answer:
            system_message = "You are a helpful assistant, but sometimes you make mistakes about geography. When asked about capitals, occasionally give the wrong city name."
            print("🔴 Instructed to give wrong answer")
            wrong_instructions += 1
        else:
            system_message = "You are a helpful assistant!"
            print("🟢 Instructed to give correct answer")
            correct_instructions += 1

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=selected_question),
        ]

        result = llm.invoke(messages)
        print(f"LLM output: {result.content}")
        
        # Add delay to avoid rate limiting (1 second between requests)
        import time
        time.sleep(1)
        
        # Show progress every 10 runs
        if i % 10 == 0:
            print(f"Progress: {i}/100 completed. Correct: {correct_instructions}, Wrong: {wrong_instructions}")
    
    # Final statistics
    print(f"\n=== Final Statistics ===")
    print(f"Total runs: 100")
    print(f"Correct instructions: {correct_instructions} ({correct_instructions}%)")
    print(f"Wrong instructions: {wrong_instructions} ({wrong_instructions}%)")

    # Un-instrument after use
    LangChainInstrumentor().uninstrument()

if __name__ == "__main__":
    main()