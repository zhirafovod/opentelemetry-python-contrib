import os

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings

from opentelemetry.instrumentation.langchain import LangChainInstrumentor



def main():

    # todo: start a server span here
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
    # Check if TracerProvider is already set to avoid "Overriding of current TracerProvider is not allowed" error
    try:
        current_provider = trace.get_tracer_provider()
        # If we get a NoOpTracerProvider, we need to set a real one
        if current_provider.__class__.__name__ == 'NoOpTracerProvider':
            trace.set_tracer_provider(TracerProvider())
    except Exception:
        # If there's any issue, set a new TracerProvider
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

    # Set up instrumentation
    LangChainInstrumentor().instrument()

    # Make sure to `pip install openai` first
    # from openai import OpenAI
    # client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    #
    # def get_embedding(text, model="model-identifier"):
    #     text = text.replace("\n", " ")
    #     return client.embeddings.create(input=[text], model=model).data[0].embedding
    #
    # print(get_embedding("Hello from openai"))
    #


    # ======== OpenAI Embeddings ========
    # model = OpenAIEmbeddings(check_embedding_ctx_length=False,
    #                          openai_api_key="lm-studio",
    #                          base_url="http://localhost:1234/v1",
    #                          model="nomic-ai/nomic-embed-text-v1.5-GGUF",
    #                          )
    #
    # response = model.embed_query("Hello world")
    # print(response)


    # ======== Azure OpenAI Embeddings ========
    endpoint = "https://etser-mf7gfr7m-eastus2.cognitiveservices.azure.com/"
    deployment = "text-embedding-3-large"


    client = AzureOpenAIEmbeddings(  # or "2023-05-15" if that's your API version
        model=deployment,
        azure_endpoint=endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version="2024-12-01-preview",
    )

    # Single query embedding
    # response = client.embed_query("Hello from Azure")
    # print(response)

    # Batch embedding multiple documents
    documents = [
        "This is the first document.",
        "Here's another document to embed.",
        "And a third document for good measure."
    ]
    batch_response = client.embed_documents(documents)
    print(batch_response)

    # ======== HuggingFace Embeddings ========
    # from langchain_huggingface import HuggingFaceEmbeddings
    #
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )
    #
    # response = embeddings.embed_query("Hello, From HuggingFace")
    #
    # print(response)

    # ======== AWS bedrock ========
    # from langchain_aws import BedrockEmbeddings
    #
    # embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    #
    # response = embeddings_model.embed_query("Hello, From AWS bedrock")
    # print(response)

    LangChainInstrumentor().uninstrument()

if __name__ == "__main__":
    main()