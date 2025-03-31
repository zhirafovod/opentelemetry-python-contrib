from langchain_core.messages import HumanMessage, SystemMessage

from opentelemetry.instrumentation.langchain import LangChainInstrumentor


def main():

    # todo: start a server span here

    # Set up instrumentation
    LangChainInstrumentor().instrument()

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]

    result = llm.invoke(messages).content
    print("LLM output:\n", result)

if __name__ == "__main__":
    main()
