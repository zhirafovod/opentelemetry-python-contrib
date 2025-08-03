import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from opentelemetry.genai.sdk.decorators import llm

# Load environment variables from .env file
load_dotenv()

@llm(name="invoke_langchain_model")
def invoke_model(messages):
    # Get API key from environment variable or set a placeholder
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    result = llm.invoke(messages)
    return result

def main():

    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]

    result = invoke_model(messages)
    print("LLM output:\n", result)

if __name__ == "__main__":
    main()
